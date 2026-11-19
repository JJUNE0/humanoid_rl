"""Contact-aware inverse-dynamics retargeting of the CMU 88_01 backflip.

All poses and sole contact forces are optimized together. The floating root
has NO actuator. Newton--Euler residuals therefore enforce linear/angular
momentum balance, including at touchdown. Separate nonnegative contact
impulses are permitted only at scheduled touchdown nodes. Motor effort and
friction limits are screened independently of the animation objectives.

This is a 20 ms collocation approximation, not a MuJoCo tracking certificate.
An unsuccessful solve must not be exported as a training-ready reference.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

import optimize_humanoid_light_cmu_landing as kin
import retarget_humanoid_light_cmu as cmu
import retarget_humanoid_light_lafan as ik

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/humanoid_light_cmu_dynamics'
sys.path.insert(0, str(ROOT / 'src'))
from humanoid_retarget.datasets.cmu import load_asfamc


def velocity_acceleration(q, dt, base_rotation=None, parameters=None, jacobian=False):
    """Centered Lie-group differences, expressed at the CURRENT root frame.

    The neighboring rotation logs are transported implicitly by using R_t^T
    R_neighbor. Differencing angular velocities expressed in different body
    frames directly would give an incorrect acceleration.
    """
    n = len(q)
    rotations = R.from_quat(q[:, 3:7], scalar_first=True).as_matrix()
    v, a = np.zeros((n, 32)), np.zeros((n, 32))
    blocks_v, blocks_a = [], []
    coords = np.c_[q[:, :3], np.zeros((n, 3)), q[:, 7:]]
    chains = np.tile(np.eye(32), (n, 1, 1))
    if parameters is not None:
        for t in range(n):
            chains[t, 3:6, 3:6] = kin.right_jacobian(parameters[t, 3:6])
    for t in range(n):
        prev, nxt = max(t-1, 0), min(t+1, n-1)
        forward = rotations[t].T @ rotations[nxt]
        backward = rotations[t].T @ rotations[prev]
        f, b = R.from_matrix(forward).as_rotvec(), R.from_matrix(backward).as_rotvec()
        v[t] = (coords[nxt] - coords[prev]) / (2*dt)
        a[t] = (coords[nxt] - 2*coords[t] + coords[prev]) / dt**2
        v[t, 3:6], a[t, 3:6] = (f-b)/(2*dt), (f+b)/dt**2
        if jacobian:
            dv = [np.eye(32)*(-.5/dt), np.zeros((32, 32)), np.eye(32)*(.5/dt)]
            da = [np.eye(32)/dt**2, -2*np.eye(32)/dt**2, np.eye(32)/dt**2]
            jf, jb = kin.right_jacobian_inverse(f), kin.right_jacobian_inverse(b)
            dv[0][3:6, 3:6] = -jb/(2*dt)
            dv[1][3:6, 3:6] = (-jf@forward.T+jb@backward.T)/(2*dt)
            dv[2][3:6, 3:6] = jf/(2*dt)
            da[0][3:6, 3:6] = jb/dt**2
            da[1][3:6, 3:6] = (-jf@forward.T-jb@backward.T)/dt**2
            da[2][3:6, 3:6] = jf/dt**2
            blocks_v.append([(j, block@chains[j]) for j, block in zip([prev, t, nxt], dv)])
            blocks_a.append([(j, block@chains[j]) for j, block in zip([prev, t, nxt], da)])
    return v, a, chains, blocks_v, blocks_a


def sole_points(model):
    """Four points INSIDE each retained sole hull, in its foot body frame."""
    _, feet, ids, _, positions, rotations = kin.foot_geometry(model)
    local = []
    for side, p, rot in zip(['left', 'right'], positions, rotations):
        vertices = feet[side][1]
        world = vertices@rot.T+p
        low = world[world[:, 2] < world[:, 2].min()+.004]
        # Convex combinations of sole vertices stay inside the footprint.
        chosen = [low[np.argmax(low[:, :2]@np.array(direction))] for direction in
                  [[1, 1], [1, -1], [-1, 1], [-1, -1]]]
        chosen = np.array(chosen)
        chosen[:, 2] = world[:, 2].min()
        local.append((chosen-p)@rot)
    return ids, np.array(local)


def strict_clip(values, lower, upper):
    padding = np.minimum(1e-10, .01*(upper-lower))
    return np.minimum(np.maximum(values, lower+padding), upper-padding)


def make_plan(model, motion):
    root, rotations, _, _, flight, human, names, times, scale = cmu.targets(motion, model, 50.)
    plan = kin.make_plan(model, times, root, rotations['pelvis_link'], human, names, flight)
    old = np.load(kin.OUT/'88_01_optimized.npz', allow_pickle=False)['qpos'].astype(float)
    # The source has a second small airborne interval. Estimate it from BOTH
    # rigid-foot landmarks, not the toe-base height alone (toe flex is present).
    source_time = np.arange(len(motion.frame_ids))/motion.fps
    heights = []
    for side in ['l', 'r']:
        values = np.stack([np.interp(times, source_time, motion.positions[:, motion.names.index(side+part), 2])
                           for part in ['foot', 'toes']], axis=1)
        floor = np.quantile(values, .05, axis=0)
        heights.append(np.min(values-floor, axis=1))
    clearance = np.min(heights, axis=0)
    candidates = ik.contact_intervals((clearance > .035) & (times > 1.45) & (times < 2.1))
    first, end = max(candidates, key=lambda span: span[1]-span[0])
    # Height-threshold crossings occur above ground. Extrapolate toward the
    # neighboring low-foot frames, keeping conservative source timing bounds.
    rebound_off = float(times[max(first-2, round(1.48*50))])
    rebound_land = float(times[min(end+2, round(1.92*50))])
    td = plan['metadata']['touchdown_seconds']
    p1 = np.asarray(plan['metadata']['landing_foot_positions'])
    p2 = p1.copy()
    ankle_ids = [names.index(side+'Foot') for side in ['Left', 'Right']]
    shift = (human[round(rebound_land*50), ankle_ids, :2].mean(axis=0)
             - human[round(td*50), ankle_ids, :2].mean(axis=0))*scale
    p2[:, :2] += shift
    _, _, _, _, _, r0 = kin.foot_geometry(model)
    q1 = kin.ground_pose(model, p1, r0, .55)
    q2 = kin.ground_pose(model, p2, r0, .7, initial=q1)
    qstand = kin.ground_pose(model, p2, r0, 0., initial=q2)
    qref = old.copy()
    com = []
    data = mujoco.MjData(model)
    def get_com(q):
        data.qpos[:] = q
        mujoco.mj_fwdPosition(model, data)
        return data.subtree_com[1].copy()
    c1, c2 = get_com(q1), get_com(q2)
    duration = rebound_land-rebound_off
    vlaunch = (c2-c1)/duration
    vlaunch[2] += .5*9.81*duration
    for i, seconds in enumerate(times):
        if seconds < td:
            pass
        elif seconds < rebound_off:
            u = kin.smoothstep((seconds-td)/(rebound_off-td))
            qref[i, :3] = (1-u)*old[round(td*50), :3]+u*q1[:3]
            qref[i, 7:] = (1-u)*old[round(td*50), 7:]+u*q1[7:]
        elif seconds < rebound_land:
            u = (seconds-rebound_off)/duration
            qref[i, :3] = (1-u)*q1[:3]+u*q2[:3]
            qref[i, 3:7] = [1, 0, 0, 0]
            qref[i, 7:] = (1-u)*q1[7:]+u*q2[7:]
            desired = c1+vlaunch*(seconds-rebound_off)+[0, 0, -.5*9.81*(seconds-rebound_off)**2]
            qref[i, :3] += desired-get_com(qref[i])
        else:
            u = kin.smoothstep((seconds-rebound_land)/.65)
            qref[i, :3] = (1-u)*q2[:3]+u*qstand[:3]
            qref[i, 7:] = (1-u)*q2[7:]+u*qstand[7:]
            qref[i, 3:7] = [1, 0, 0, 0]
        com.append(get_com(qref[i]))
    airborne = ((times > plan['metadata']['takeoff_seconds']) & (times < td)) | ((times > rebound_off) & (times < rebound_land))
    plan['contacts'] = np.tile((~airborne)[:, None], (1, 2))
    plan['foot_pos'][times >= rebound_land] = p2
    plan['com'] = np.array(com)
    plan['root'] = qref[:, :3].copy()
    plan['joint_reference'] = qref[:, 7:].copy()
    # Weaker nonphysical posture priors are enough here: the free root dynamics
    # determine how quickly the robot actually rotates/loses momentum.
    plan['metadata'].update(
        post_landing_contact='first support, source-informed rebound flight, second support',
        rebound_takeoff_seconds=rebound_off, rebound_touchdown_seconds=rebound_land,
        rebound_landing_foot_positions=p2.tolist(),
        momentum_method='full-body floating-base inverse dynamics; no root actuator',
        impulse_method='entire contact reaction in each 20 ms touchdown cell is assigned to a unilateral impulse; continuous-force part is zero in those cells',
        kinematic_only=False, dynamic_feasibility_verified=False,
        contact_schedule_is_heuristic=True, contact_model='rigid sole point forces; Coulomb pyramid mu=0.7',
        discretization_seconds=.02, recovery_end_seconds=rebound_land+.65,
        root_orientation='soft full-turn prior; root rotation is allowed to change under momentum balance',
        airborne_com='soft COM prior plus explicit floating-base Newton--Euler balance',
        foot_contact='load-weighted sole point complementarity; heel/toe rocking permitted')
    return plan, qref, human, names, scale


class DynamicsProblem:
    def __init__(self, model, plan, weight=100.):
        self.model, self.plan, self.weight = model, plan, weight
        self.n = len(plan['times'])
        self.dt = .02
        animation_plan = copy.deepcopy(plan)
        times = plan['times']
        # Flat ankle frames are inappropriate during toe-off and first
        # impact. Loaded SOLE POINTS constrain contact in these transitions.
        for lo, hi in plan.get('transition_intervals',[(.60, .74), (1.34, 1.58), (1.84, 2.04)]):
            animation_plan['contacts'][(times >= lo)&(times <= hi)] = False
        self.kin = kin.TrajectoryProblem(model, animation_plan)
        # A COM support-polygon condition is a static balance heuristic; it
        # must not prevent dynamic landing/capture movements.
        self.kin.hull[:] = 0.
        self.kin.hull[:, 2] = -1.
        self.kin.smoothing *= .25
        spec = mujoco.MjSpec.from_file(str(ROOT/'outputs/humanoid_light_conversion/scene.xml'))
        spec.add_sensor(name='retarget_total_angular_momentum', type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
                        objtype=mujoco.mjtObj.mjOBJ_BODY, objname='pelvis_link')
        self.dyn_model = spec.compile()
        np.testing.assert_allclose(self.dyn_model.body_mass, model.body_mass, atol=0, rtol=0)
        np.testing.assert_allclose(self.dyn_model.body_inertia, model.body_inertia, atol=0, rtol=0)
        self.dyn_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        self.data = mujoco.MjData(self.dyn_model)
        self.mass = float(model.body_mass.sum())
        self.bw = self.mass*9.81
        self.feet, self.points = sole_points(model)
        self.active = [t for t in range(1, self.n-1) if plan['contacts'][t].any()]
        self.impacts = [int(np.argmin(abs(plan['times']-plan['metadata'][key]))) for key in ['touchdown_seconds', 'rebound_touchdown_seconds']]
        self.force_index = {t: i for i, t in enumerate(self.active)}
        self.impulse_index = {t: len(self.active)+i for i, t in enumerate(self.impacts)}
        self.nq = self.n*32
        self.nw = (len(self.active)+len(self.impacts))*24
        self.size = self.nq+self.nw
        self.effort = np.zeros(26)
        for actuator in range(model.nu):
            joint = model.actuator_trnid[actuator, 0]
            self.effort[model.jnt_dofadr[joint]-6] = model.actuator_ctrlrange[actuator, 1]
        self.cache_x = None
        self.calls = 0
        self.started = time.perf_counter()

    def unpack(self, z):
        x = z[:self.nq].reshape(self.n, 32)
        q = kin.parameters_to_poses(x, self.plan['rotation'])
        f, impulses = np.zeros((self.n, 8, 3)), np.zeros((self.n, 8, 3))
        for t, index in self.force_index.items():
            f[t] = z[self.nq+index*24:self.nq+(index+1)*24].reshape(8, 3)*self.bw
        for t, index in self.impulse_index.items():
            impulses[t] = z[self.nq+index*24:self.nq+(index+1)*24].reshape(8, 3)*self.bw*self.dt
        return x, q, f, impulses

    def force_matrix(self, data):
        matrices, points = [], []
        for side, body in enumerate(self.feet):
            rot = data.xmat[body].reshape(3, 3)
            jp, jr = np.empty((3, 32)), np.empty((3, 32))
            mujoco.mj_jacBody(self.dyn_model, data, jp, jr, body)
            for point in self.points[side]:
                arm = rot@point
                matrices.append((jp-kin.skew(arm)@jr).T)
                points.append(data.xpos[body]+arm)
        return np.concatenate(matrices, axis=1), np.array(points)

    def initial(self, q):
        x = kin.poses_to_parameters(q, self.plan['rotation']).ravel()
        w = np.zeros((self.nw//3, 3))
        for t, index in self.force_index.items():
            w[index*8:(index+1)*8, 2] = 1/8
        z = np.r_[x, w.ravel()]
        lo = np.r_[np.tile(np.r_[[-2., -1., .25], np.full(3, -1.4), self.model.jnt_range[1:, 0]+1e-5], self.n), np.tile([-20., -20., 0.], self.nw//3)]
        hi = np.r_[np.tile(np.r_[[1., 1., 2.3], np.full(3, 1.4), self.model.jnt_range[1:, 1]-1e-5], self.n), np.tile([20., 20., 20.], self.nw//3)]
        for t in self.impacts:
            index = self.force_index[t]
            start = self.nq+index*24
            lo[start:start+24] = np.tile([-1e-10, -1e-10, 0.], 8)
            hi[start:start+24] = 1e-10
        return strict_clip(self.canonical_impulses(z), lo, hi), lo, hi

    def canonical_impulses(self, z):
        """Avoid an unidentifiable force/impulse split at a touchdown node.

        Both columns enter the finite-cell momentum balance identically.
        Assign ALL first-cell ground reaction to its integrated impulse.
        This does not change the poses or any total generalized reaction.
        """
        z = z.copy()
        for t in self.impacts:
            force = self.nq+self.force_index[t]*24
            impulse = self.nq+self.impulse_index[t]*24
            z[impulse:impulse+24] += z[force:force+24]
            z[force:force+24] = 0.
        return z

    def project(self, z, lower, upper):
        z = strict_clip(z, lower, upper)
        w = z[self.nq:].reshape(-1, 3)
        w[:, :2] = np.clip(w[:, :2], -.7*w[:, 2, None], .7*w[:, 2, None])
        return z

    def evaluate(self, z):
        if self.cache_x is not None and np.array_equal(z, self.cache_x):
            return self.cache_r, self.cache_j
        x, q, forces, impulses = self.unpack(z)
        v, a, chains, vblocks, ablocks = velocity_acceleration(q, self.dt, parameters=x, jacobian=True)
        rk, jk = self.kin.evaluate(z[:self.nq])
        residuals = [rk]
        jacobians = [sparse.hstack([jk, sparse.csr_matrix((len(rk), self.nw))], format='csr')]
        dynamic_rows, torque_rows, physics, torque = [], [], [], []
        contact_rows, contact_values = [], []
        momentum, momentum_jac = {}, {}
        coms, com_jac = {}, {}
        self.frame_data = []
        m, d = self.dyn_model, self.data
        for t in range(1, self.n-1):
            d.qpos[:], d.qvel[:], d.qacc[:] = q[t], v[t], a[t]
            mujoco.mj_inverse(m, d)
            inverse = d.qfrc_inverse.copy()
            angular = d.sensordata.copy()
            coms[t] = d.subtree_com[1].copy()
            jc = np.empty((3, 32))
            mujoco.mj_jacSubtreeCom(m, d, jc, 1)
            com_jac[t] = sparse.csr_matrix((jc@chains[t]).ravel().reshape(3, 32))
            fmat, points = self.force_matrix(d)
            combined = (forces[t]+impulses[t]/self.dt).ravel()
            required = inverse-fmat@combined
            fq, fv, fa = np.empty((32, 32)), np.empty((32, 32)), np.empty((32, 32))
            sq, sv = np.empty((32, 3)), np.empty((32, 3))
            mujoco.mjd_inverseFD(m, d, 1e-6, False, fq, fv, fa, sq, sv, None, None)
            hblocks = {}
            for frame, block in vblocks[t]:
                hblocks[frame] = hblocks.get(frame, 0.)+sv.T@block
            hblocks[t] += sq.T@chains[t]
            hcols = sorted(hblocks)
            hv = np.concatenate([hblocks[i] for i in hcols], axis=1)
            hc = np.concatenate([np.arange(i*32, (i+1)*32) for i in hcols])
            momentum[t] = angular
            momentum_jac[t] = sparse.csr_matrix((hv.ravel(), (np.repeat(np.arange(3), len(hc)), np.tile(hc, 3))), shape=(3, self.size))
            fq, fv, fa = fq.T.copy(), fv.T.copy(), fa.T.copy()
            # Derivative of J(q)^T f. Native inverse-dynamics derivatives do
            # not include our explicit external contact forces.
            if np.linalg.norm(combined) > 1e-10:
                pose = q[t].copy()
                for k in range(3, 32):
                    perturb = np.zeros(32)
                    perturb[k] = 1e-6
                    d.qpos[:] = pose
                    mujoco.mj_integratePos(m, d.qpos, perturb, 1.)
                    mujoco.mj_kinematics(m, d)
                    mujoco.mj_comPos(m, d)
                    shifted, _ = self.force_matrix(d)
                    fq[:, k] -= ((shifted-fmat)@combined)/1e-6
                d.qpos[:] = pose
            blocks = {}
            for frame, block in vblocks[t]:
                blocks[frame] = blocks.get(frame, 0.)+fv@block
            for frame, block in ablocks[t]:
                blocks[frame] = blocks.get(frame, 0.)+fa@block
            blocks[t] += fq@chains[t]
            cols = sorted(blocks)
            compact = np.concatenate([blocks[i] for i in cols], axis=1)
            indexes = np.concatenate([np.arange(i*32, (i+1)*32) for i in cols])
            contact_blocks = []
            if t in self.force_index:
                contact_blocks.append(self.force_index[t])
            if t in self.impulse_index:
                contact_blocks.append(self.impulse_index[t])
            for index in contact_blocks:
                compact = np.c_[compact, -fmat*self.bw]
                indexes = np.r_[indexes, np.arange(self.nq+24*index, self.nq+24*(index+1))]
            row = sparse.csr_matrix((compact.ravel(), (np.repeat(np.arange(32), len(indexes)), np.tile(indexes, 32))), shape=(32, self.size))
            if contact_blocks:
                anchors = np.concatenate([self.points[side]@self.plan['foot_rot'][t, side].T+self.plan['foot_pos'][t, side]
                                          for side in range(2)])
                if self.plan.get('ground_plane_anchors',False):
                    anchors[:,2] = .0007
                normal = (forces[t, :, 2]+impulses[t, :, 2]/self.dt)/self.bw
                error = points-anchors
                comp = normal[:, None]*error
                contact_values.extend(comp.ravel())
                contact_q = np.concatenate([normal[k]*fmat[:, 3*k:3*k+3].T@chains[t] for k in range(8)])
                force_comp = np.zeros((24, 24))
                for k in range(8):
                    force_comp[3*k:3*k+3, 3*k+2] = error[k]
                indexes_c = np.arange(t*32, (t+1)*32)
                values_c = contact_q
                for index in contact_blocks:
                    values_c = np.c_[values_c, force_comp]
                    indexes_c = np.r_[indexes_c, np.arange(self.nq+24*index, self.nq+24*(index+1))]
                contact_rows.append(sparse.csr_matrix((values_c.ravel(),
                    (np.repeat(np.arange(24), len(indexes_c)), np.tile(indexes_c, 24))), shape=(24, self.size)))
            force_scale = np.r_[np.full(3, self.bw), np.full(3, self.bw*.6)]
            root_value = required[:6]/force_scale
            root_jac = sparse.diags(1/force_scale)@row[:6]
            physics.extend(root_value)
            dynamic_rows.append(root_jac)
            # Joint effort: qfrc_inverse minus contact/impulse generalized
            # force is the ONLY available motor torque, with original limits.
            ratio = required[6:]/self.effort
            excess = np.maximum(abs(ratio)-.98, 0.)
            active = excess > 0
            residuals.extend([self.weight*root_value, self.weight*np.sign(ratio)*excess])
            jacobians.extend([self.weight*root_jac, self.weight*sparse.diags(active/self.effort)@row[6:]])
            torque_rows.append(sparse.diags(1/self.effort)@row[6:])
            torque.extend(ratio)
            self.frame_data.append(dict(required=required, force_matrix=fmat, points=points))
        # Continuous inverse dynamics at sparse nodes can fit accelerations
        # that do NOT conserve momentum between samples. Constrain actual
        # finite pose differences too, so that this collocation loophole is
        # visible to the optimizer rather than only the independent validator.
        for t in [0, self.n-1]:
            d.qpos[:] = q[t]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            coms[t] = d.subtree_com[1].copy()
            jc = np.empty((3, 32))
            mujoco.mj_jacSubtreeCom(m, d, jc, 1)
            com_jac[t] = sparse.csr_matrix(jc@chains[t])
        air_rows, air_values = [], []
        air = ~self.plan['contacts'].any(axis=1)
        for t in range(1, self.n-1):
            if not air[t]:
                continue
            error = (coms[t+1]-2*coms[t]+coms[t-1])/self.dt**2-m.opt.gravity
            error /= 9.81
            blocks = [com_jac[t-1]/(self.dt**2*9.81), -2*com_jac[t]/(self.dt**2*9.81), com_jac[t+1]/(self.dt**2*9.81)]
            row = sparse.hstack([sparse.csr_matrix((3, (t-1)*32)), *blocks,
                                  sparse.csr_matrix((3, self.size-(t+2)*32))], format='csr')
            air_values.extend(error)
            air_rows.append(row)
            if t > 1 and air[t-1]:
                scale_h = self.bw*.6*self.dt
                air_values.extend((momentum[t]-momentum[t-1])/scale_h)
                air_rows.append((momentum_jac[t]-momentum_jac[t-1])/scale_h)
        if 'initial_angular_momentum' in self.plan:
            scale_h=self.bw*.6*self.dt
            air_values.extend((momentum[1]-self.plan['initial_angular_momentum'])/scale_h)
            air_rows.append(momentum_jac[1]/scale_h)
        self.discrete_momentum = np.asarray(air_values)
        self.discrete_momentum_jac = sparse.vstack(air_rows, format='csr')
        residuals.append(2*self.weight*self.discrete_momentum)
        jacobians.append(2*self.weight*self.discrete_momentum_jac)
        # Friction pyramids for every sole point, including impulses. There
        # are no force/impulse variables at arbitrary airborne frames.
        w = z[self.nq:].reshape(-1, 3)
        cones = np.c_[w[:, 0]-.7*w[:, 2], -w[:, 0]-.7*w[:, 2], w[:, 1]-.7*w[:, 2], -w[:, 1]-.7*w[:, 2]]
        pattern = sparse.kron(sparse.eye(len(w)), np.array([[1, 0, -.7], [-1, 0, -.7], [0, 1, -.7], [0, -1, -.7]]), format='csr')
        cone_jac = sparse.hstack([sparse.csr_matrix((len(w)*4, self.nq)), pattern], format='csr')
        residuals.extend([self.weight*np.maximum(cones.ravel(), 0.), .03*z[self.nq:]])
        jacobians.extend([self.weight*sparse.diags((cones.ravel()>0).astype(float))@cone_jac,
                          sparse.hstack([sparse.csr_matrix((self.nw, self.nq)), .03*sparse.eye(self.nw)], format='csr')])
        self.physics = np.array(physics)
        self.physics_jac = sparse.vstack(dynamic_rows, format='csr')
        self.physics = np.r_[self.physics, self.discrete_momentum]
        self.physics_jac = sparse.vstack([self.physics_jac, self.discrete_momentum_jac], format='csr')
        self.torque_ratio = np.array(torque)
        self.torque_jac = sparse.vstack(torque_rows, format='csr')
        self.cones, self.cone_jac = cones.ravel(), cone_jac
        self.contact_values = np.array(contact_values)
        self.contact_jac = sparse.vstack(contact_rows, format='csr')
        contact_scale = np.tile([300., 300., 1200.], len(contact_values)//3)
        residuals.append(contact_scale*self.contact_values)
        jacobians.append(sparse.diags(contact_scale)@self.contact_jac)
        residual, jacobian = np.concatenate(residuals), sparse.vstack(jacobians, format='csr')
        self.cache_x, self.cache_r, self.cache_j = z.copy(), residual, jacobian
        self.calls += 1
        if self.calls % 5 == 0:
            print(json.dumps(dict(evaluation=self.calls, cost=float(.5*residual@residual),
                root_residual_max=float(abs(self.physics).max()), torque_ratio_max=float(abs(self.torque_ratio).max()),
                friction_violation=float(np.maximum(self.cones, 0).max()), seconds=time.perf_counter()-self.started)), flush=True)
        return residual, jacobian

    def fun(self, z):
        return self.evaluate(z)[0]

    def jac(self, z):
        return self.evaluate(z)[1]


def jacobian_check(problem, z):
    residual, jacobian = problem.evaluate(z)
    # Hinge penalties (effort, friction and collision activation) have no
    # unique derivative at their kink. Check the smooth physical equations
    # directly instead of misclassifying active-set changes as sensor errors.
    nk = len(problem.kin.cache_r)
    nr = (problem.n-2)*32
    na = len(problem.discrete_momentum)
    cone_size = problem.nw//3*4
    rows = np.r_[np.concatenate([np.arange(nk+i*32, nk+i*32+6) for i in range(problem.n-2)]),
                 np.arange(nk+nr, nk+nr+na),
                 np.arange(nk+nr+na+cone_size, len(residual))]
    rng = np.random.default_rng(81)
    # Airborne, contact, impact and rotation/force columns all get exercised.
    columns = np.r_[np.arange(50*32, 50*32+32), np.arange(67*32, 67*32+32),
                    problem.nq+rng.integers(0, problem.nw, 12)]
    errors = []
    for column in columns:
        trial = z.copy()
        trial[column] += 1e-6
        numeric = ((problem.fun(trial)-residual)/1e-6)[rows]
        exact = jacobian[rows, int(column)].toarray().ravel()
        errors.append(np.linalg.norm(numeric-exact)/max(1., np.linalg.norm(numeric)))
    result = dict(columns=len(columns), smooth_rows=len(rows),
        scope='floating-base dynamics, discrete momentum, force regularization and smooth sole complementarity; excludes active-set hinge penalties',
        relative_error_max=float(max(errors)), relative_error_p95=float(np.quantile(errors, .95)))
    print('JACOBIAN', json.dumps(result), flush=True)
    if result['relative_error_max'] > .005:
        raise AssertionError(result)
    return result


def write_result(problem, z, solver):
    _, q, forces, impulses = problem.unpack(z)
    v, a, *_ = velocity_acceleration(q, .02)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/'88_01_candidate.npz', qpos=q, qvel=v, qacc=a,
        contact_forces_world=forces, contact_impulses_world=impulses,
        frequency=np.asarray(50.), joint_names=np.asarray(['root']+[problem.model.joint(i).name for i in range(1, problem.model.njnt)]),
        split_points=np.array([0, len(q)]), source_contact=problem.plan['contacts'], training_ready=np.asarray(False),
        optimization_variables=z)
    (OUT/'phase_plan.json').write_text(json.dumps(problem.plan['metadata'], indent=2)+'\n')
    (OUT/'solver.json').write_text(json.dumps(solver, indent=2)+'\n')


def polish(problem, z, lower, upper, iterations):
    """Sequential QPs: hard momentum balance, effort, speed and friction.

    Nonlinear geometry is checked after each trial. A failed QP is reported;
    numerical solver success is not a trajectory feasibility certificate.
    """
    sys.path.insert(0, str(kin.OUT/'python_deps'))
    import osqp
    identity = sparse.eye(problem.size, format='csc')
    difference = sparse.hstack([problem.kin.difference, sparse.csr_matrix((problem.kin.difference.shape[0], problem.nw))], format='csc')
    finite = np.flatnonzero(np.isfinite(problem.kin.velocity_bounds))
    velocity = difference[finite]
    speed = problem.kin.velocity_bounds[finite]*.999
    history = []
    def merit(cost):
        violation = (np.abs(problem.physics).sum()
            + np.maximum(abs(problem.torque_ratio)-1., 0.).sum()
            + np.maximum(problem.cones, 0.).sum()
            + 100*np.abs(problem.contact_values).sum()
            + 100*np.maximum(-problem.kin.geometry_clearance, 0.).sum())
        return cost+1e6*violation
    for iteration in range(iterations):
        residual, jac = problem.evaluate(z)
        trust = np.r_[np.tile(np.r_[np.full(3, .06), np.full(3, .8), np.full(26, .3)], problem.n), np.full(problem.nw, 3.)]
        if hasattr(problem, 'trust_limits'):
            trust = np.minimum(trust, problem.trust_limits)
        continuous_count = len(problem.physics)-len(problem.discrete_momentum)
        root_error = abs(problem.physics[:continuous_count]).max()
        root_radius = .0001 if root_error < .05 else .8*root_error
        physics_radius = np.r_[np.full(continuous_count, root_radius), np.full(len(problem.discrete_momentum), .001)]
        effort_radius = max(.999, .8*abs(problem.torque_ratio).max())
        matrices = [identity, velocity, problem.physics_jac, problem.torque_jac, problem.cone_jac]
        lows = [np.maximum(lower-z, -trust), -speed-velocity@z, -physics_radius-problem.physics,
                -effort_radius-problem.torque_ratio, np.full(len(problem.cones), -np.inf)]
        highs = [np.minimum(upper-z, trust), speed-velocity@z, physics_radius-problem.physics,
                 effort_radius-problem.torque_ratio, -problem.cones]
        geometry = sparse.hstack([problem.kin.geometry_jacobian,
                                 sparse.csr_matrix((problem.kin.geometry_jacobian.shape[0], problem.nw))], format='csc')
        matrices.append(geometry)
        lows.append(-problem.kin.geometry_clearance+1e-7)
        highs.append(np.full(geometry.shape[0], np.inf))
        # Complementarity permits a heel to lift when its normal load falls
        # to zero, while a loaded toe remains attached to its contact anchor.
        contact_radius = max(.00005, .8*abs(problem.contact_values).max())
        matrices.append(problem.contact_jac)
        lows.append(-contact_radius-problem.contact_values)
        highs.append(contact_radius-problem.contact_values)
        qp = osqp.OSQP()
        qp.setup(P=sparse.triu(jac.T@jac+.5*identity, format='csc'), q=np.asarray(jac.T@residual).ravel(),
            A=sparse.vstack(matrices, format='csc'), l=np.concatenate(lows), u=np.concatenate(highs),
            eps_abs=1e-5, eps_rel=1e-5, max_iter=25000, time_limit=45., polishing=True, verbose=False)
        result = qp.solve(raise_error=False)
        # Inexact/time-limited QP iterates can help the SEARCH; they are not
        # feasibility certificates. True geometry, effort and momentum are
        # re-evaluated before accepting a merit-decreasing step.
        if result.info.status_val not in (1, 2, 7, 8) or result.x is None or not np.isfinite(result.x).all():
            history.append(dict(iteration=iteration, status=result.info.status))
            print('SQP', json.dumps(history[-1]), flush=True)
            break
        current_cost = float(.5*residual@residual)
        current_merit = merit(current_cost)
        accepted = None
        for fraction in [1., .5, .25, .125]:
            trial = problem.project(z+fraction*result.x, lower, upper)
            r = problem.fun(trial)
            cost = float(.5*r@r)
            if merit(cost) < current_merit:
                accepted = (trial, cost, fraction)
                break
        if accepted is None:
            problem.evaluate(z)
            history.append(dict(iteration=iteration, status='no merit-decreasing step'))
            break
        z, cost, fraction = accepted
        history.append(dict(iteration=iteration, cost=cost, step_fraction=fraction,
            root_residual_max=float(abs(problem.physics).max()),
            effort_ratio_max=float(abs(problem.torque_ratio).max()), qp_status=result.info.status,
            merit=merit(cost), qp_iterations=result.info.iter))
        print('SQP', json.dumps(history[-1]), flush=True)
        checkpoint_dir = getattr(problem,'output_directory',OUT)
        np.savez_compressed(checkpoint_dir/'optimization_checkpoint.npz', optimization_variables=z)
        (checkpoint_dir/'sqp_checkpoint.json').write_text(json.dumps(history, indent=2)+'\n')
        if abs(problem.physics).max()<1e-4 and abs(problem.torque_ratio).max()<1.0001:
            break
    return z, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-nfev', type=int, default=100)
    parser.add_argument('--weight', type=float, default=100.)
    parser.add_argument('--check-jacobian', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--sqp-iterations', type=int, default=0)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(ROOT/'outputs/humanoid_light_conversion/scene.xml'))
    motion = load_asfamc(cmu.DATA/'raw/88.asf', cmu.DATA/'raw/88_01.amc', cmu.DATA/'raw/88_01.c3d.header')
    plan, seed, human, names, scale = make_plan(model, motion)
    print('PHASES', json.dumps(plan['metadata']), flush=True)
    problem = DynamicsProblem(model, plan, args.weight)
    z, lower, upper = problem.initial(seed)
    if args.resume:
        z = problem.project(problem.canonical_impulses(np.load(args.resume, allow_pickle=False)['optimization_variables']), lower, upper)
    check = jacobian_check(problem, z) if args.check_jacobian else None
    started = time.perf_counter()
    if args.max_nfev > 0:
        result = least_squares(problem.fun, z, jac=problem.jac, bounds=(lower, upper),
            max_nfev=args.max_nfev, ftol=2e-6, xtol=1e-7, gtol=1e-5, tr_solver='lsmr', x_scale='jac',
            tr_options={'atol':1e-6, 'btol':1e-6, 'maxiter':200})
        z = result.x
        ls = dict(success=bool(result.success), message=result.message, nfev=result.nfev,
                  cost=float(result.cost), optimality=float(result.optimality))
    else:
        ls = dict(resume_only=True)
    z, sqp = polish(problem, z, lower, upper, args.sqp_iterations) if args.sqp_iterations else (z, [])
    residual, _ = problem.evaluate(z)
    solver = dict(least_squares=ls, sqp=sqp, cost=float(.5*residual@residual), wall_seconds=time.perf_counter()-started,
        jacobian_check=check, root_residual_max=float(abs(problem.physics).max()),
        torque_ratio_max=float(abs(problem.torque_ratio).max()), friction_violation=float(np.maximum(problem.cones, 0).max()))
    write_result(problem, z, solver)
    print('RESULT', json.dumps(solver), flush=True)


if __name__ == '__main__':
    main()
