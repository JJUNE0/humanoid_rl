"""COM/body-first CMU retargeting with free landing footprints and phase search.

Reference limbs are soft preferences. Airborne COM follows gravity, while
contact locations/headings are decisions and touchdown is searched on the
50 Hz time grid. Motor limits and the independent physics screen still apply.
The default experiment starts from a prescribed aerial state at 1.06 seconds;
takeoff and the earlier part of the backflip are not validated.
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
from scipy.sparse.linalg import spsolve
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R, Slerp

import optimize_humanoid_light_cmu_dynamics as dyn
import optimize_humanoid_light_cmu_landing as kin
import retarget_humanoid_light_lafan as ik

ROOT = dyn.ROOT
OUT = ROOT/'outputs/humanoid_light_cmu_com_first'


def free_ground_pose(model, center, com_height, pitch_degrees, knee_hint=1., initial=None,
                     fixed_feet=None, forward_com=.10):
    """A reachable low pose, with free foot XY/yaw and original joint bounds."""
    data = mujoco.MjData(model)
    _, feet, ids, standing, p0, r0 = kin.foot_geometry(model)
    root_rotation = R.from_euler('y', pitch_degrees, degrees=True).as_quat(scalar_first=True)
    q0 = standing.copy() if initial is None else initial.copy()
    yaw0 = np.zeros(2)
    if initial is None:
        q0[:2] += np.asarray(center)-p0.mean(axis=0)[:2]
    else:
        data.qpos[:] = initial
        mujoco.mj_kinematics(model, data)
        rel = data.xmat[ids].reshape(2, 3, 3)@np.swapaxes(r0, -1, -2)
        yaw0 = np.arctan2(rel[:, 1, 0], rel[:, 0, 0])
    hint = q0[7:].copy()
    for side in ['left', 'right']:
        hint[model.joint(side+'_knee_joint').qposadr[0]-7] = knee_hint
    lower = np.r_[np.asarray(center)-.55, .25, model.jnt_range[1:, 0]+1e-5, [-1.3, -1.3]]
    upper = np.r_[np.asarray(center)+.55, .95, model.jnt_range[1:, 1]-1e-5, [1.3, 1.3]]

    def evaluate(x):
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = x[:3], root_rotation, x[3:29]
        mujoco.mj_fwdPosition(model, data)
        position = data.xpos[ids].copy()
        rotation = data.xmat[ids].reshape(2, 3, 3)
        desired = R.from_rotvec(np.c_[np.zeros((2, 2)), x[-2:]]).as_matrix()@r0
        values = [500*(data.subtree_com[1, 2]-com_height),
                  *(40*(data.subtree_com[1, :2]-position.mean(axis=0)[:2]-[forward_com, 0.])),
                  *(5*(position.mean(axis=0)[:2]-center)),
                  *(150*R.from_matrix(np.swapaxes(desired, -1, -2)@rotation).as_rotvec().ravel()),
                  *(.3*(x[3:29]-hint)), *(.1*x[-2:])]
        for side, body, rot, pos in zip(['left', 'right'], ids, rotation, position):
            bottom = (feet[side][1]@rot.T+pos)[:, 2].min()
            values.append(1200*(bottom-.0007))
        width = position[0, 1]-position[1, 1]
        values.extend([800*min(width-.13, 0.), 800*max(width-.42, 0.),
                       400*max(abs(position[0, 0]-position[1, 0])-.20, 0.)])
        if fixed_feet is not None:
            fp, fr = fixed_feet
            values.extend((700*(position-fp)).ravel())
            values.extend((200*R.from_matrix(np.swapaxes(fr, -1, -2)@rotation).as_rotvec()).ravel())
        return np.array(values)

    x0 = np.r_[q0[:3], q0[7:], yaw0]
    fit = least_squares(evaluate, np.clip(x0, lower+1e-7, upper-1e-7), bounds=(lower, upper),
                        max_nfev=180, ftol=1e-8, xtol=1e-8, gtol=1e-7)
    evaluate(fit.x)
    q = data.qpos.copy()
    meshes, _ = ik.geometry(model)
    vertices = dict(meshes)
    floor_min = min(float((v@data.geom_xmat[g].reshape(3,3)[2]+data.geom_xpos[g,2]).min()) for g,v in meshes)
    self_min = min(kin.checked_mesh_distance(model,data,a,b,vertices)[0] for a,b in ik.self_collision_pairs(model))
    return q, dict(requested_com_height_m=com_height, achieved_com_height_m=float(data.subtree_com[1, 2]),
        com_position_m=data.subtree_com[1].tolist(), minimum_floor_height_m=floor_min, minimum_selected_self_gap_m=float(self_min),
        root_pitch_degrees=pitch_degrees, root_height_m=float(q[2]),
        foot_positions=data.xpos[ids].tolist(), foot_rotations=data.xmat[ids].reshape(2, 3, 3).tolist(),
        foot_yaw_degrees=np.rad2deg(fit.x[-2:]).tolist(), cost=float(fit.cost), solver_success=bool(fit.success))


def ground_seeds(model):
    raw = kin.json_load(kin.OUT/'phase_plan.json')
    center = np.array(raw['landing_foot_positions']).mean(axis=0)[:2]
    choices = []
    for pitch in [25., 40., 55.]:
        q, report = free_ground_pose(model, center, .65, pitch)
        choices.append((q, report))
    pairs = []
    for pitch in [0., 15., 30.]:
        qa, absorb = free_ground_pose(model, center, .59, pitch, 1.3, forward_com=.03)
        fixed = (np.asarray(absorb['foot_positions']),np.asarray(absorb['foot_rotations']))
        ql, landing = free_ground_pose(model, center, .65, 40., .9, initial=qa,
                                        fixed_feet=fixed, forward_com=.12)
        score = landing['cost']+absorb['cost']+1e5*max(-landing['minimum_selected_self_gap_m'],0.)
        pairs.append((score, ql, qa, landing, absorb))
    _, qland, qabs, landing, absorb = min(pairs,key=lambda item:item[0])
    info = dict(landing_candidates=[r for _, r in choices], selected_landing=landing,
                compatible_pairs=[dict(score=s,landing=l,absorption=a) for s,_,_,l,a in pairs], selected_absorption=absorb)
    return qland, qabs, info


def hermite(a, b, va, vb, seconds, duration):
    u = np.clip(seconds/duration, 0., 1.)
    return (2*u**3-3*u**2+1)*a+(u**3-2*u**2+u)*duration*va+(-2*u**3+3*u**2)*b+(u**3-u**2)*duration*vb


def make_com_plan(model, motion, qland, qabs, ground, phases):
    plan, _, human, names, scale = dyn.make_plan(model, motion)
    times = plan['times']
    takeoff = plan['metadata']['takeoff_seconds']
    touchdown, rebound_off, rebound_land = phases
    old = np.load(kin.OUT/'88_01_optimized.npz', allow_pickle=False)['qpos'].astype(float)
    source = np.load(dyn.OUT/'88_01_candidate.npz', allow_pickle=False)['qpos'].astype(float)
    knots_old = np.array([0., takeoff, 1.34, 1.58, 1.84, times[-1]])
    knots_new = np.array([0., takeoff, touchdown, rebound_off, rebound_land, times[-1]])
    source_t = np.interp(times, knots_new, knots_old)
    seed = np.empty_like(source)
    seed[:, :3] = np.stack([np.interp(source_t, times, source[:, k]) for k in range(3)], axis=1)
    seed[:, 7:] = np.stack([np.interp(source_t, times, source[:, k]) for k in range(7, 33)], axis=1)
    seed[:, 3:7] = Slerp(times, R.from_quat(source[:, 3:7], scalar_first=True))(source_t).as_quat(scalar_first=True)
    warm_start = seed.copy()
    data = mujoco.MjData(model)
    def com(q):
        data.qpos[:] = q
        mujoco.mj_fwdPosition(model, data)
        return data.subtree_com[1].copy()
    ids = [model.body(side+'_ankle_roll_link').id for side in ['left','right']]
    launch_q = old[round(takeoff*50)].copy()
    c0, cl, ca = com(launch_q), com(qland), com(qabs)
    fp1 = np.asarray(ground['selected_absorption']['foot_positions'])
    fr1 = np.asarray(ground['selected_absorption']['foot_rotations'])
    center1 = fp1.mean(axis=0)[:2]
    qlaunch2, launch2_info = free_ground_pose(model, center1, .72, 0., .25,
        initial=qabs, fixed_feet=(fp1, fr1), forward_com=.03)
    c_launch2 = com(qlaunch2)
    shift = np.asarray(plan['metadata']['rebound_landing_foot_positions']).mean(axis=0)[:2]-np.asarray(plan['metadata']['landing_foot_positions']).mean(axis=0)[:2]
    qland2, land2 = free_ground_pose(model, center1+shift, .68, 0., .7, forward_com=.03)
    fp2, fr2 = np.asarray(land2['foot_positions']), np.asarray(land2['foot_rotations'])
    qstand, _ = free_ground_pose(model, center1+shift, .74, 0., .1, initial=qland2,
                                 fixed_feet=(fp2, fr2), forward_com=0.)
    c2, cstand = com(qland2), com(qstand)
    gravity = model.opt.gravity
    flight_time = touchdown-takeoff
    v0 = (cl-c0)/flight_time-.5*gravity*flight_time
    rebound_time = rebound_land-rebound_off
    v2 = (c2-c_launch2)/rebound_time-.5*gravity*rebound_time
    initial_pitch = R.from_quat(launch_q[3:7], scalar_first=True).as_euler('xyz')[1]
    landing_pitch = np.deg2rad(ground['selected_landing']['root_pitch_degrees'])
    theta_land = -2*np.pi+landing_pitch
    angular_rate = (theta_land-initial_pitch)/flight_time
    absorption_end = touchdown+.12
    rotation = R.from_quat(old[:,3:7],scalar_first=True).as_matrix()
    target_com = np.stack([com(q) for q in old])
    for t, seconds in enumerate(times):
        if seconds < takeoff:
            continue
        if seconds < touchdown:
            tau = seconds-takeoff
            theta = initial_pitch+angular_rate*tau
            target_com[t] = c0+v0*tau+.5*gravity*tau*tau
            blend = kin.smoothstep((seconds-(touchdown-.18))/.18)
            seed[t,7:] = (1-blend)*seed[t,7:]+blend*qland[7:]
        elif seconds < absorption_end:
            tau = seconds-touchdown
            theta = float(hermite(theta_land,-2*np.pi,angular_rate,0.,tau,.22))
            target_com[t] = hermite(cl,ca,np.array([0.,0.,-.65]),np.zeros(3),tau,.12)
            u = kin.smoothstep(tau/.12)
            seed[t,7:] = (1-u)*qland[7:]+u*qabs[7:]
        elif seconds < rebound_off:
            tau = seconds-absorption_end
            theta = float(hermite(theta_land,-2*np.pi,angular_rate,0.,seconds-touchdown,.22))
            target_com[t] = hermite(ca,c_launch2,np.zeros(3),v2,tau,rebound_off-absorption_end)
            u = kin.smoothstep(tau/(rebound_off-absorption_end))
            seed[t,7:] = (1-u)*qabs[7:]+u*qlaunch2[7:]
        elif seconds < rebound_land:
            tau = seconds-rebound_off
            theta = -2*np.pi
            target_com[t] = c_launch2+v2*tau+.5*gravity*tau*tau
            u = kin.smoothstep(tau/rebound_time)
            seed[t,7:] = (1-u)*qlaunch2[7:]+u*qland2[7:]
        else:
            theta = -2*np.pi
            tau = seconds-rebound_land
            target_com[t] = hermite(c2,cstand,np.zeros(3),np.zeros(3),tau,.65)
            u = kin.smoothstep(tau/.65)
            seed[t,7:] = (1-u)*qland2[7:]+u*qstand[7:]
        rotation[t] = R.from_rotvec([0.,theta,0.]).as_matrix()
        seed[t,3:7] = R.from_matrix(rotation[t]).as_quat(scalar_first=True)
        seed[t,:3] += target_com[t]-com(seed[t])
    contacts = np.ones((len(times),2),dtype=bool)
    contacts[((times>takeoff)&(times<touchdown))|((times>rebound_off)&(times<rebound_land))] = False
    foot_pos, foot_rot = plan['foot_pos'].copy(),plan['foot_rot'].copy()
    foot_pos[times>=touchdown],foot_rot[times>=touchdown] = fp1,fr1
    foot_pos[times>=rebound_land],foot_rot[times>=rebound_land] = fp2,fr2
    remap = rotation@np.swapaxes(plan['rotation'],-1,-2)
    plan['bone_target'] = np.einsum('tij,tbj->tbi',remap,plan['bone_target'])
    plan.update(rotation=rotation,rotation_target=rotation.copy(),com=target_com,
        root=seed[:,:3].copy(),joint_reference=seed[:,7:].copy(),foot_pos=foot_pos,foot_rot=foot_rot,
        contacts=contacts,q_reference=seed.copy(),ground_plane_anchors=True,
        transition_intervals=[(takeoff-.14,takeoff),(touchdown,rebound_off),(rebound_land,rebound_land+.2)],
        weights=dict(root_position=0.,root_rotation=80.,joint_air=.03,joint_ground=.08,
                     bone_direction=.2,contact_position=0.,contact_orientation=0.,
                     com_air=[50.,50.,220.],com_ground=[5.,5.,220.]))
    plan['metadata'].update(touchdown_seconds=touchdown,rebound_takeoff_seconds=rebound_off,
        rebound_touchdown_seconds=rebound_land,absorption_end_seconds=absorption_end,
        recovery_end_seconds=rebound_land+.65,landing_foot_positions=fp1.tolist(),
        rebound_landing_foot_positions=fp2.tolist(),
        landing_com_target_m=cl.tolist(),absorption_com_target_m=ca.tolist(),
        root_rotation_priority='strong continuous turn, forward-leaning touchdown, rotation completed after contact',
        limb_priority='weak source bone directions and joint posture',
        weights=plan['weights'],contact_time_method='outer search on the 20 ms grid',
        foot_location_method='continuous free XY and yaw per foot, independently for both landings',
        physical_constraints_retained=True,ground_seed=ground,
        rebound_launch_com_target_m=c_launch2.tolist())
    plan['warm_source_times'] = source_t
    plan['warm_qpos'] = warm_start
    return plan, seed, human, names, scale


def landing_window(model,plan,seed,qland,start=1.06):
    """Same physical aerial initial state for every touchdown-time candidate.

    Launch is deliberately outside this study. Two initial poses come from
    real free flight under bounded holding torques, not arbitrary pose edits.
    """
    old=np.load(kin.OUT/'88_01_optimized.npz',allow_pickle=False)['qpos'].astype(float)
    data=mujoco.MjData(model);data.qpos[:]=old[37];mujoco.mj_forward(model,data)
    launch_com=data.subtree_com[1].copy()
    landing_com=np.array(plan['metadata']['landing_com_target_m'])
    gravity=model.opt.gravity
    launch_v=(landing_com-launch_com)/.60-.5*gravity*.60
    tau=start-.74
    initial_com=launch_com+launch_v*tau+.5*gravity*tau*tau
    initial_com_v=launch_v+gravity*tau
    theta0=R.from_quat(old[37,3:7],scalar_first=True).as_euler('xyz')[1]
    theta_land=-2*np.pi+np.deg2rad(plan['metadata']['ground_seed']['selected_landing']['root_pitch_degrees'])
    initial_theta=theta0+(theta_land-theta0)*(start-.74)/.60
    q0=model.qpos0.copy();q0[model.joint('torso_pitch_joint').qposadr[0]]=.02
    q0[3:7]=R.from_rotvec([0.,initial_theta,0.]).as_quat(scalar_first=True)
    data.qpos[:]=q0;mujoco.mj_forward(model,data);q0[:3]+=initial_com-data.subtree_com[1]
    data.qpos[:]=q0;mujoco.mj_forward(model,data)
    angular=np.empty((3,32));mujoco.mj_angmomMat(model,data,angular,1)
    v0=np.zeros(32);initial_h=np.array([0.,-35.6,0.])
    v0[3:6]=np.linalg.solve(angular[:,3:6],initial_h)
    jc=np.empty((3,32));mujoco.mj_jacSubtreeCom(model,data,jc,1)
    v0[:3]=initial_com_v-jc[:,3:]@v0[3:]
    sim=copy.deepcopy(model);sim.opt.timestep=.001;sim.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    state=mujoco.MjData(sim);state.qpos[:]=q0;state.qvel[:]=v0
    motor_j=np.array([sim.jnt_dofadr[j]-6 for j in sim.actuator_trnid[:,0]])
    for _ in range(20):
        mujoco.mj_forward(sim,state)
        mass=np.empty((32,32));bias=np.empty(32);mujoco.mj_fullM(sim,state,mass);mujoco.mj_rne(sim,state,0,bias)
        aj=-100*(state.qpos[7:]-q0[7:])-20*state.qvel[6:]
        ar=np.linalg.solve(mass[:6,:6],-bias[:6]-mass[:6,6:]@aj)
        torque=mass[6:,:6]@ar+mass[6:,6:]@aj+bias[6:]
        state.ctrl[:]=np.clip(torque[motor_j],sim.actuator_ctrlrange[:,0],sim.actuator_ctrlrange[:,1])
        mujoco.mj_step(sim,state)
    q1=state.qpos.copy();data.qpos[:]=q1;mujoco.mj_forward(model,data)
    q1[:3]+=initial_com+initial_com_v*.02+.5*gravity*.02**2-data.subtree_com[1]
    begin=round(start*50);count=len(seed)
    for key,value in list(plan.items()):
        if isinstance(value,np.ndarray) and len(value)==count:plan[key]=value[begin:].copy()
    seed=seed[begin:].copy();times=plan['times']
    td=plan['metadata']['touchdown_seconds'];abs_end=plan['metadata']['absorption_end_seconds']
    off=plan['metadata']['rebound_takeoff_seconds'];td2=plan['metadata']['rebound_touchdown_seconds']
    com_target=plan['com'].copy();rotation=plan['rotation'].copy()
    touch_com=initial_com+initial_com_v*(td-start)+.5*gravity*(td-start)**2
    absorb_com=np.array(plan['metadata']['absorption_com_target_m'])
    absorb_com[:2]=touch_com[:2]+(initial_com_v+gravity*(td-start))[:2]*.06
    absorb_com[2]=min(absorb_com[2],touch_com[2]-.03)
    post_v=initial_com_v+gravity*(td-start);post_v[2]=-2*(touch_com[2]-absorb_com[2])/.12
    for t,seconds in enumerate(times):
        if seconds<td:
            dt=seconds-start
            com_target[t]=initial_com+initial_com_v*dt+.5*gravity*dt*dt
            angle=initial_theta+(theta_land-initial_theta)*dt/(td-start)
            u=kin.smoothstep(dt/(td-start))
            seed[t,7:]=(1-u)*q0[7:]+u*qland[7:]
            rotation[t]=R.from_rotvec([0.,angle,0.]).as_matrix()
        elif seconds<abs_end:
            com_target[t]=hermite(touch_com,absorb_com,post_v,np.zeros(3),seconds-td,.12)
        elif seconds<off:
            com_target[t]=hermite(absorb_com,plan['com'][int(np.argmin(abs(times-off)))],
                np.zeros(3),np.gradient(plan['com'],.02,axis=0)[int(np.argmin(abs(times-off)))],seconds-abs_end,off-abs_end)
        data.qpos[:]=seed[t];data.qpos[3:7]=R.from_matrix(rotation[t]).as_quat(scalar_first=True)
        mujoco.mj_fwdPosition(model,data)
        seed[t]=data.qpos;seed[t,:3]+=com_target[t]-data.subtree_com[1]
    seed[0],seed[1]=q0,q1
    remap=rotation@np.swapaxes(plan['rotation'],-1,-2)
    plan['bone_target']=np.einsum('tij,tbj->tbi',remap,plan['bone_target'])
    plan.update(rotation=rotation,rotation_target=rotation.copy(),com=com_target,
                joint_reference=seed[:,7:].copy(),root=seed[:,:3].copy(),warm_qpos=seed.copy(),
                initial_angular_momentum=initial_h,initial_velocity=v0,fixed_initial_poses=2)
    plan['metadata'].update(partial_motion=True,time_origin_seconds=start,
        initial_condition='synthetic valid aerial state; launch not validated',
        initial_angular_momentum=initial_h.tolist(),initial_velocity=v0.tolist(),
        initial_com_position=initial_com.tolist(),initial_com_velocity=initial_com_v.tolist(),
        landing_com_target_m=touch_com.tolist(),absorption_com_target_m=absorb_com.tolist(),
        expected_rotation_in_window_degrees=float(np.rad2deg(-2*np.pi-initial_theta)))
    return plan,seed


class FreeFootProblem:
    """Wrap full-body dynamics with 12 free footprint XY/yaw variables."""
    def __init__(self, model, plan):
        self.base = dyn.DynamicsProblem(model,plan)
        self.model,self.plan,self.kin = model,plan,self.base.kin
        self.output_directory = OUT
        self.n,self.dt,self.nq = self.base.n,self.base.dt,self.base.nq
        self.anchor_count = 12
        self.nw,self.size = self.base.nw+12,self.base.size+12
        self.anchor_position = plan['foot_pos'].copy()
        self.anchor_rotation = plan['foot_rot'].copy()
        self.td = int(np.argmin(abs(plan['times']-plan['metadata']['touchdown_seconds'])))
        self.td2 = int(np.argmin(abs(plan['times']-plan['metadata']['rebound_touchdown_seconds'])))
        self.cache_x = None
        self.last_anchors = None
        self.trust_limits = np.r_[np.full(self.base.size,np.inf),np.tile([.04,.04,.15],4)]
        d = sparse.lil_matrix((self.n,self.n))
        d[0,0],d[0,1] = -1/self.dt,1/self.dt
        d[-1,-2],d[-1,-1] = -1/self.dt,1/self.dt
        for t in range(1,self.n-1):d[t,t-1],d[t,t+1] = -.5/self.dt,.5/self.dt
        self.com_derivative = sparse.kron(d.tocsr(),sparse.eye(3),format='csr')

    def phase(self,t):
        return None if t<self.td else 0 if t<self.td2 else 1

    def update_anchors(self,a):
        for t in range(self.td,self.n):
            phase=self.phase(t)
            for side in range(2):
                x=a[phase,side]
                self.plan['foot_pos'][t,side]=self.anchor_position[t,side]+[x[0],x[1],0.]
                self.plan['foot_rot'][t,side]=R.from_rotvec([0.,0.,x[2]]).as_matrix()@self.anchor_rotation[t,side]

    def initial(self,q):
        z,lo,hi=self.base.initial(q)
        if self.plan.get('fixed_initial_poses',0):
            stop=self.plan['fixed_initial_poses']*32
            lo[:stop],hi[:stop]=z[:stop]-1e-10,z[:stop]+1e-10
        return np.r_[z,np.zeros(12)],np.r_[lo,np.tile([-.45,-.35,-1.2],4)],np.r_[hi,np.tile([.45,.35,1.2],4)]

    def project(self,z,lo,hi):
        return np.r_[self.base.project(z[:self.base.size],lo[:self.base.size],hi[:self.base.size]),
                     dyn.strict_clip(z[self.base.size:],lo[self.base.size:],hi[self.base.size:])]

    def unpack(self,z):return self.base.unpack(z[:self.base.size])

    def evaluate(self,z):
        if self.cache_x is not None and np.array_equal(z,self.cache_x):return self.cache_r,self.cache_j
        a=z[self.base.size:].reshape(2,2,3)
        if self.last_anchors is None or not np.array_equal(a,self.last_anchors):
            self.update_anchors(a)
            self.base.cache_x=None
            self.last_anchors=a.copy()
        r,j=self.base.evaluate(z[:self.base.size])
        _,q,forces,impulses=self.base.unpack(z[:self.base.size])
        contact_extra=np.zeros((len(self.base.contact_values),12))
        row=0
        for t in self.base.active:
            phase=self.phase(t)
            if phase is not None:
                normal=(forces[t,:,2]+impulses[t,:,2]/self.dt)/self.base.bw
                for side in range(2):
                    rot=self.plan['foot_rot'][t,side]
                    for k,point in enumerate(self.base.points[side]):
                        slot=side*4+k
                        start=row+3*slot
                        col=(phase*2+side)*3
                        contact_extra[start:start+3,col:col+2]=-normal[slot]*np.eye(3)[:,:2]
                        contact_extra[start:start+3,col+2]=-normal[slot]*np.cross([0.,0.,1.],rot@point)
            row+=24
        extra=sparse.vstack([sparse.csr_matrix((len(r)-len(self.base.contact_values),12)),
                             sparse.diags(np.tile([300.,300.,1200.],len(self.base.contact_values)//3))@sparse.csr_matrix(contact_extra)],format='csr')
        j=sparse.hstack([j,extra],format='csr')
        # Weak footprint prior; separation constraints prevent crossed feet.
        prior=np.tile([.5,.5,.15],4)
        constraints=[];constraint_j=[]
        for phase,t in enumerate([self.td,self.td2]):
            pos=self.plan['foot_pos'][t]
            width=pos[0,1]-pos[1,1]
            for value,sign in [(width-.12,-1.),(width-.45,1.)]:
                active=sign*value>0
                constraints.append(300*max(sign*value,0.))
                row_j=np.zeros(self.size)
                if active:
                    row_j[self.base.size+phase*6+1]=300*sign
                    row_j[self.base.size+phase*6+4]=-300*sign
                constraint_j.append(row_j)
        # COM velocity matters at the transition as well as COM position.
        coms=[];crows=[];data=mujoco.MjData(self.model)
        x=z[:self.nq].reshape(self.n,32)
        for t in range(self.n):
            data.qpos[:]=q[t];mujoco.mj_kinematics(self.model,data);mujoco.mj_comPos(self.model,data)
            coms.append(data.subtree_com[1].copy())
            jc=np.empty((3,32));mujoco.mj_jacSubtreeCom(self.model,data,jc,1)
            chain=np.eye(32);chain[3:6,3:6]=kin.right_jacobian(x[t,3:6])
            crows.append(sparse.csr_matrix(jc@chain))
        cjac=sparse.hstack([sparse.block_diag(crows,format='csr'),sparse.csr_matrix((3*self.n,self.nw))],format='csr')
        velocity=self.com_derivative@np.asarray(coms).ravel()
        target_v=self.com_derivative@self.plan['com'].ravel()
        velocity_weight=np.tile([1.,1.,10.],self.n)
        r=np.r_[r,prior*a.ravel(),constraints,velocity_weight*(velocity-target_v)]
        j=sparse.vstack([j,sparse.hstack([sparse.csr_matrix((12,self.base.size)),sparse.diags(prior)]),
                         sparse.csr_matrix(np.array(constraint_j)),sparse.diags(velocity_weight)@self.com_derivative@cjac],format='csr')
        for name in ['physics','torque_ratio','cones','contact_values','discrete_momentum']:
            setattr(self,name,getattr(self.base,name))
        for name in ['physics_jac','torque_jac','cone_jac','discrete_momentum_jac']:
            setattr(self,name,sparse.hstack([getattr(self.base,name),sparse.csr_matrix((getattr(self.base,name).shape[0],12))],format='csr'))
        self.contact_jac=sparse.hstack([self.base.contact_jac,sparse.csr_matrix(contact_extra)],format='csr')
        self.cache_x,self.cache_r,self.cache_j=z.copy(),r,j
        return r,j

    def fun(self,z):return self.evaluate(z)[0]
    def jac(self,z):return self.evaluate(z)[1]


def optimize_case(model,motion,qland,qabs,ground,phases,nfev,window_start=0.):
    plan,seed,human,names,scale=make_com_plan(model,motion,qland,qabs,ground,phases)
    if window_start:plan,seed=landing_window(model,plan,seed,qland,window_start)
    problem=FreeFootProblem(model,plan)
    z,lo,hi=problem.initial(plan['warm_qpos'])
    previous=np.load(dyn.OUT/'88_01_candidate.npz',allow_pickle=False)
    source_times=plan['warm_source_times']
    for t,index in problem.base.force_index.items():
        f=np.stack([np.interp(source_times[t],np.arange(len(previous['qpos']))*.02,previous['contact_forces_world'][:,k,j])
                    for k in range(8) for j in range(3)]).reshape(8,3)
        z[problem.nq+index*24:problem.nq+(index+1)*24]=f.ravel()/problem.base.bw
    for t,index in problem.base.impulse_index.items():
        old_t=67 if t==problem.td else 92
        impulse=previous['contact_impulses_world'][old_t]
        z[problem.nq+index*24:problem.nq+(index+1)*24]=impulse.ravel()/(problem.base.bw*problem.dt)
    # Align the initial anchor decisions with the warm-start feet; the low
    # landing footprints remain weak preferences, not imposed initial poses.
    data=mujoco.MjData(model)
    anchors=z[problem.base.size:].reshape(2,2,3)
    for phase,t in enumerate([problem.td,problem.td2]):
        data.qpos[:]=plan['warm_qpos'][t];mujoco.mj_kinematics(model,data)
        for side,bid in enumerate(problem.base.feet):
            anchors[phase,side,:2]=data.xpos[bid,:2]-problem.anchor_position[t,side,:2]
            rel=data.xmat[bid].reshape(3,3)@problem.anchor_rotation[t,side].T
            anchors[phase,side,2]=np.arctan2(rel[1,0],rel[0,0])
    z=problem.project(z,lo,hi)
    fit,z=fit_free(problem,z,lo,hi,nfev)
    problem.evaluate(z)
    # Rank physical violations first, then source/body fit.
    score=float(np.mean(problem.physics**2)+np.mean(np.maximum(abs(problem.torque_ratio)-1,0)**2)
                +100*np.mean(problem.contact_values**2)+1e-7*fit.cost)
    return problem,z,lo,hi,dict(phases=phases,nfev=fit.nfev,cost=float(fit.cost),physical_score=score),seed


def fit_free(problem,z,lo,hi,nfev):
    """Eliminate the deliberately fixed touchdown-force columns from TRF."""
    free=np.flatnonzero(hi-lo>1e-8)
    fixed=z.copy()
    def unpack(y):
        value=fixed.copy();value[free]=y;return value
    result=least_squares(lambda y:problem.fun(unpack(y)),z[free],
        jac=lambda y:problem.jac(unpack(y))[:,free],bounds=(lo[free],hi[free]),max_nfev=nfev,
        tr_solver='lsmr',x_scale='jac',tr_options={'maxiter':2000,'atol':1e-7,'btol':1e-7},
        ftol=1e-6,xtol=1e-7,gtol=1e-5)
    return result,unpack(result.x)


def projected_refinement(problem,z,lo,hi,iterations=12):
    """Sparse damped Gauss--Newton steps for the new body/anchor decisions.

    Direct factorization propagates body changes across the entire trajectory;
    a short LSMR inner solve can otherwise make almost no progress along these
    smooth, globally coupled directions. Physical screening remains mandatory.
    """
    free=np.flatnonzero(hi-lo>1e-8)
    trust=np.r_[np.tile(np.r_[np.full(3,.08),np.full(3,.3),np.full(26,.5)],problem.n),
                 np.full(problem.base.nw,5.),np.tile([.15,.15,.5],4)]
    history=[];damping=.3
    for iteration in range(iterations):
        residual,jac=problem.evaluate(z);cost=float(.5*residual@residual)
        j=jac[:,free].tocsc()
        delta=np.zeros(problem.size)
        delta[free]=spsolve((j.T@j+damping*sparse.eye(len(free))).tocsc(),-np.asarray(j.T@residual).ravel())
        if not np.isfinite(delta).all():raise ArithmeticError('Nonfinite Gauss-Newton direction')
        delta=np.clip(delta,-trust,trust)
        best=None
        for fraction in [1.,.5,.25,.125,.0625,.03125]:
            candidate=problem.project(z+fraction*delta,lo,hi)
            r=problem.fun(candidate);candidate_cost=float(.5*r@r)
            if candidate_cost<cost and (best is None or candidate_cost<best[0]):
                best=(candidate_cost,candidate.copy(),fraction)
        if best is None:
            damping*=10
            problem.evaluate(z)
            history.append(dict(iteration=iteration,status='no decreasing step',damping=damping))
            if damping>3e4:break
            continue
        cost,z,fraction=best
        problem.evaluate(z)
        damping=max(.03,damping*.5)
        info=dict(iteration=iteration,cost=cost,step_fraction=fraction,
                  dynamics_residual_max=float(abs(problem.physics).max()),effort_ratio_max=float(abs(problem.torque_ratio).max()))
        history.append(info);print('PROJECTED_GN',json.dumps(info),flush=True)
        np.savez_compressed(OUT/'projected_checkpoint.npz',optimization_variables=z)
    return z,history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ground-poses-only', action='store_true')
    parser.add_argument('--screen-nfev',type=int,default=8)
    parser.add_argument('--refine-nfev',type=int,default=40)
    parser.add_argument('--sqp-iterations',type=int,default=3)
    parser.add_argument('--landing-window-start',type=float,default=1.06)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT/'outputs/humanoid_light_conversion/scene.xml'))
    qland, qabs, report = ground_seeds(model)
    np.savez_compressed(OUT/'ground_seeds.npz', landing=qland, absorption=qabs)
    (OUT/'ground_seeds.json').write_text(json.dumps(report, indent=2)+'\n')
    if args.ground_poses_only:
        print(json.dumps(report, indent=2), flush=True)
        return
    motion=dyn.load_asfamc(dyn.cmu.DATA/'raw/88.asf',dyn.cmu.DATA/'raw/88_01.amc',dyn.cmu.DATA/'raw/88_01.c3d.header')
    candidates=[]
    for phases in [(1.34,1.58,1.84),(1.30,1.60,1.88),(1.34,1.64,1.92),(1.38,1.68,1.98),(1.42,1.72,2.04)]:
        result=optimize_case(model,motion,qland,qabs,report,phases,args.screen_nfev,args.landing_window_start)
        print('TIMING_SCREEN',json.dumps(result[4]),flush=True)
        candidates.append(result)
    problem,z,lo,hi,chosen,seed=min(candidates,key=lambda item:item[4]['physical_score'])
    (OUT/'timing_search.json').write_text(json.dumps(dict(candidates=[item[4] for item in candidates],selected=chosen),indent=2)+'\n')
    started=time.perf_counter()
    fit,z=fit_free(problem,z,lo,hi,args.refine_nfev)
    z=problem.project(z,lo,hi)
    z,sqp=dyn.polish(problem,z,lo,hi,args.sqp_iterations)
    problem.evaluate(z)
    _,q,forces,impulses=problem.unpack(z)
    velocity,acceleration,*_=dyn.velocity_acceleration(q,.02)
    if 'initial_velocity' in problem.plan:
        velocity[0]=problem.plan['initial_velocity']
        acceleration[0]=acceleration[1]
    problem.plan['metadata'].update(optimized_foot_positions_first=problem.plan['foot_pos'][problem.td].tolist(),
        optimized_foot_positions_second=problem.plan['foot_pos'][problem.td2].tolist(),
        optimized_foot_rotation_first=problem.plan['foot_rot'][problem.td].tolist(),
        optimized_foot_rotation_second=problem.plan['foot_rot'][problem.td2].tolist(),
        source_touchdown_seconds=1.34,contact_time_selected_by_search=True)
    np.savez_compressed(OUT/'88_01_com_first.npz',qpos=q,qvel=velocity,qacc=acceleration,times=problem.plan['times'],
        frequency=np.asarray(50.),joint_names=np.array(['root']+[model.joint(i).name for i in range(1,model.njnt)]),
        split_points=np.array([0,len(q)]),source_contact=problem.plan['contacts'],
        contact_forces_world=forces,contact_impulses_world=impulses,foot_targets=problem.plan['foot_pos'],
        foot_rotations=problem.plan['foot_rot'],com_target=problem.plan['com'],body_rotation_target=problem.plan['rotation'],
        optimization_variables=z,training_ready=np.asarray(False))
    np.savez_compressed(OUT/'seed.npz',qpos=seed,com_target=problem.plan['com'],body_rotation_target=problem.plan['rotation'])
    (OUT/'phase_plan.json').write_text(json.dumps(problem.plan['metadata'],indent=2)+'\n')
    solver=dict(selected=chosen,refinement_nfev=fit.nfev,sqp=sqp,cost=float(.5*problem.cache_r@problem.cache_r),
                max_dynamics_residual=float(abs(problem.physics).max()),max_effort_ratio=float(abs(problem.torque_ratio).max()),
                seconds=time.perf_counter()-started)
    (OUT/'solver.json').write_text(json.dumps(solver,indent=2)+'\n')
    print('RESULT',json.dumps(solver),flush=True)


if __name__ == '__main__':
    main()
