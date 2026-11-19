"""Independent momentum/effort/contact screening, plus a free-base rollout.

Animation fit, collocation convergence, and successful physical tracking are
separate results. No outcome of this script silently marks a clip trainable.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import sparse

import optimize_humanoid_light_cmu_dynamics as opt
import retarget_humanoid_light_lafan as ik
import validate_humanoid_light_cmu_landing as geometry_check

ROOT, OUT = opt.ROOT, opt.OUT


class TorqueTracker:
    """Small inverse-dynamics feedback QP; output is motor torque only.

    QP contact forces are predictions, NEVER applied to the simulator. The
    actual floor reaction is computed exclusively by MuJoCo's contact solver.
    """
    def __init__(self, model):
        sys.path.insert(0, str(opt.kin.OUT/'python_deps'))
        import osqp
        self.osqp = osqp
        self.model = model
        self.feet = set(opt.sole_points(model)[0])
        self.effort = np.zeros(26)
        for i, j in enumerate(model.actuator_trnid[:, 0]):
            self.effort[model.jnt_dofadr[j]-6] = model.actuator_ctrlrange[i, 1]
        self.bw = float(model.body_mass.sum()*9.81)
        self.failures = 0
        self.calls = 0

    def control(self, data, goal, velocity, acceleration):
        model = self.model
        mujoco.mj_forward(model, data)
        mass, bias = np.empty((32, 32)), np.empty(32)
        mujoco.mj_fullM(model, data, mass)
        mujoco.mj_rne(model, data, 0, bias)
        bias -= data.qfrc_passive
        desired = acceleration.copy()
        desired[:3] += 100*(goal[:3]-data.qpos[:3])+20*(velocity[:3]-data.qvel[:3])
        now = R.from_quat(data.qpos[3:7], scalar_first=True).as_matrix()
        target = R.from_quat(goal[3:7], scalar_first=True).as_matrix()
        desired[3:6] = now.T@target@acceleration[3:6]+80*R.from_matrix(now.T@target).as_rotvec()+16*(now.T@target@velocity[3:6]-data.qvel[3:6])
        desired[6:] += 100*(goal[7:]-data.qpos[7:])+20*(velocity[6:]-data.qvel[6:])
        points = []
        floor = model.geom('floor').id
        for i in range(data.ncon):
            contact = data.contact[i]
            if floor not in (contact.geom1, contact.geom2) or contact.dist > .003:
                continue
            other = contact.geom2 if contact.geom1 == floor else contact.geom1
            body = int(model.geom_bodyid[other])
            if body not in self.feet:
                continue
            if any(old_body == body and np.linalg.norm(point-contact.pos) < .008 for old_body, point in points):
                continue
            if sum(old_body == body for old_body, _ in points) >= 4:
                continue
            points.append((body, contact.pos.copy()))
        count = len(points)
        jacobians, drift, target_accel = [], [], []
        for body, point in points:
            jac, derivative = np.empty((3, 32)), np.empty((3, 32))
            mujoco.mj_jac(model, data, jac, None, point, body)
            mujoco.mj_jacDot(model, data, derivative, None, point, body)
            speed = jac@data.qvel
            desired_point = -40*speed
            desired_point[2] -= 3000*(point[2]+.001)
            jacobians.append(jac)
            drift.extend(derivative@data.qvel)
            target_accel.extend(desired_point)
        contact_j = np.concatenate(jacobians) if count else np.zeros((0, 32))
        generalized = contact_j.T*self.bw
        width = 32+3*count
        weights = np.r_[np.full(3, 8.), np.full(3, 6.), np.ones(26)]
        cost_a = sparse.hstack([sparse.diags(weights), sparse.csr_matrix((32, 3*count))], format='csr')
        goals = weights*desired
        if count:
            cost_a = sparse.vstack([cost_a, sparse.hstack([10*sparse.csr_matrix(contact_j), sparse.csr_matrix((3*count, 3*count))])], format='csr')
            goals = np.r_[goals, 10*(np.array(target_accel)-drift)]
        dynamics = sparse.csr_matrix(np.c_[mass, -generalized])
        matrices = [dynamics[:6], dynamics[6:]]
        lows = [-bias[:6], -self.effort-bias[6:]]
        highs = [-bias[:6], self.effort-bias[6:]]
        if count:
            cone = np.array([[1, 0, -.7], [-1, 0, -.7], [0, 1, -.7], [0, -1, -.7], [0, 0, -1.]])
            matrices.append(sparse.hstack([sparse.csr_matrix((5*count, 32)), sparse.kron(sparse.eye(count), cone)], format='csr'))
            lows.append(np.full(5*count, -np.inf))
            highs.append(np.zeros(5*count))
        qp = self.osqp.OSQP()
        qp.setup(P=sparse.triu(cost_a.T@cost_a+sparse.diags(np.r_[np.full(32, 1e-4), np.full(3*count, .001)]), format='csc'),
            q=-np.asarray(cost_a.T@goals).ravel(), A=sparse.vstack(matrices, format='csc'),
            l=np.concatenate(lows), u=np.concatenate(highs), max_iter=2000, time_limit=.04,
            eps_abs=1e-5, eps_rel=1e-5, verbose=False)
        result = qp.solve(raise_error=False)
        self.calls += 1
        if result.info.status_val not in (1, 2):
            self.failures += 1
            return None
        force = mass@result.x[:32]+bias-generalized@result.x[32:]
        return np.clip(force[6:], -self.effort, self.effort)


def screen(model, q, contacts, forces=None, impulses=None, phase_plan=None):
    """Recompute all dynamics from numeric poses, not optimizer residuals."""
    n = len(q)
    dt = .02
    v, a, *_ = opt.velocity_acceleration(q, dt)
    origin=0. if phase_plan is None else phase_plan.get('time_origin_seconds',0.)
    if phase_plan is not None and 'initial_velocity' in phase_plan:
        v[0]=np.asarray(phase_plan['initial_velocity'])
        a[0]=a[1]
    dyn = copy.deepcopy(model)
    dyn.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    data = mujoco.MjData(dyn)
    mass = float(model.body_mass.sum())
    bw = mass*9.81
    feet, points = opt.sole_points(dyn)
    effort = np.zeros(26)
    speed_limits = opt.kin.json_load(ik.OUTPUT/'light_profile.json')['velocity_limits_joint_coordinate_rad_s']
    speed = np.array([speed_limits[model.joint(i).name] for i in range(1, model.njnt)])
    for i in range(model.nu):
        j = model.actuator_trnid[i, 0]
        effort[model.jnt_dofadr[j]-6] = model.actuator_ctrlrange[i, 1]
    has_contact_fit = forces is not None
    if forces is None:
        forces = np.zeros((n, 8, 3))
    if impulses is None:
        impulses = np.zeros_like(forces)
    meshes, _ = ik.geometry(model)
    mesh_dict = dict(meshes)
    pairs = ik.self_collision_pairs(model)
    com, momentum, angular, root_wrench, torques = [], [], [], [], []
    impact_energy = []
    bottoms, foot_heights, foot_world, pairs_gap, point_velocities = [], [], [], [], []
    for t in range(n):
        data.qpos[:], data.qvel[:], data.qacc[:] = q[t], v[t], a[t]
        mujoco.mj_inverse(dyn, data)
        inverse = data.qfrc_inverse.copy()
        mujoco.mj_comVel(dyn, data)
        mujoco.mj_subtreeVel(dyn, data)
        com.append(data.subtree_com[1].copy())
        momentum.append(mass*data.subtree_linvel[1].copy())
        angular.append(data.subtree_angmom[1].copy())
        contact_generalized = np.zeros(32)
        impulse_generalized = np.zeros(32)
        frame_points = []
        frame_velocity = []
        for side, body in enumerate(feet):
            world = points[side]@data.xmat[body].reshape(3, 3).T+data.xpos[body]
            frame_points.extend(world)
            for k, point in enumerate(world):
                f = forces[t, side*4+k]+impulses[t, side*4+k]/dt
                mujoco.mj_applyFT(dyn, data, f, np.zeros(3), point, body, contact_generalized)
                mujoco.mj_applyFT(dyn, data, impulses[t, side*4+k], np.zeros(3), point, body, impulse_generalized)
                jp = np.empty((3, 32))
                mujoco.mj_jac(dyn, data, jp, None, point, body)
                frame_velocity.append(jp@v[t])
        if t > 0 and np.linalg.norm(impulse_generalized) > 1e-7:
            mass_matrix = np.empty((32, 32))
            mujoco.mj_fullM(dyn, data, mass_matrix)
            incoming = np.r_[(q[t, :3]-q[t-1, :3])/dt, np.zeros(3), (q[t, 7:]-q[t-1, 7:])/dt]
            rotation_now = R.from_quat(q[t, 3:7], scalar_first=True).as_matrix()
            rotation_prev = R.from_quat(q[t-1, 3:7], scalar_first=True).as_matrix()
            incoming[3:6] = -R.from_matrix(rotation_now.T@rotation_prev).as_rotvec()/dt
            change = float(incoming@impulse_generalized+.5*impulse_generalized@np.linalg.solve(mass_matrix, impulse_generalized))
            impact_energy.append(dict(time_seconds=origin+t*dt, impulse_only_kinetic_energy_change_J=change))
        required = inverse-contact_generalized
        root_wrench.append(required[:6])
        torques.append(required[6:])
        heights = []
        sole = []
        for gid, vertices in meshes:
            height = float((vertices@data.geom_xmat[gid].reshape(3, 3)[2]+data.geom_xpos[gid, 2]).min())
            heights.append(height)
            if model.geom_bodyid[gid] in feet:
                sole.append(height)
        bottoms.append(min(heights))
        foot_heights.append(sole)
        foot_world.append(frame_points)
        point_velocities.append(frame_velocity)
        distances = [opt.kin.checked_mesh_distance(dyn, data, i, j, mesh_dict)[0] for i, j in pairs]
        pairs_gap.append(min(distances))
    com, momentum, angular = np.array(com), np.array(momentum), np.array(angular)
    wrench, torque = np.array(root_wrench), np.array(torques)
    bottoms, foot_heights, foot_world = np.array(bottoms), np.array(foot_heights), np.array(foot_world)
    airborne = ~contacts.any(axis=1)
    # Exclude one node at phase boundaries from span tests; the collocation
    # residual itself still includes ALL interior nodes, including touchdown.
    interior_air = airborne.copy()
    interior_air[1:] &= airborne[:-1]
    interior_air[:-1] &= airborne[1:]
    interior_air[[0, -1]] = False
    spans = []
    for start, end in ik.contact_intervals(interior_air):
        if end-start < 3:
            continue
        values = angular[start:end]
        mean = np.mean(values, axis=0)
        span = float(np.linalg.norm(np.ptp(values, axis=0)))
        ballistic = np.diff(com[start-1:end+1], n=2, axis=0)/dt**2-[0, 0, -9.81]
        spans.append(dict(start_seconds=origin+start*dt, end_seconds=origin+(end-1)*dt,
            angular_momentum_span_kg_m2_s=span, relative_span=span/max(1., np.linalg.norm(mean)),
            com_non_gravity_acceleration_max_m_s2=float(np.linalg.norm(ballistic, axis=1).max())))
    contact_force = forces.sum(axis=1)
    impulse = impulses.sum(axis=1)
    # Compare incoming/outgoing midpoint momentum with the actual integrated
    # contact reaction. A low instantaneous inverse-dynamics residual alone
    # is not evidence that the velocity jump is supported by an impulse.
    interval_h = []
    for t in range(n-1):
        middle = .5*(q[t]+q[t+1])
        left_r = R.from_quat(q[t, 3:7], scalar_first=True)
        relative = left_r.inv()*R.from_quat(q[t+1, 3:7], scalar_first=True)
        middle[3:7] = (left_r*R.from_rotvec(.5*relative.as_rotvec())).as_quat(scalar_first=True)
        interval_v = np.empty(32)
        mujoco.mj_differentiatePos(dyn, interval_v, dt, q[t], q[t+1])
        data.qpos[:] = middle
        mujoco.mj_kinematics(dyn, data)
        mujoco.mj_comPos(dyn, data)
        matrix = np.empty((3, 32))
        mujoco.mj_angmomMat(dyn, data, matrix, 1)
        interval_h.append(matrix@interval_v)
    interval_h = np.asarray(interval_h)
    impact_balance = []
    touchdown_nodes = np.flatnonzero(contacts[1:].any(axis=1)&~contacts[:-1].any(axis=1))+1
    for t in touchdown_nodes:
        if not 0 < t < n-1:
            continue
        incoming_p = mass*(com[t]-com[t-1])/dt
        outgoing_p = mass*(com[t+1]-com[t])/dt
        predicted_p = mass*dyn.opt.gravity*dt+contact_force[t]*dt+impulse[t]
        arms = foot_world[t]-com[t]
        predicted_h = np.cross(arms, forces[t]*dt+impulses[t]).sum(axis=0)
        delta_h = interval_h[t]-interval_h[t-1]
        impact_balance.append(dict(time_seconds=float(origin+t*dt),
            observed_linear_momentum_change_N_s=(outgoing_p-incoming_p).tolist(),
            predicted_linear_impulse_N_s=predicted_p.tolist(),
            observed_angular_momentum_change_kg_m2_s=delta_h.tolist(),
            predicted_angular_impulse_kg_m2_s=predicted_h.tolist(),
            linear_relative_residual=float(np.linalg.norm(outgoing_p-incoming_p-predicted_p)/max(1., np.linalg.norm(incoming_p), np.linalg.norm(predicted_p))),
            angular_relative_residual=float(np.linalg.norm(delta_h-predicted_h)/max(1., np.linalg.norm(interval_h[t-1]), np.linalg.norm(predicted_h)))))
    active_f = np.linalg.norm(forces, axis=-1) > 1.
    active_j = np.linalg.norm(impulses, axis=-1) > .1
    loaded = active_f | active_j
    separation = abs(foot_world[:, :, 2][loaded])
    sustained = forces[:, :, 2] > .02*bw
    for t in np.flatnonzero(np.linalg.norm(impulses, axis=(1, 2)) > .1):
        sustained[max(0, t-1):min(n, t+2)] = False
    loaded_speed = np.linalg.norm(np.array(point_velocities)[:, :, :2], axis=-1)[sustained]
    friction = []
    for values in [forces, impulses]:
        friction.append(float(np.maximum(np.linalg.norm(values[:, :, :2], axis=-1)-values[:, :, 2], 0.).max()))
    limit_excess = np.maximum(model.jnt_range[1:, 0]-q[:, 7:], q[:, 7:]-model.jnt_range[1:, 1])
    limits_ratio = float((abs(np.diff(q[:, 7:], axis=0)/dt)/speed).max())
    rotations = R.from_quat(q[:, 3:7], scalar_first=True).as_matrix()
    turn = float(np.rad2deg(R.from_matrix(np.swapaxes(rotations[:-1], -1, -2)@rotations[1:]).as_rotvec()[:, 1].sum()))
    terminal_tilt = float(np.rad2deg(np.arccos(np.clip(rotations[-1, 2, 2], -1., 1.))))
    hop_start,hop_end=(1.6,1.82) if phase_plan is None else (
        phase_plan['rebound_takeoff_seconds']+.02,phase_plan['rebound_touchdown_seconds']-.02)
    rebound_frames = (origin+np.arange(n)*dt > hop_start)&(origin+np.arange(n)*dt < hop_end)
    rebound_clearance = float(np.min(foot_heights[rebound_frames], axis=1).max()) if rebound_frames.any() else 0.
    root_scale = np.r_[np.full(3, bw), np.full(3, bw*.6)]
    root_max = float((abs(wrench[1:-1])/root_scale).max())
    effort_ratio = float((abs(torque[1:-1])/effort).max())
    checks = dict(
        floating_base_dynamics_residual_below_1pct=root_max < .01,
        original_motor_effort_limits_respected=effort_ratio < 1.001,
        original_joint_ranges_respected=float(limit_excess.max()) < 1e-6,
        original_joint_speeds_respected=limits_ratio < 1.001,
        forces_and_impulses_push_only=float(min(forces[:, :, 2].min(), impulses[:, :, 2].min())) >= -1e-5,
        friction_within_actual_floor_mu_1=max(friction) < 1e-4,
        no_contact_force_while_airborne=float(abs(forces[airborne]).max(initial=0.)) < 1e-5 and float(abs(impulses[airborne]).max(initial=0.)) < 1e-5,
        loaded_contact_points_near_ground=(float(separation.max()) if len(separation) else 0.) < .005,
        sustained_contact_material_slip_below_5cm_s=(float(loaded_speed.max()) if len(loaded_speed) else 0.) < .05,
        all_robot_floor_hulls_clear=float(bottoms.min()) > -.001,
        selected_self_hulls_clear=float(min(pairs_gap)) > .0002,
        airborne_angular_momentum_span_below_5pct=all(s['relative_span'] < .05 for s in spans),
        airborne_com_non_gravity_acceleration_below_0_5=all(s['com_non_gravity_acceleration_max_m_s2'] < .5 for s in spans),
        instantaneous_impulses_do_not_inject_energy=all(s['impulse_only_kinetic_energy_change_J'] < 1. for s in impact_energy))
    checks.update(complete_backflip=-400 < turn < -320,
        settled_terminal_pose=terminal_tilt < 10 and float(abs(v[-5:]).max()) < .1,
        secondary_airborne_rebound_preserved=rebound_clearance > .01,
        touchdown_linear_impulse_balance_below_5pct=all(item['linear_relative_residual'] < .05 for item in impact_balance),
        touchdown_angular_impulse_balance_below_5pct=all(item['angular_relative_residual'] < .05 for item in impact_balance))
    if phase_plan is not None and phase_plan.get('partial_motion',False):
        checks.pop('complete_backflip')
        checks['rotation_in_window_matches_target']=abs(turn-phase_plan['expected_rotation_in_window_degrees'])<20
    report = dict(checks=checks, collocation_passed=all(checks.values()), root_dynamics_max_relative=root_max,
        contact_forces_provided=has_contact_fit,
        support_phase_dynamics_interpretation='explicit fitted contact forces' if has_contact_fit else 'support forces not estimated; use airborne momentum tests for baseline conclusions',
        required_motor_torque_to_limit_max=effort_ratio, interval_joint_speed_to_limit_max=limits_ratio,
        minimum_floor_height_m=float(bottoms.min()), minimum_selected_self_gap_m=float(min(pairs_gap)),
        max_loaded_contact_height_error_m=float(separation.max()) if len(separation) else None,
        max_sustained_contact_material_xy_speed_m_s=float(loaded_speed.max()) if len(loaded_speed) else None,
        accumulated_pitch_degrees=turn, terminal_tilt_degrees=terminal_tilt,
        rebound_two_sole_clearance_max_m=rebound_clearance,
        airborne_spans=spans, impulse_energy=impact_energy, touchdown_impulse_balance=impact_balance,
        touchdown_impulses=[dict(time_seconds=float(origin+t*dt),
          net_impulse_N_s=impulse[t].tolist(), cell_contact_force_impulse_N_s=(contact_force[t]*dt).tolist())
          for t in np.flatnonzero(np.linalg.norm(impulse, axis=1) > .1)])
    series = dict(com=com, linear_momentum=momentum, angular_momentum=angular,
        root_required_external_wrench=wrench, joint_torque_required=torque,
        contact_force=contact_force, contact_impulse=impulse, interval_angular_momentum=interval_h, sole_heights=foot_heights,
        floor_heights=bottoms, self_gap=np.array(pairs_gap), qvel=v, qacc=a)
    return report, series


def rollout(model, q, series, start=.76, feedback=True, initial_velocity=None, time_origin=0.):
    """Ideal bounded joint torque diagnostic; root state never overwritten.

    Initialize ONCE from the reference at start, then use mj_step exclusively.
    No root wrench, state reset, foot anchor, trajectory projection or weld.
    This is not the source delayed/coupled hardware actuator implementation.
    """
    sim = copy.deepcopy(model)
    sim.opt.timestep = .001
    sim.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(sim)
    begin = round(start*50)
    data.qpos[:] = q[begin]
    data.qvel[:] = series['qvel'][begin] if initial_velocity is None else initial_velocity
    mujoco.mj_forward(sim, data)
    joint_columns = np.array([sim.jnt_dofadr[j]-6 for j in sim.actuator_trnid[:, 0]])
    effort = sim.actuator_ctrlrange[:, 1].copy()
    tracker = TorqueTracker(sim) if feedback else None
    feedback_torque = None
    kp, kd = np.full(26, 100.), np.full(26, 4.)
    for i, j in enumerate(sim.actuator_trnid[:, 0]):
        name = sim.joint(j).name
        if 'ankle' in name:
            kp[i], kd[i] = 80., 3.
        elif 'head' in name:
            kp[i], kd[i] = 20., .5
        elif any(part in name for part in ['elbow', 'wrist', 'shoulder_yaw']):
            kp[i], kd[i] = 50., 2.
    poses, velocities, controls, ground_force = [], [], [], []
    frames = len(q)-begin
    speed_profile = opt.kin.json_load(ik.OUTPUT/'light_profile.json')['velocity_limits_joint_coordinate_rad_s']
    speed_limit = np.array([speed_profile[sim.joint(j).name] for j in range(1, sim.njnt)])
    effort_ratio, speed_ratio, position_excess = 0., 0., 0.
    for step in range((frames-1)*20+1):
        s = step/20+begin
        i = min(int(s), len(q)-2)
        u = s-i
        desired = (1-u)*q[i, 7:]+u*q[i+1, 7:]
        desired_v = (1-u)*series['qvel'][i, 6:]+u*series['qvel'][i+1, 6:]
        ff = (1-u)*series['joint_torque_required'][i]+u*series['joint_torque_required'][i+1]
        ctrl = ff[joint_columns]+kp*(desired[joint_columns]-data.qpos[7:][joint_columns])+kd*(desired_v[joint_columns]-data.qvel[6:][joint_columns])
        if tracker is not None and step % 10 == 0:
            goal = q[i].copy()
            goal[:3], goal[7:] = (1-u)*q[i, :3]+u*q[i+1, :3], desired
            goal[3:7] = (R.from_quat(q[i, 3:7], scalar_first=True)*R.from_rotvec(u*R.from_matrix(R.from_quat(q[i, 3:7], scalar_first=True).as_matrix().T@R.from_quat(q[i+1, 3:7], scalar_first=True).as_matrix()).as_rotvec())).as_quat(scalar_first=True)
            vel = (1-u)*series['qvel'][i]+u*series['qvel'][i+1]
            acc = (1-u)*series['qacc'][i]+u*series['qacc'][i+1]
            feedback_torque = tracker.control(data, goal, vel, acc)
        if feedback_torque is not None:
            ctrl = feedback_torque[joint_columns]
        data.ctrl[:] = np.clip(ctrl, -effort, effort)
        effort_ratio = max(effort_ratio, float((abs(data.ctrl)/effort).max()))
        speed_ratio = max(speed_ratio, float((abs(data.qvel[6:])/speed_limit).max()))
        position_excess = max(position_excess, float(np.maximum(sim.jnt_range[1:, 0]-data.qpos[7:], data.qpos[7:]-sim.jnt_range[1:, 1]).max()))
        if step % 20 == 0:
            poses.append(data.qpos.copy())
            velocities.append(data.qvel.copy())
            controls.append(data.ctrl.copy())
            total = np.zeros(3)
            for k in range(data.ncon):
                contact = data.contact[k]
                if sim.geom('floor').id not in (contact.geom1, contact.geom2):
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(sim, data, k, force)
                world = contact.frame.reshape(3, 3).T@force[:3]
                if contact.geom2 == sim.geom('floor').id:
                    world *= -1
                total += world
            ground_force.append(total)
        if step < (frames-1)*20:
            mujoco.mj_step(sim, data)
    poses, velocities = np.array(poses), np.array(velocities)
    root_error = np.linalg.norm(poses[:, :3]-q[begin:, :3], axis=1)
    rotation = R.from_quat(poses[:, 3:7], scalar_first=True).as_matrix()
    tilt = np.rad2deg(np.arccos(np.clip(rotation[:, 2, 2], -1., 1.)))
    post = np.arange(len(poses))/50+start+time_origin > 2.2
    report = dict(start_seconds=start, initialize_from_reference_once=True, external_root_force=False,
        start_world_seconds=start+time_origin,explicit_initial_velocity=initial_velocity is not None,
        root_position_error_max_m=float(root_error.max()),
        terminal_root_height_m=float(poses[-1, 2]), terminal_root_tilt_deg=float(tilt[-1]),
        tracking_passed=bool(np.max(root_error) < .12 and np.max(tilt[post]) < 30 and poses[-1, 2] > .5 and speed_ratio < 1.05 and position_excess < .01),
        maximum_actual_motor_effort_ratio=effort_ratio,
        maximum_actual_joint_speed_ratio=speed_ratio, maximum_actual_joint_limit_excess_rad=position_excess,
        controller='bounded inverse-dynamics feedback QP' if feedback else 'bounded feedforward/PD diagnostic',
        source_delay_and_motor_coupling_modeled=False,
        feedback_calls=tracker.calls if tracker else 0, feedback_failures=tracker.failures if tracker else 0,
        integration_timestep_seconds=.001)
    return report, dict(qpos=poses, qvel=velocities, control=np.array(controls), ground_force=np.array(ground_force),
                        start_seconds=np.asarray(start), frequency=np.asarray(50.))


def render_diagnostics(model, q, previous, after_series, before_series, plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    time = np.arange(len(q))/50
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)
    axes[0].plot(time, before_series['angular_momentum'][:, 1], '--', label='Previous: angular momentum Y')
    axes[0].plot(time, after_series['angular_momentum'][:, 1], label='Dynamics candidate: angular momentum Y')
    axes[0].set(ylabel='kg m^2 / s')
    axes[1].plot(time, before_series['sole_heights'][:, 0], '--', label='Previous left sole')
    axes[1].plot(time, after_series['sole_heights'][:, 0], label='Candidate left sole')
    axes[1].plot(time, after_series['sole_heights'][:, 1], label='Candidate right sole')
    axes[1].set(ylabel='Sole height (m)', ylim=(-.03, .35))
    axes[2].plot(time, after_series['contact_force'][:, 2]/float(model.body_mass.sum()*9.81), label='Continuous vertical ground force / BW')
    for t in np.flatnonzero(np.linalg.norm(after_series['contact_impulse'], axis=1) > .1):
        axes[2].annotate(f"Jz={after_series['contact_impulse'][t, 2]:.1f} N s", (time[t], 0), xytext=(time[t]-.1, 2.))
    axes[2].set(ylabel='Body weights')
    axes[3].plot(time, np.linalg.norm(before_series['root_required_external_wrench'][:, 3:6], axis=1), '--', label='Previous required external root torque')
    axes[3].plot(time, np.linalg.norm(after_series['root_required_external_wrench'][:, 3:6], axis=1), label='Candidate unexplained root torque')
    axes[3].set(ylabel='Root torque (N m)', xlabel='Time (s)')
    for axis in axes:
        for key in ['takeoff_seconds', 'touchdown_seconds', 'rebound_takeoff_seconds', 'rebound_touchdown_seconds']:
            axis.axvline(plan[key], color='black', linewidth=.5)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(OUT/'momentum_contact_audit.png', dpi=140)
    plt.close(fig)
    actual = np.load(OUT/'physics_rollout.npz', allow_pickle=False)
    actual_q, first = actual['qpos'], round(float(actual['start_seconds'])*50)
    status = opt.kin.json_load(OUT/'validation.json')['status']
    option, camera = mujoco.MjvOption(), mujoco.MjvCamera()
    option.geomgroup[:] = 0
    option.geomgroup[[0, 2]] = 1
    data = mujoco.MjData(model)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    labels = ['Previous pose-only reference', 'Inverse-dynamics candidate', 'Actual MuJoCo / bounded motor torques']
    with mujoco.Renderer(model, height=440, width=560) as renderer, imageio.get_writer(
        OUT/'landing_physics_diagnostic.mp4', fps=25, codec='libx264', quality=7, macro_block_size=1) as writer:
        for i in range(round(1.08*50), min(len(q), round(2.4*50))):
            canvas = Image.new('RGB', (1680, 512), '#e9eef2')
            draw = ImageDraw.Draw(canvas)
            poses = [previous[i], q[i], actual_q[i-first]]
            for panel, pose in enumerate(poses):
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                camera.lookat[:] = [q[i, 0], q[i, 1], .75]
                camera.distance, camera.azimuth, camera.elevation = 3.6, 55., -12.
                renderer.update_scene(data, camera=camera, scene_option=option)
                canvas.paste(Image.fromarray(renderer.render()), (panel*560, 35))
                draw.text((panel*560+8, 8), labels[panel], fill='#17242d', font=font)
            draw.text((10, 480), f'{i/50:.2f}s | 0.5x playback | {status} | Training ready: FALSE', fill='#9b2929', font=font)
            writer.append_data(np.asarray(canvas))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    z = np.load(OUT/'88_01_candidate.npz', allow_pickle=False)
    q, contacts = z['qpos'], z['source_contact']
    plan = opt.kin.json_load(OUT/'phase_plan.json')
    model_path = ROOT/'outputs/humanoid_light_conversion/scene.xml'
    model = mujoco.MjModel.from_xml_path(str(model_path))
    after, after_series = screen(model, q, contacts, z['contact_forces_world'], z['contact_impulses_world'])
    subframes = geometry_check.subframe_check(model, q)
    after['checks'].update(
        interpolated_robot_floor_hulls_clear=subframes['minimum_floor_height_m'] > -.001,
        interpolated_selected_self_hulls_clear=subframes['minimum_selected_self_gap_m'] > .0002,
        interpolated_mesh_queries_resolved=subframes['unresolved_zero_distance_queries'] == 0)
    after['collocation_passed'] = all(after['checks'].values())
    previous = np.load(opt.kin.OUT/'88_01_optimized.npz', allow_pickle=False)['qpos'].astype(float)
    old_contacts = np.load(opt.kin.OUT/'88_01_optimized.npz', allow_pickle=False)['source_contact']
    before, before_series = screen(model, previous, old_contacts)
    simulated, data = rollout(model, q, after_series)
    np.savez_compressed(OUT/'physics_rollout.npz', **data)
    np.savez_compressed(OUT/'metrics.npz', **after_series)
    report = dict(status='collocation_passed_pending_full_validation' if after['collocation_passed'] else 'dynamics_candidate_rejected',
        before=before, after=after, subframe_checks=subframes, forward_rollout=simulated, training_ready=False,
        dynamic_feasibility_verified=False, plan=plan,
        robot_xml_sha256=ik.digest(ROOT/'outputs/humanoid_light_conversion/humanoid_light.xml'),
        optimizer_sha256=ik.digest(Path(opt.__file__)), validator_sha256=ik.digest(Path(__file__)),
        limitations=['20 ms inverse-dynamics collocation approximates touchdown impulse over a finite time cell.',
                    'Source-informed touchdown/rebound timing is fixed; contact-mode search is not performed.',
                    'Selected self-collision hull pairs are screened, not every possible body pair.',
                    'Forward rollout uses bounded ideal joint torques, not the source delayed/coupled motor controller.',
                    'Passing pose-only validation is not evidence of dynamic feasibility.'])
    (OUT/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    previous_report_path = opt.kin.OUT/'validation.json'
    previous_report = opt.kin.json_load(previous_report_path)
    previous_report.update(status='passed_kinematic_checks_failed_momentum_screening',
        dynamic_feasibility_verified=False, training_ready=False,
        subsequent_dynamics_audit=dict(report=str(OUT/'validation.json'),
            airborne_spans=before['airborne_spans'],
            conclusion='Pose/geometry checks passed, but airborne momentum balance did not. Do not use as a validated physical reference.'))
    previous_report_path.write_text(json.dumps(previous_report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    if not args.no_render:
        render_diagnostics(model, q, previous, after_series, before_series, plan)


if __name__ == '__main__':
    main()
