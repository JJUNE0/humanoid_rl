"""Kinematic LAFAN experiment: trunk tilt/twist and hand/foot velocities.

Default: no joint-pose, bone-direction or Cartesian position reference costs.
Optional --pose-weight adds a weak anatomical bone-direction servo, without
joint-angle or Cartesian position targets.
Optional --intermediate-weight / --shoulder-weight add human/robot landmark
linear-velocity matching; they do not change joint angular-speed bounds.
--hip-weight similarly matches the two hip landmark linear velocities.
A neutral initial robot pose is an initial condition, not a tracked reference.
Original joint limits/speeds and nonlinear floor/selected-self clearances apply.
This is differential IK playback, not reinforcement learning or a torque test.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import gaussian_filter1d

import optimize_humanoid_light_cmu_landing as kin
import retarget_humanoid_light_lafan as source
import validate_humanoid_light_cmu_landing as geometry_checks
from humanoid_light_collision import MeshCollisionIndex,body_collision_pairs

ROOT = source.ROOT
OUT = ROOT / 'outputs/humanoid_light_lafan_velocity'
EFFECTORS = [('LeftHand', 'left_wrist_link'), ('RightHand', 'right_wrist_link'),
             ('LeftFoot', 'left_ankle_roll_link'), ('RightFoot', 'right_ankle_roll_link')]
INTERMEDIATES = [('LeftForeArm','left_elbow_link'),('RightForeArm','right_elbow_link'),
                 ('LeftLeg','left_knee_link'),('RightLeg','right_knee_link')]
SHOULDERS = [('LeftArm','left_shoulder_yaw_link'),('RightArm','right_shoulder_yaw_link')]
HIPS = [('LeftUpLeg','left_hip_pitch_link'),('RightUpLeg','right_hip_pitch_link')]
HIP_ROTATION_BODIES = ['left_hip_yaw_link','right_hip_yaw_link']
CONTACTS = [('LeftToe','left_ankle_roll_link',.04),('RightToe','right_ankle_roll_link',.04),
            ('LeftHand','left_wrist_link',.08),('RightHand','right_wrist_link',.08),
            ('LeftLeg','left_knee_link',.10),('RightLeg','right_knee_link',.10),
            ('Hips','pelvis_link',.13),('Spine2','torso_roll_link',.16)]


def rotation_velocity(rotations,fps):
    """Left-trivialized angular velocity, including batches of relative frames."""
    result=np.empty(rotations.shape[:-2]+(3,))
    for i in range(len(rotations)):
        a,b=max(0,i-1),min(len(rotations)-1,i+1)
        delta=rotations[b]@np.swapaxes(rotations[a],-1,-2)
        result[i]=R.from_matrix(delta).as_rotvec()/((b-a)/fps)
    return result


def interpolate_rotation(first,second,u):
    delta=R.from_matrix(np.swapaxes(first,-1,-2)@second).as_rotvec()
    return first@R.from_rotvec(u*delta).as_matrix()


def target_at_step(ref,frame,sub,substeps,include_rotation=False):
    u=(sub+.5)/substeps;u_next=(sub+1)/substeps;u0=sub/substeps
    mix=lambda key,alpha=u:(1-alpha)*ref[key][frame]+alpha*ref[key][frame+1]
    gravity=mix('gravity',u_next);gravity/=np.linalg.norm(gravity)
    bones=mix('bone_directions',u_next);bones/=np.linalg.norm(bones,axis=-1,keepdims=True)
    result=dict(target_velocity=mix('linear_velocity'),target_angular=mix('angular_velocity'),
        target_gravity=gravity,target_bones=bones,target_intermediates=mix('intermediate_linear_velocity'),
        target_shoulders=mix('shoulder_linear_velocity'),target_hips=mix('hip_linear_velocity'),
        target_height=mix('trunk_height_target',u0),target_height_rate=mix('trunk_height_rate'),
        contact_confidence=mix('contact_confidence'))
    if include_rotation:
        result.update(target_hip_rotation=interpolate_rotation(ref['hip_relative_rotation'][frame],ref['hip_relative_rotation'][frame+1],u0),
            target_hip_angular=mix('hip_relative_angular_velocity'),
            target_pelvis_rotation=interpolate_rotation(ref['pelvis_rotation'][frame],ref['pelvis_rotation'][frame+1],u0),
            target_pelvis_angular=mix('pelvis_angular_velocity'))
    return result


def height_contact_reference(model,human,names,body_rotation,scale,full_positions,full_names,extract,utils,fps):
    """Soft surface-contact cues; BVH provides no measured contact forces.

    Calibrate the tracked trunk landmark with the same upright human reference
    used for the morphology scale. Rotate its offset with the trunk, rather
    than adding a fixed vertical offset to lying poses.
    """
    floor=float(np.quantile(full_positions[:,[full_names.index('LeftToe'),full_names.index('RightToe')],2],.02)*scale)
    cache=getattr(height_contact_reference,'standing_reference',None)
    if cache is None:
        ref_names,ref_h,_,_,_=source.load_human(ROOT/'src/humanoid_retarget/datasets/lafan1/raw/walk1_subject1.bvh',extract,utils)
        ref_floor=float(np.quantile(ref_h[:,[ref_names.index('LeftToe'),ref_names.index('RightToe')],2],.02))
        cache=(float(ref_h[0,ref_names.index('Spine2'),2]),ref_floor)
        height_contact_reference.standing_reference=cache
    data=mujoco.MjData(model);data.qpos[:]=kin.foot_geometry(model)[3];mujoco.mj_forward(model,data)
    bid=model.body('torso_roll_link').id
    delta=data.xpos[bid,2]-(cache[0]-cache[1])*scale
    offset=data.xmat[bid].reshape(3,3).T @ np.array([0.,0.,delta])
    target=human[:,names.index('Spine2'),2]-floor+(body_rotation@offset)[:,2]
    points=human[:,[names.index(h) for h,_,_ in CONTACTS]]
    velocities=np.gradient(points,1/fps,axis=0,edge_order=2)
    limits=np.array([threshold for _,_,threshold in CONTACTS])
    confidence=np.clip((limits-(points[:,:,2]-floor))/(limits*.6),0.,1.)
    confidence*=np.clip((.35-np.linalg.norm(velocities[:,:,:2],axis=-1))/.20,0.,1.)
    confidence*=np.clip((.4-abs(velocities[:,:,2]))/.25,0.,1.)
    confidence=gaussian_filter1d(confidence,1.,axis=0,mode='nearest')
    return dict(trunk_height_target=target,trunk_height_rate=np.gradient(target,1/fps,edge_order=2),
                contact_confidence=confidence,source_floor_z_m=floor,
                trunk_height_local_offset_m=offset,contact_body_names=[b for _,b,_ in CONTACTS])


def references(path, start, duration, fps, model, extract, utils):
    names, positions, rotations, offsets, source_fps = source.load_human(path, extract, utils)
    times = start + np.arange(round(duration*fps)+1)/fps
    if times[-1] > (len(positions)-1)/source_fps:
        raise ValueError('Requested clip exceeds source duration')
    original_times = np.arange(len(positions))/source_fps
    human = np.empty((len(times), len(names), 3))
    for joint in range(len(names)):
        for axis in range(3):
            human[:, joint, axis] = np.interp(times, original_times, positions[:, joint, axis])
    profile = json.loads((source.OUTPUT/'light_profile.json').read_text())
    calibration = profile['rotation_offsets']['torso_roll_link']
    source_body = names.index(calibration['human'])
    body_rotation = Slerp(original_times, R.from_matrix(rotations[:, source_body]))(times).as_matrix()
    body_rotation = body_rotation @ R.from_quat(calibration['rotation_wxyz'], scalar_first=True).as_matrix()
    pelvis_rotation = Slerp(original_times, R.from_matrix(rotations[:, names.index('Hips')]))(times).as_matrix()
    pelvis_rotation = pelvis_rotation @ R.from_quat(
        profile['rotation_offsets']['pelvis_link']['rotation_wxyz'],scalar_first=True).as_matrix()
    initial_heading = np.arctan2(pelvis_rotation[0, 1, 0], pelvis_rotation[0, 0, 0])
    align = R.from_euler('z', -initial_heading).as_matrix()
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    robot_leg = sum(np.linalg.norm(data.xpos[model.body('left_'+b+'_link').id] -
                   data.xpos[model.body('left_'+a+'_link').id])
                   for a, b in [('hip_pitch', 'knee'), ('knee', 'ankle_pitch')])
    human_leg = sum(np.linalg.norm(offsets[names.index(n)]) for n in ['LeftLeg', 'LeftFoot'])
    scale = robot_leg/human_leg
    human = (human @ align.T)*scale
    origin_xy = human[0, names.index('Hips'), :2].copy()
    human[:, :, :2] -= origin_xy
    body_rotation = align @ body_rotation
    pelvis_rotation=align@pelvis_rotation
    thighs=[]
    for body in HIP_ROTATION_BODIES:
        mapping=profile['rotation_offsets'][body]
        matrix=Slerp(original_times,R.from_matrix(rotations[:,names.index(mapping['human'])]))(times).as_matrix()
        thighs.append(align@matrix@R.from_quat(mapping['rotation_wxyz'],scalar_first=True).as_matrix())
    thigh_rotation=np.stack(thighs,axis=1)
    hip_relative_rotation=np.swapaxes(pelvis_rotation,-1,-2)[:,None]@thigh_rotation
    selected = [source_body] + [names.index(h) for h, _ in EFFECTORS]
    velocity = np.gradient(human[:, selected], 1/fps, axis=0, edge_order=2)
    intermediate_velocity = np.gradient(human[:,[names.index(h) for h,_ in INTERMEDIATES]],
                                        1/fps,axis=0,edge_order=2)
    shoulder_velocity = np.gradient(human[:,[names.index(h) for h,_ in SHOULDERS]],
                                    1/fps,axis=0,edge_order=2)
    hip_velocity = np.gradient(human[:,[names.index(h) for h,_ in HIPS]],
                               1/fps,axis=0,edge_order=2)
    angular = np.empty((len(times), 3))
    for i in range(len(times)):
        a, b = max(0, i-1), min(len(times)-1, i+1)
        angular[i] = R.from_matrix(body_rotation[b] @ body_rotation[a].T).as_rotvec()/((b-a)/fps)
    gravity = np.einsum('tji,j->ti', body_rotation, [0., 0., -1.])
    bone_directions = np.stack([human[:,names.index(h2)]-human[:,names.index(h1)]
        for _,h1,h2,_,_ in source.bone_correspondences()],axis=1)
    bone_directions /= np.linalg.norm(bone_directions,axis=-1,keepdims=True)
    height_contact=height_contact_reference(model,human,names,body_rotation,scale,positions,names,extract,utils,fps)
    return dict(times=times-start, source_times=times, human=human, names=names,
                body_rotation=body_rotation, linear_velocity=velocity,
                angular_velocity=angular, gravity=gravity, scale=scale,
                bone_directions=bone_directions,
                intermediate_linear_velocity=intermediate_velocity,
                shoulder_linear_velocity=shoulder_velocity,
                hip_linear_velocity=hip_velocity,
                pelvis_rotation=pelvis_rotation,pelvis_angular_velocity=rotation_velocity(pelvis_rotation,fps),
                hip_relative_rotation=hip_relative_rotation,
                hip_relative_angular_velocity=rotation_velocity(hip_relative_rotation,fps),
                alignment=align, origin_xy=origin_xy, source_fps=source_fps,
                tracked_bodies=['torso_roll_link']+[b for _, b in EFFECTORS],**height_contact)


class VelocityIK:
    def __init__(self, model, profile, dt, pose_weight=0., pose_gain=2.,
                 intermediate_weight=0., shoulder_weight=0., hip_weight=0.,
                 height_weight=0.,contact_weight=0.,slip_weight=0.,
                 hip_rotation_weight=0.,pelvis_rotation_weight=0.,
                 joint_limit_weight=0.,self_collision_weight=0.,collision_coverage='selected'):
        from qpsolvers import solve_qp
        self.solve_qp = solve_qp
        self.model, self.dt = model, dt
        self.data = mujoco.MjData(model)
        self.meshes, _, _, self.initial, _, _ = kin.foot_geometry(model)
        self.vertices = dict(self.meshes)
        self.pairs = source.self_collision_pairs(model)
        self.structural_exclusions=[]
        if collision_coverage=='full':
            self.pairs,self.structural_exclusions=body_collision_pairs(model,self.meshes,kin.checked_mesh_distance)
        self.collision_coverage=collision_coverage
        self.collision_index=(MeshCollisionIndex(model,self.meshes,self.pairs,kin.checked_mesh_distance)
                              if collision_coverage=='full' else None)
        self._world_meshes=None;self._collision_entries=None
        self.joint_limit_weight,self.self_collision_weight=joint_limit_weight,self_collision_weight
        self.joint_margin=np.minimum(.1*(model.jnt_range[1:,1]-model.jnt_range[1:,0]),np.deg2rad(5.))
        self.self_clearance=.008
        self.bodies = [model.body(n).id for n in ['torso_roll_link']+[b for _, b in EFFECTORS]]
        self.bones = [(model.body(a).id,model.body(b).id)
                      for _,_,_,a,b in source.bone_correspondences()]
        self.pose_weight,self.pose_gain = pose_weight,pose_gain
        self.intermediate_weight,self.shoulder_weight = intermediate_weight,shoulder_weight
        self.intermediate_bodies = [model.body(b).id for _,b in INTERMEDIATES]
        self.shoulder_bodies = [model.body(b).id for _,b in SHOULDERS]
        self.hip_weight = hip_weight
        self.hip_bodies = [model.body(b).id for _,b in HIPS]
        self.pelvis_body=model.body('pelvis_link').id
        self.thigh_bodies=[model.body(b).id for b in HIP_ROTATION_BODIES]
        self.hip_rotation_weight,self.pelvis_rotation_weight=hip_rotation_weight,pelvis_rotation_weight
        self.height_weight,self.contact_weight,self.slip_weight=height_weight,contact_weight,slip_weight
        by_body={model.geom_bodyid[g]:(g,vertices) for g,vertices in self.meshes}
        self.contact_meshes=[by_body[model.body(b).id] for _,b,_ in CONTACTS]
        self.speed = np.r_[[4.]*3, [12.]*3,
            [profile['velocity_limits_joint_coordinate_rad_s'][model.joint(j).name]
             for j in range(1, model.njnt)]]
        self.rejected_steps = self.qp_failures = self.backtracked_steps = 0
        self.minimum_step_fraction = 1.

    def set_pose(self, q):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self._world_meshes=None;self._collision_entries=None

    def world_meshes(self):
        if self._world_meshes is None:
            if self.collision_index is not None:
                self.collision_index.update(self.data)
                self._world_meshes=list(self.collision_index.near_floor(self.data,.03))
            else:
                self._world_meshes=[(g,v@self.data.geom_xmat[g].reshape(3,3).T+self.data.geom_xpos[g])
                                    for g,v in self.meshes]
        return self._world_meshes

    def collision_entries(self):
        if self._collision_entries is None:
            self.world_meshes()
            if self.collision_index is not None:
                self._collision_entries=list(self.collision_index.query(self.data,.03))
            else:
                self._collision_entries=[]
                for a,b in self.pairs:
                    value,witness,certificate=kin.checked_mesh_distance(self.model,self.data,a,b,self.vertices)
                    self._collision_entries.append((a,b,value,witness,certificate))
        return self._collision_entries

    def collision_gradient(self,a,b,distance,witness):
        certified=(self.collision_index.separating_normals.get((a,b))
                   if self.collision_index is not None else None)
        if certified is not None:normal=certified
        else:
            normal=witness[3:]-witness[:3];length=np.linalg.norm(normal)
            if length<1e-12:return None
            normal*=np.sign(distance if distance else 1.)/length
        ja,jb=np.empty((3,self.model.nv)),np.empty((3,self.model.nv))
        mujoco.mj_jac(self.model,self.data,ja,None,witness[:3],self.model.geom_bodyid[a])
        mujoco.mj_jac(self.model,self.data,jb,None,witness[3:],self.model.geom_bodyid[b])
        return normal@(jb-ja)

    def joint_limit_terms(self,q):
        low=self.model.jnt_range[1:,0]+self.joint_margin
        high=self.model.jnt_range[1:,1]-self.joint_margin
        violation=np.where(q[7:]<low,low-q[7:],np.where(q[7:]>high,high-q[7:],0.))
        active=np.flatnonzero(violation)
        jac=np.eye(self.model.nv)[active+6]
        return jac,4.*violation[active]

    def clear(self):
        d = self.data
        for _,world in self.world_meshes():
            if world[:,2].min() < .00005:
                return False
        for a,b,gap,_,_ in self.collision_entries():
            if gap < .00005:
                return False
        return True

    def bone_features(self):
        directions,jacobians = [],[]
        for first,second in self.bones:
            vector = self.data.xpos[second]-self.data.xpos[first]
            length = np.linalg.norm(vector); direction = vector/length
            ja,jb = np.empty((3,self.model.nv)),np.empty((3,self.model.nv))
            mujoco.mj_jacBody(self.model,self.data,ja,None,first)
            mujoco.mj_jacBody(self.model,self.data,jb,None,second)
            directions.append(direction)
            jacobians.append((np.eye(3)-np.outer(direction,direction)) @ (jb-ja)/length)
        return np.asarray(directions),np.asarray(jacobians)

    def initialize_height_contact(self,ref):
        """One-time source-aligned initial condition; no per-frame pose reset."""
        q=self.initial.copy();self.set_pose(q)
        trunk=self.model.body('torso_roll_link').id
        root=R.from_quat(q[3:7],scalar_first=True).as_matrix()
        relative=root.T@self.data.xmat[trunk].reshape(3,3)
        q[3:7]=R.from_matrix(ref['body_rotation'][0]@relative.T).as_quat(scalar_first=True)
        self.set_pose(q)
        q[2]+=ref['trunk_height_target'][0]-self.data.xpos[trunk,2]
        self.set_pose(q)
        bottom=min(float((vertices@self.data.geom_xmat[g].reshape(3,3)[2]+self.data.geom_xpos[g,2]).min())
                   for g,vertices in self.meshes)
        q[2]+=max(0.,.0007-bottom)
        self.set_pose(q)
        if not self.clear():raise ValueError('Source-aligned initial seed is not collision-free')
        return q

    def contact_features(self):
        heights,jacobians=[],[]
        for gid,vertices in self.contact_meshes:
            world=vertices@self.data.geom_xmat[gid].reshape(3,3).T+self.data.geom_xpos[gid]
            point=world[int(np.argmin(world[:,2]))]
            jac=np.empty((3,self.model.nv))
            mujoco.mj_jac(self.model,self.data,jac,None,point,self.model.geom_bodyid[gid])
            heights.append(point[2]);jacobians.append(jac)
        return np.array(heights),np.array(jacobians)

    def hip_rotation_features(self):
        """Relative SO(3) frames and angular Jacobians in pelvis coordinates."""
        pelvis=self.data.xmat[self.pelvis_body].reshape(3,3).copy()
        jp=np.empty((3,self.model.nv));mujoco.mj_jacBody(self.model,self.data,None,jp,self.pelvis_body)
        relative,jacobians=[],[]
        for bid in self.thigh_bodies:
            thigh=self.data.xmat[bid].reshape(3,3)
            jt=np.empty((3,self.model.nv));mujoco.mj_jacBody(self.model,self.data,None,jt,bid)
            relative.append(pelvis.T@thigh)
            jacobians.append(pelvis.T@(jt-jp))
        return pelvis,jp,np.asarray(relative),np.asarray(jacobians)

    def step(self, q, target_velocity, target_angular, target_gravity, target_bones=None,
             target_intermediates=None, target_shoulders=None, target_hips=None,
             target_height=None,target_height_rate=0.,contact_confidence=None,
             target_hip_rotation=None,target_hip_angular=None,
             target_pelvis_rotation=None,target_pelvis_angular=None):
        m, d, dt = self.model, self.data, self.dt
        self.set_pose(q)
        matrices, targets = [], []
        if self.hip_rotation_weight>0 or self.pelvis_rotation_weight>0:
            pelvis,jp,relative,jacobians=self.hip_rotation_features()
            if self.hip_rotation_weight>0:
                if target_hip_rotation is None or target_hip_angular is None:
                    raise ValueError('Hip rotation tracking requires relative orientation and angular velocity')
                error=R.from_matrix(target_hip_rotation@np.swapaxes(relative,-1,-2)).as_rotvec()
                matrices.append(self.hip_rotation_weight*jacobians.reshape(-1,m.nv))
                targets.append(self.hip_rotation_weight*(target_hip_angular+4.*error).ravel())
            if self.pelvis_rotation_weight>0:
                if target_pelvis_rotation is None or target_pelvis_angular is None:
                    raise ValueError('Pelvis rotation tracking requires orientation and angular velocity')
                error=R.from_matrix(target_pelvis_rotation@pelvis.T).as_rotvec()
                matrices.append(self.pelvis_rotation_weight*jp)
                targets.append(self.pelvis_rotation_weight*(target_pelvis_angular+4.*error))
        for i, bid in enumerate(self.bodies):
            jp, jw = np.empty((3, m.nv)), np.empty((3, m.nv))
            mujoco.mj_jacBody(m, d, jp, jw, bid)
            weight = 20. if i == 0 else 12.
            matrices.append(weight*jp); targets.append(weight*target_velocity[i])
            if i == 0:
                actual = d.xmat[bid].reshape(3, 3)
                gravity = actual.T @ np.array([0., 0., -1.])
                matrices.extend([6.*jw, 20.*kin.skew(gravity) @ actual.T @ jw])
                targets.extend([6.*target_angular, 20.*(target_gravity-gravity)/dt])
                if self.height_weight>0:
                    if target_height is None:raise ValueError('Height weight requires a height target')
                    matrices.append(self.height_weight*jp[2:3])
                    targets.append(np.array([self.height_weight*(target_height_rate+4.*(target_height-d.xpos[bid,2]))]))
        if self.contact_weight>0 or self.slip_weight>0:
            if contact_confidence is None:raise ValueError('Contact weights require inferred contact cues')
            heights,jacobians=self.contact_features()
            for confidence,height,jac in zip(contact_confidence,heights,jacobians):
                if confidence<1e-4:continue
                weight=self.contact_weight*np.sqrt(confidence)
                if weight>0:
                    matrices.append(weight*jac[2:3])
                    targets.append(np.array([weight*8.*(.0007-height)]))
                # Tangential sticking is activated near actual ground; no XY
                # source position or fixed world-space footprint is prescribed.
                weight=self.slip_weight*np.sqrt(confidence*np.clip((.03-height)/.025,0.,1.))
                if weight>0:
                    matrices.append(weight*jac[:2]);targets.append(np.zeros(2))
        for weight,bodies,target in [(self.intermediate_weight,self.intermediate_bodies,target_intermediates),
                                      (self.shoulder_weight,self.shoulder_bodies,target_shoulders),
                                      (self.hip_weight,self.hip_bodies,target_hips)]:
            if weight<=0:
                continue
            if target is None:
                raise ValueError('Positive landmark velocity weight requires matching target velocities')
            for i,bid in enumerate(bodies):
                jp = np.empty((3,m.nv)); mujoco.mj_jacBody(m,d,jp,None,bid)
                matrices.append(weight*jp); targets.append(weight*target[i])
        if self.pose_weight > 0:
            if target_bones is None:
                raise ValueError('A positive pose weight requires anatomical direction targets')
            directions,jacobians = self.bone_features()
            matrices.append(self.pose_weight*jacobians.reshape(-1,m.nv))
            targets.append(self.pose_weight*self.pose_gain*(target_bones-directions).ravel())
        if self.joint_limit_weight>0:
            jac,target=self.joint_limit_terms(q)
            if len(target):matrices.append(self.joint_limit_weight*jac);targets.append(self.joint_limit_weight*target)
        if self.self_collision_weight>0:
            for a,b,distance,witness,_ in self.collision_entries():
                if distance>=self.self_clearance:continue
                gradient=self.collision_gradient(a,b,distance,witness)
                if gradient is None:continue
                matrices.append(self.self_collision_weight*gradient[None])
                targets.append(np.array([self.self_collision_weight*8.*(self.self_clearance-distance)]))
        a, b = np.vstack(matrices), np.concatenate(targets)
        # Numerical minimum-velocity regularization; no configuration target.
        p, c = a.T @ a + 1e-4*np.eye(m.nv), -a.T @ b
        lo, hi = -self.speed.copy(), self.speed.copy()
        lo[6:] = np.maximum(lo[6:], (m.jnt_range[1:, 0]+1e-5-q[7:])/dt)
        hi[6:] = np.minimum(hi[6:], (m.jnt_range[1:, 1]-1e-5-q[7:])/dt)
        rows, bounds = [-np.eye(m.nv), np.eye(m.nv)], [-lo, hi]
        for gid,world in self.world_meshes():
            near = world[world[:, 2] < .03]
            if not len(near):
                continue
            jp, jr = np.empty((3, m.nv)), np.empty((3, m.nv))
            bid = m.geom_bodyid[gid]; mujoco.mj_jacBody(m, d, jp, jr, bid)
            radius = near-d.xpos[bid]
            dz = jp[2] + radius[:, 1, None]*jr[0] - radius[:, 0, None]*jr[1]
            rows.append(-dt*dz); bounds.append(.85*(near[:, 2]-.0003))
        for first,second,gap,witness,_ in self.collision_entries():
            if gap >= .025:
                continue
            gradient=self.collision_gradient(first,second,gap,witness)
            if gradient is None:continue
            rows.append((-dt*gradient)[None]); bounds.append(np.array([.85*(gap-.001)]))
        g, h = np.vstack(rows), np.concatenate(bounds)
        v = self.solve_qp(p, c, g, h, solver='daqp')
        if v is None or not np.isfinite(v).all() or np.max(g @ v-h) > 2e-5:
            self.qp_failures += 1
            return q.copy()
        for backtrack in range(11):
            fraction = 2.**(-backtrack)
            candidate = q.copy(); mujoco.mj_integratePos(m, candidate, v, dt*fraction)
            self.set_pose(candidate)
            if self.clear():
                self.backtracked_steps += int(backtrack > 0)
                self.minimum_step_fraction = min(self.minimum_step_fraction, fraction)
                return candidate
        self.rejected_steps += 1
        self.set_pose(q)
        return q.copy()


def height_contact_measurements(model,q,qvel,ref):
    data=mujoco.MjData(model);meshes,_=source.geometry(model)
    by_body={model.geom_bodyid[g]:(g,vertices) for g,vertices in meshes}
    contact_meshes=[by_body[model.body(b).id] for _,b,_ in CONTACTS]
    trunk=model.body('torso_roll_link').id
    trunk_height,heights,speeds,lowest=[],[],[],[]
    for pose,velocity in zip(q,qvel):
        data.qpos[:]=pose;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        trunk_height.append(data.xpos[trunk,2])
        lowest.append(min(float((vertices@data.geom_xmat[g].reshape(3,3)[2]+data.geom_xpos[g,2]).min())
                          for g,vertices in meshes))
        local_h,local_v=[],[]
        for gid,vertices in contact_meshes:
            world=vertices@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
            point=world[int(np.argmin(world[:,2]))]
            jac=np.empty((3,model.nv));mujoco.mj_jac(model,data,jac,None,point,model.geom_bodyid[gid])
            local_h.append(point[2]);local_v.append(jac@velocity)
        heights.append(local_h);speeds.append(local_v)
    trunk_height,heights,speeds,lowest=map(np.asarray,[trunk_height,heights,speeds,lowest])
    error=trunk_height-ref['trunk_height_target'];active=ref['contact_confidence']>.5
    near=active&(heights<.03);source_contact=active.any(axis=1)
    summary=dict(trunk_height_error_rmse_m=float(np.sqrt(np.mean(error**2))),
        trunk_height_error_p95_m=float(np.quantile(abs(error),.95)),
        inferred_contact_point_frames=int(active.sum()),inferred_contact_frames=int(source_contact.sum()),
        inferred_contact_height_error_p95_m=float(np.quantile(abs(heights[active]-.0007),.95)) if active.any() else None,
        near_ground_contact_material_speed_p95_m_s=float(np.quantile(np.linalg.norm(speeds[:,:,:2],axis=-1)[near],.95)) if near.any() else None,
        hover_during_inferred_contact_fraction=float(np.mean(lowest[source_contact]>.05)) if source_contact.any() else None,
        lowest_hull_height_during_inferred_contact_median_m=float(np.median(lowest[source_contact])) if source_contact.any() else None,
        contact_labels_are_inferred=True)
    return summary,dict(actual_trunk_height=trunk_height,actual_contact_heights=heights,
                         actual_contact_material_velocity=speeds,actual_lowest_hull_height=lowest)


def rotation_measurements(model,q,ref):
    data=mujoco.MjData(model);pelvis=model.body('pelvis_link').id
    thighs=[model.body(b).id for b in HIP_ROTATION_BODIES]
    actual_pelvis,actual_relative=[],[]
    for pose in q:
        data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
        rp=data.xmat[pelvis].reshape(3,3).copy()
        actual_pelvis.append(rp)
        actual_relative.append(rp.T@data.xmat[thighs].reshape(2,3,3))
    actual_pelvis,actual_relative=np.asarray(actual_pelvis),np.asarray(actual_relative)
    error=R.from_matrix((np.swapaxes(ref['hip_relative_rotation'],-1,-2)@actual_relative).reshape(-1,3,3)).magnitude().reshape(len(q),2)*180/np.pi
    pe=R.from_matrix(np.swapaxes(ref['pelvis_rotation'],-1,-2)@actual_pelvis).magnitude()*180/np.pi
    summary=dict(hip_relative_rotation_error_deg=dict(median=float(np.median(error)),p95=float(np.quantile(error,.95)),
        left_median=float(np.median(error[:,0])),right_median=float(np.median(error[:,1]))),
        pelvis_rotation_error_deg=dict(median=float(np.median(pe)),p95=float(np.quantile(pe,.95))))
    return summary,dict(actual_hip_relative_rotation=actual_relative,actual_pelvis_rotation=actual_pelvis,
                         hip_relative_rotation_error_deg=error,pelvis_rotation_error_deg=pe)


def measure(model, q, ref, fps,collision_pairs=None):
    data = mujoco.MjData(model)
    bodies = [model.body(n).id for n in ref['tracked_bodies']]
    positions, rotation, bone_directions, intermediate_positions, shoulder_positions = [], [], [], [], []
    intermediate_bodies = [model.body(b).id for _,b in INTERMEDIATES]
    shoulder_bodies = [model.body(b).id for _,b in SHOULDERS]
    hip_bodies = [model.body(b).id for _,b in HIPS]
    hip_positions = []
    bones = [(model.body(a).id,model.body(b).id) for _,_,_,a,b in source.bone_correspondences()]
    for pose in q:
        data.qpos[:] = pose; mujoco.mj_kinematics(model, data)
        positions.append(data.xpos[bodies].copy())
        rotation.append(data.xmat[bodies[0]].reshape(3, 3).copy())
        intermediate_positions.append(data.xpos[intermediate_bodies].copy())
        shoulder_positions.append(data.xpos[shoulder_bodies].copy())
        hip_positions.append(data.xpos[hip_bodies].copy())
        bone = np.array([data.xpos[b]-data.xpos[a] for a,b in bones])
        bone_directions.append(bone/np.linalg.norm(bone,axis=-1,keepdims=True))
    positions, rotation = np.asarray(positions), np.asarray(rotation)
    actual_velocity = np.gradient(positions, 1/fps, axis=0, edge_order=2)
    error = np.linalg.norm(actual_velocity-ref['linear_velocity'], axis=-1)
    intermediate_velocity = np.gradient(intermediate_positions,1/fps,axis=0,edge_order=2)
    shoulder_velocity = np.gradient(shoulder_positions,1/fps,axis=0,edge_order=2)
    hip_velocity = np.gradient(hip_positions,1/fps,axis=0,edge_order=2)
    intermediate_error = np.linalg.norm(intermediate_velocity-ref['intermediate_linear_velocity'],axis=-1)
    shoulder_error = np.linalg.norm(shoulder_velocity-ref['shoulder_linear_velocity'],axis=-1)
    hip_error = np.linalg.norm(hip_velocity-ref['hip_linear_velocity'],axis=-1)
    angular = np.empty((len(q), 3))
    for i in range(len(q)):
        a, b = max(0, i-1), min(len(q)-1, i+1)
        angular[i] = R.from_matrix(rotation[b] @ rotation[a].T).as_rotvec()/((b-a)/fps)
    gravity = np.einsum('tji,j->ti', rotation, [0., 0., -1.])
    tilt = np.rad2deg(np.arccos(np.clip(np.sum(gravity*ref['gravity'], axis=-1), -1., 1.)))
    bone_error = np.rad2deg(np.arccos(np.clip(np.sum(
        np.asarray(bone_directions)*ref['bone_directions'],axis=-1),-1.,1.)))
    qvel = source.differentiate(model, q, fps)
    profile = json.loads((source.OUTPUT/'light_profile.json').read_text())
    limits = np.array([profile['velocity_limits_joint_coordinate_rad_s'][model.joint(j).name]
                       for j in range(1, model.njnt)])
    height_contact,contact_series=height_contact_measurements(model,q,qvel,ref)
    rotation_summary,rotation_series=rotation_measurements(model,q,ref)
    summary = dict(velocity_rmse_m_s={name:float(np.sqrt(np.mean(error[:, i]**2)))
                                   for i, name in enumerate(ref['tracked_bodies'])},
        trunk_tilt_error_deg=dict(median=float(np.median(tilt)), p95=float(np.quantile(tilt, .95))),
        intermediate_velocity_rmse_m_s={name:float(np.sqrt(np.mean(intermediate_error[:,i]**2)))
                                      for i,(_,name) in enumerate(INTERMEDIATES)},
        shoulder_velocity_rmse_m_s={name:float(np.sqrt(np.mean(shoulder_error[:,i]**2)))
                                  for i,(_,name) in enumerate(SHOULDERS)},
        hip_velocity_rmse_m_s={name:float(np.sqrt(np.mean(hip_error[:,i]**2)))
                             for i,(_,name) in enumerate(HIPS)},
        bone_direction_error_deg=dict(median=float(np.median(bone_error)),
            p95=float(np.quantile(bone_error,.95)),
            per_bone_median={name:float(np.median(bone_error[:,i]))
                            for i,(name,*_) in enumerate(source.bone_correspondences())}),
        trunk_angular_velocity_rmse_rad_s=float(np.sqrt(np.mean(np.sum((angular-ref['angular_velocity'])**2, axis=-1)))),
        joint_speed_to_limit_max=float(np.max(abs(qvel[:, 6:])/limits)),
        joint_limit_excess_rad=float(max(0., (model.jnt_range[1:, 0]-q[:, 7:]).max(),
                                         (q[:, 7:]-model.jnt_range[1:, 1]).max())),
        height_contact=height_contact,
        subframes=geometry_checks.subframe_check(model, q,pairs=collision_pairs),**rotation_summary)
    return summary, dict(actual_linear_velocity=actual_velocity, actual_angular_velocity=angular,
                         actual_gravity=gravity, actual_body_positions=positions,
                         actual_intermediate_linear_velocity=intermediate_velocity,
                         actual_shoulder_linear_velocity=shoulder_velocity,
                         actual_hip_linear_velocity=hip_velocity,
                         actual_bone_directions=np.asarray(bone_directions),
                         bone_direction_error_deg=bone_error,trunk_tilt_error_deg=tilt, qvel=qvel,**contact_series,**rotation_series)


def render(model, q, baseline, ref, fps, path,
           baseline_title='Existing pose / position IK',candidate_title='Velocity + trunk tilt; no pose prior'):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    chains = [['Hips','LeftUpLeg','LeftLeg','LeftFoot','LeftToe'],
              ['Hips','RightUpLeg','RightLeg','RightFoot','RightToe'],
              ['Hips','Spine','Spine1','Spine2','Neck','Head'],
              ['Spine2','LeftShoulder','LeftArm','LeftForeArm','LeftHand'],
              ['Spine2','RightShoulder','RightArm','RightForeArm','RightHand']]
    edges = [(ref['names'].index(a),ref['names'].index(b)) for chain in chains for a,b in zip(chain,chain[1:])]
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    option, camera, data = mujoco.MjvOption(), mujoco.MjvCamera(), mujoco.MjData(model)
    human = ref['human'].copy()
    foot_ids = [ref['names'].index(n) for n in ['LeftToe','RightToe']]
    human[:,:,2] -= np.quantile(human[:,foot_ids,2], .05)
    titles = ['Original LAFAN human', baseline_title, candidate_title]
    with mujoco.Renderer(model, height=440, width=560) as renderer, imageio.get_writer(
            path, fps=25, codec='libx264', quality=7, macro_block_size=1) as writer:
        for i in range(0,len(q),max(1,round(fps/25))):
            canvas = Image.new('RGB',(1680,512),'#e9eef2'); draw = ImageDraw.Draw(canvas)
            for column in range(3):
                data.qpos[:] = baseline[i] if column == 1 else q[i]
                mujoco.mj_forward(model,data)
                option.geomgroup[:] = 0; option.geomgroup[0] = 1
                if column: option.geomgroup[2] = 1
                camera.lookat[:] = .5*(q[i,:3]+human[i,ref['names'].index('Hips')])
                camera.lookat[2] = .7
                camera.distance, camera.azimuth, camera.elevation = 3., 10., -12.
                renderer.update_scene(data,camera=camera,scene_option=option)
                if column == 0:
                    for a,b in edges:
                        scene = renderer.scene; g = scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),
                            np.eye(3).ravel(),np.array([1.,.55,.12,1.],np.float32))
                        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.012,human[i,a],human[i,b])
                        scene.ngeom += 1
                canvas.paste(Image.fromarray(renderer.render()),(560*column,35))
                draw.text((560*column+8,8),titles[column],font=font,fill='#17242d')
            draw.text((8,480),f'{path.stem} | {i/fps:.2f}s | Kinematic replay; no torque / dynamic certificate',font=font,fill='#9b2929')
            writer.append_data(np.asarray(canvas))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip',action='append',help='BVH stem:start:duration')
    parser.add_argument('--fps',type=float,default=50.)
    parser.add_argument('--substeps',type=int,default=4)
    parser.add_argument('--pose-weight',type=float,default=0.)
    parser.add_argument('--pose-gain',type=float,default=2.,help='Bone-direction proportional gain, per second')
    parser.add_argument('--intermediate-weight',type=float,default=0.,help='Elbow/knee linear-velocity residual weight')
    parser.add_argument('--shoulder-weight',type=float,default=0.,help='Shoulder landmark linear-velocity residual weight')
    parser.add_argument('--hip-weight',type=float,default=0.,help='Hip landmark linear-velocity residual weight')
    parser.add_argument('--height-weight',type=float,default=0.,help='Trunk height servo residual weight')
    parser.add_argument('--contact-weight',type=float,default=0.,help='Inferred support surface height residual weight')
    parser.add_argument('--slip-weight',type=float,default=0.,help='Near-ground tangential velocity residual weight')
    parser.add_argument('--hip-rotation-weight',type=float,default=0.,help='Pelvis-relative thigh orientation/angular velocity weight')
    parser.add_argument('--pelvis-rotation-weight',type=float,default=0.,help='World pelvis orientation/angular velocity weight')
    parser.add_argument('--joint-limit-weight',type=float,default=0.,help='Soft avoidance inside joint-limit safety margins')
    parser.add_argument('--self-collision-weight',type=float,default=0.,help='Soft mesh clearance avoidance residual')
    parser.add_argument('--collision-coverage',choices=['selected','full'],default='selected')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--comparison-dir',type=Path,
                        help='Default: existing IK for weight=0, previous velocity-only output otherwise')
    parser.add_argument('--no-render',action='store_true')
    parser.add_argument('--skip-comparison',action='store_true',help='Do not load an older comparison trajectory')
    args = parser.parse_args()
    if args.fps <= 0 or args.substeps < 1: parser.error('fps and substeps must be positive')
    if (not np.isfinite([args.pose_weight,args.pose_gain,args.fps,args.intermediate_weight,args.shoulder_weight,args.hip_weight,
                         args.height_weight,args.contact_weight,args.slip_weight,args.hip_rotation_weight,args.pelvis_rotation_weight,
                         args.joint_limit_weight,args.self_collision_weight]).all()
            or min(args.pose_weight,args.intermediate_weight,args.shoulder_weight,args.hip_weight,
                   args.height_weight,args.contact_weight,args.slip_weight,args.hip_rotation_weight,args.pelvis_rotation_weight,
                   args.joint_limit_weight,args.self_collision_weight)<0 or args.pose_gain<=0):
        parser.error('Weights must be finite/nonnegative and gain/fps finite/positive')
    output = args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    augmented = max(args.pose_weight,args.intermediate_weight,args.shoulder_weight,args.hip_weight,
                    args.height_weight,args.contact_weight,args.slip_weight,args.hip_rotation_weight,args.pelvis_rotation_weight,
                    args.joint_limit_weight,args.self_collision_weight)>0
    rotation_enabled=max(args.hip_rotation_weight,args.pelvis_rotation_weight)>0
    height_contact_enabled=max(args.height_weight,args.contact_weight,args.slip_weight)>0
    comparison = args.comparison_dir or (OUT if augmented else source.OUTPUT)
    if augmented and output==OUT.resolve():
        parser.error('Use a separate --output-dir to preserve the velocity-only baseline')
    _, extract, utils = source.prepare()
    model = mujoco.MjModel.from_xml_path(str(ROOT/'outputs/humanoid_light_conversion/scene.xml'))
    profile = json.loads((source.OUTPUT/'light_profile.json').read_text())
    reports = {}
    for spec in args.clip or ['walk1_subject1:5:5','dance1_subject2:5:5','fight1_subject2:5:5']:
        name,start,duration = spec.split(':'); start,duration = float(start),float(duration)
        if duration*args.fps < 2: raise ValueError('Need at least three frames')
        path = ROOT/'src/humanoid_retarget/datasets/lafan1/raw'/f'{name}.bvh'
        ref = references(path,start,duration,args.fps,model,extract,utils)
        solver = VelocityIK(model,profile,1/(args.fps*args.substeps),args.pose_weight,args.pose_gain,
                            args.intermediate_weight,args.shoulder_weight,args.hip_weight,
                            args.height_weight,args.contact_weight,args.slip_weight,
                            args.hip_rotation_weight,args.pelvis_rotation_weight,args.joint_limit_weight,
                            args.self_collision_weight,args.collision_coverage)
        initial=solver.initialize_height_contact(ref) if height_contact_enabled else solver.initial.copy()
        q = [initial]; dense=[initial.copy()]; began = time.perf_counter()
        for frame in range(len(ref['times'])-1):
            pose = q[-1]
            for sub in range(args.substeps):
                pose = solver.step(pose,**target_at_step(ref,frame,sub,args.substeps,rotation_enabled))
                dense.append(pose.copy())
            q.append(pose.copy())
        q = np.asarray(q); label = f'{name}_{start:g}s_{duration:g}s'
        report, actual = measure(model,q,ref,args.fps,solver.pairs if args.collision_coverage=='full' else None)
        report.update(source=str(path.relative_to(ROOT)),source_sha256=source.digest(path),
            source_start_seconds=start,fps=args.fps,frames=len(q),scale=ref['scale'],
            source_fps=ref['source_fps'],wall_seconds=time.perf_counter()-began,
            objective_terms=['trunk projected gravity','trunk linear velocity','trunk angular velocity','four end-effector linear velocities'],
            no_joint_pose_prior=True,no_bone_direction_prior=args.pose_weight==0,
            no_cartesian_position_prior=max(args.height_weight,args.contact_weight)==0,
            no_cartesian_xy_position_prior=True,
            pose_weight=args.pose_weight,pose_gain_per_second=args.pose_gain,
            intermediate_velocity_weight=args.intermediate_weight,shoulder_velocity_weight=args.shoulder_weight,
            hip_velocity_weight=args.hip_weight,
            trunk_height_weight=args.height_weight,contact_height_weight=args.contact_weight,contact_slip_weight=args.slip_weight,
            hip_rotation_weight=args.hip_rotation_weight,pelvis_rotation_weight=args.pelvis_rotation_weight,
            joint_limit_weight=args.joint_limit_weight,self_collision_weight=args.self_collision_weight,
            joint_limit_margin_rad=solver.joint_margin.tolist(),self_clearance_target_m=solver.self_clearance,
            collision_coverage=args.collision_coverage,collision_pairs=len(solver.pairs),
            mesh_distance_method=('legacy libccd with original-vertex separation certificates'
                                  if args.collision_coverage=='full' else 'existing native CCD with checked zero repairs'),
            structural_collision_exclusions=solver.structural_exclusions,
            hip_rotation_target='anatomically calibrated thigh orientation relative to pelvis; native bounded robot axes',
            no_contact_schedule_objective=max(args.contact_weight,args.slip_weight)==0,
            contact_method='soft source height/velocity cues; lowest robot hull point; no fixed XY footprint',
            initial_condition=('source-aligned trunk tilt and calibrated height, neutral joints, floor-clear seed'
                               if height_contact_enabled else 'neutral robot pose; source limb pose is not fitted'),
            velocity_damping=1e-4,substeps=args.substeps,qp_failures=solver.qp_failures,
            rejected_steps=solver.rejected_steps,backtracked_steps=solver.backtracked_steps,
            minimum_step_fraction=solver.minimum_step_fraction,training_ready=False,
            dynamic_feasibility_verified=False,optimizer_sha256=source.digest(Path(__file__)),
            robot_sha256=source.digest(ROOT/'outputs/humanoid_light_conversion/humanoid_light.xml'))
        if args.pose_weight>0:
            report['objective_terms'].append('weak eight anatomical bone directions')
        if args.intermediate_weight>0:
            report['objective_terms'].append('four elbow/knee landmark linear velocities')
        if args.shoulder_weight>0:
            report['objective_terms'].append('two shoulder landmark linear velocities')
        if args.hip_weight>0:
            report['objective_terms'].append('two hip landmark linear velocities')
        if args.height_weight>0:report['objective_terms'].append('calibrated trunk height')
        if args.contact_weight>0:report['objective_terms'].append('inferred support surface height')
        if args.slip_weight>0:report['objective_terms'].append('near-ground support material tangential velocity')
        if args.hip_rotation_weight>0:report['objective_terms'].append('pelvis-relative thigh SO(3) orientation and angular velocity')
        if args.pelvis_rotation_weight>0:report['objective_terms'].append('pelvis SO(3) orientation and angular velocity')
        if args.joint_limit_weight>0:report['objective_terms'].append('joint-limit margin avoidance')
        if args.self_collision_weight>0:report['objective_terms'].append('body mesh clearance avoidance')
        report['checks'] = dict(original_joint_ranges=report['joint_limit_excess_rad']<1e-6,
            original_joint_speeds=report['joint_speed_to_limit_max']<1.001,
            interpolated_floor_clear=report['subframes']['minimum_floor_height_m']>-.001,
            interpolated_selected_self_clear=report['subframes']['minimum_selected_self_gap_m']>.00005,
            interpolated_distance_queries_resolved=report['subframes']['unresolved_zero_distance_queries']==0)
        report['status'] = ('kinematic_preview_needs_review' if all(report['checks'].values())
                            else 'kinematic_preview_needs_correction')
        baseline_path = comparison/f'{label}.npz'
        baseline = None
        if not args.skip_comparison and baseline_path.exists():
            with np.load(baseline_path,allow_pickle=False) as old:
                if np.isclose(float(old['frequency']),args.fps) and len(old['qpos'])==len(q):
                    baseline = old['qpos'].astype(float)
            if baseline is not None:
                # Existing IK already uses the same initial pelvis heading
                # and source-origin alignment; applying it twice is wrong.
                report['baseline'], _ = measure(model,baseline,ref,args.fps)
                report['baseline_path']=str(baseline_path.resolve())
                report['baseline_sha256']=source.digest(baseline_path)
        np.savez_compressed(output/f'{label}.npz',qpos=q,frequency=np.array(args.fps),
            qpos_substeps=np.asarray(dense),substep_frequency=np.array(args.fps*args.substeps),
            joint_names=np.array(['root']+[model.joint(j).name for j in range(1,model.njnt)]),
            split_points=np.array([0,len(q)]),training_ready=np.array(False),**actual)
        np.savez_compressed(output/f'{label}_targets.npz',times=ref['times'],source_times=ref['source_times'],
            human_positions=ref['human'],human_joint_names=np.array(ref['names']),
            target_linear_velocity=ref['linear_velocity'],target_angular_velocity=ref['angular_velocity'],
            target_gravity=ref['gravity'],target_bone_directions=ref['bone_directions'],
            target_intermediate_linear_velocity=ref['intermediate_linear_velocity'],
            target_shoulder_linear_velocity=ref['shoulder_linear_velocity'],
            target_hip_linear_velocity=ref['hip_linear_velocity'],
            target_hip_relative_rotation=ref['hip_relative_rotation'],
            target_hip_relative_angular_velocity=ref['hip_relative_angular_velocity'],
            target_pelvis_rotation=ref['pelvis_rotation'],target_pelvis_angular_velocity=ref['pelvis_angular_velocity'],
            trunk_height_target=ref['trunk_height_target'],trunk_height_rate=ref['trunk_height_rate'],
            contact_confidence=ref['contact_confidence'],contact_body_names=np.array(ref['contact_body_names']),
            tracked_bodies=np.array(ref['tracked_bodies']))
        (output/f'{label}.json').write_text(json.dumps(report,indent=2)+'\n')
        reports[label] = report; print(label,json.dumps(report),flush=True)
        if not args.no_render and baseline is not None:
            titles = (dict(baseline_title='Velocity + tilt; no pose prior',
                           candidate_title=f'Velocity + tilt + weak pose (w={args.pose_weight:g})')
                      if args.pose_weight>0 else {})
            if args.comparison_dir:
                titles['baseline_title']='Previous candidate'
            if args.intermediate_weight>0 or args.shoulder_weight>0 or args.hip_weight>0 or height_contact_enabled or rotation_enabled:
                terms=['Velocity + tilt']+(['pose'] if args.pose_weight>0 else [])
                if args.intermediate_weight>0:terms.append('elbow/knee')
                if args.shoulder_weight>0:terms.append('shoulders')
                if args.hip_weight>0:terms.append('hips')
                if height_contact_enabled:terms.append('height/contact')
                if rotation_enabled:terms.append('hip rotation')
                titles['candidate_title']=' + '.join(terms)
            render(model,q,baseline,ref,args.fps,output/f'{label}.mp4',**titles)
    (output/'samples.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__ == '__main__':
    main()
