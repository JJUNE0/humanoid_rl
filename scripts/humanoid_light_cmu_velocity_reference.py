"""CMU ASF/AMC -> the same feature objectives used by the full LAFAN run.

Preserves source timing and complete rotations. Position/footprint targets
from the older CMU experiment are not used by the velocity optimizer.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R,Slerp

import retarget_humanoid_light_cmu as cmu
import retarget_humanoid_light_lafan_velocity as velocity


def neutral_points(motion):
    """Reference ASF skeleton with zero motion; never use a tucked first pose."""
    positions={}
    for bone in motion.bones:
        positions[bone.name]=(bone.direction*motion.meter_scale if bone.parent is None
            else positions[bone.parent]+bone.direction*bone.length*motion.meter_scale)
    yup_to_zup=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
    return {name:value@yup_to_zup.T for name,value in positions.items()}


def references(motion,model,fps=50.):
    _,rotations,_,_,airborne,human,names,times,scale=cmu.targets(motion,model,fps)
    human=human*scale
    pelvis=rotations['pelvis_link']
    # A pelvis-to-neck line includes spinal bending and is not the chest's
    # actual orientation. Match LAFAN's tracked thorax rotation semantics.
    neutral_robot=mujoco.MjData(model);mujoco.mj_forward(model,neutral_robot)
    anatomical=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    first_pelvis=motion.rotations[0,0]@anatomical
    align=R.from_euler('z',-np.arctan2(first_pelvis[1,0],first_pelvis[0,0])).as_matrix()
    original_times=np.arange(len(motion.frame_ids))/motion.fps
    chest=Slerp(original_times,R.from_matrix(motion.rotations[:,motion.names.index('thorax')]))(times).as_matrix()
    trunk=align@chest@anatomical@neutral_robot.xmat[model.body('torso_roll_link').id].reshape(3,3)
    thighs=np.stack([rotations[name] for name in velocity.HIP_ROTATION_BODIES],axis=1)
    relative=np.swapaxes(pelvis,-1,-2)[:,None]@thighs
    def landmark_velocity(points):
        selected=human[:,[names.index(name) for name in points]]
        return np.gradient(selected,1/fps,axis=0,edge_order=2)
    bones=np.stack([human[:,names.index(b)]-human[:,names.index(a)]
        for _,a,b,_,_ in velocity.source.bone_correspondences()],axis=1)
    bones/=np.linalg.norm(bones,axis=-1,keepdims=True)
    floor=float(np.quantile(human[:,[names.index('LeftToe'),names.index('RightToe')],2],.02))
    neutral=neutral_points(motion)
    neutral_floor=min(neutral[name][2] for name in ['lfoot','rfoot'])
    neutral_trunk_height=float((neutral['thorax'][2]-neutral_floor)*scale)
    data=mujoco.MjData(model);data.qpos[:]=velocity.kin.foot_geometry(model)[3]
    mujoco.mj_forward(model,data)
    body=model.body('torso_roll_link').id
    delta=float(data.xpos[body,2]-neutral_trunk_height)
    offset=data.xmat[body].reshape(3,3).T@np.array([0.,0.,delta])
    target_height=human[:,names.index('Spine2'),2]-floor+(trunk@offset)[:,2]
    contact_points=human[:,[names.index(name) for name,_,_ in velocity.CONTACTS]]
    contact_velocity=np.gradient(contact_points,1/fps,axis=0,edge_order=2)
    thresholds=np.array([height for _,_,height in velocity.CONTACTS])
    confidence=np.clip((thresholds-(contact_points[:,:,2]-floor))/(thresholds*.6),0.,1.)
    confidence*=np.clip((.35-np.linalg.norm(contact_velocity[:,:,:2],axis=-1))/.20,0.,1.)
    confidence*=np.clip((.4-abs(contact_velocity[:,:,2]))/.25,0.,1.)
    confidence=gaussian_filter1d(confidence,1.,axis=0,mode='nearest')
    result=dict(times=times,source_times=times,human=human,names=names,body_rotation=trunk,
        linear_velocity=landmark_velocity(['Spine2']+[name for name,_ in velocity.EFFECTORS]),
        angular_velocity=velocity.rotation_velocity(trunk,fps),
        gravity=np.einsum('tji,j->ti',trunk,[0.,0.,-1.]),
        bone_directions=bones,intermediate_linear_velocity=landmark_velocity([name for name,_ in velocity.INTERMEDIATES]),
        shoulder_linear_velocity=landmark_velocity([name for name,_ in velocity.SHOULDERS]),
        hip_linear_velocity=landmark_velocity([name for name,_ in velocity.HIPS]),
        pelvis_rotation=pelvis,pelvis_angular_velocity=velocity.rotation_velocity(pelvis,fps),
        hip_relative_rotation=relative,hip_relative_angular_velocity=velocity.rotation_velocity(relative,fps),
        trunk_height_target=target_height,trunk_height_rate=np.gradient(target_height,1/fps,edge_order=2),
        contact_confidence=confidence,contact_body_names=[body for _,body,_ in velocity.CONTACTS],
        source_floor_z_m=floor,trunk_height_local_offset_m=offset,source_fps=motion.fps,scale=scale,
        source_airborne=airborne,tracked_bodies=['torso_roll_link']+[body for _,body in velocity.EFFECTORS],
        height_calibration='ASF neutral thorax height above feet; offset rotates with trunk',
        trunk_orientation_method='ASF thorax FK rotation, anatomical neutral frame, native robot body frame',
        neutral_source_trunk_height_m=neutral_trunk_height)
    for key,value in result.items():
        if isinstance(value,np.ndarray) and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite CMU feature: {key}')
    return result
