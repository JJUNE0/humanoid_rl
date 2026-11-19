"""Independently screen optimized landing poses and create review artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R, Slerp

import optimize_humanoid_light_cmu_landing as opt
import retarget_humanoid_light_lafan as ik
from humanoid_light_collision import MeshCollisionIndex

ROOT=Path(__file__).resolve().parents[1]
OUT=opt.OUT
sys.path.insert(0,str(ROOT/'src'))
from humanoid_retarget.datasets.robot_motion import load_robot_motion


def measurements(model,qpos,foot_goal,td):
    data=mujoco.MjData(model)
    meshes,foot_meshes,ids,_,p0,r0=opt.foot_geometry(model)
    vertices=dict(meshes);pairs=ik.self_collision_pairs(model)
    footprints=[]
    for i,side in enumerate(['left','right']):
        v=foot_meshes[side][1]@r0[i].T+foot_goal[i]
        footprints.extend(v[v[:,2]<v[:,2].min()+.004,:2])
    planes=ConvexHull(np.array(footprints)).equations
    positions=[];rotations=[];bottoms=[];soles=[];coms=[];gaps=[];certificates=0;unresolved=0
    geom_r=[];geom_p=[];low_vertices=[]
    for pose in qpos:
        data.qpos[:]=pose;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        positions.append(data.xpos[ids].copy());rotations.append(data.xmat[ids].reshape(2,3,3).copy());coms.append(data.subtree_com[1].copy())
        all_bottom=[];foot_bottom=[];foot_r=[];foot_p=[];foot_v=[]
        for g,v in meshes:
            z=v@data.geom_xmat[g].reshape(3,3)[2]+data.geom_xpos[g,2];all_bottom.append(z.min())
        for side in ['left','right']:
            g=foot_meshes[side][0];v=vertices[g];matrix=data.geom_xmat[g].reshape(3,3);z=v@matrix[2]+data.geom_xpos[g,2]
            foot_bottom.append(float(z.min()));foot_r.append(matrix.copy());foot_p.append(data.geom_xpos[g].copy());foot_v.append(v[z.argmin()].copy())
        bottoms.append(min(all_bottom));soles.append(foot_bottom);geom_r.append(foot_r);geom_p.append(foot_p);low_vertices.append(foot_v)
        distances=[]
        for a,b in pairs:
            distance,_,certificate=opt.checked_mesh_distance(model,data,a,b,vertices)
            distances.append(distance);certificates+=int(certificate);unresolved+=int(distance==0)
        gaps.append(min(distances))
    positions,rotations,coms=np.array(positions),np.array(rotations),np.array(coms)
    geom_r,geom_p,low_vertices=np.array(geom_r),np.array(geom_p),np.array(low_vertices)
    velocity=np.gradient(geom_p,.02,axis=0)+np.einsum('tsij,tsj->tsi',np.gradient(geom_r,.02,axis=0),low_vertices)
    speed=np.linalg.norm(velocity[:,:,:2],axis=-1)
    post=np.arange(len(qpos))>=td+2
    error=np.linalg.norm(positions[td:]-foot_goal,axis=-1)
    root_r=R.from_quat(qpos[:,3:7],scalar_first=True).as_matrix()
    tilt=np.rad2deg(np.arccos(np.clip(root_r[:,2,2],-1,1)))
    foot_up=rotations@np.swapaxes(r0,-1,-2)
    foot_tilt=np.rad2deg(np.arccos(np.clip(foot_up[:,:,2,2],-1,1)))
    margin=-np.max(coms[td:,:2]@planes[:,:2].T+planes[:,2],axis=1)
    result=dict(landing_foot_position_error_m_max=float(error.max()),landing_foot_position_error_m_p95=float(np.quantile(error,.95)),
        landing_sole_material_xy_speed_m_s_max=float(speed[post].max()),landing_sole_material_xy_speed_m_s_p95=float(np.quantile(speed[post],.95)),
        touchdown_root_tilt_deg=float(tilt[td]),post_landing_root_tilt_deg_max=float(tilt[td:].max()),
        post_landing_foot_tilt_deg_max=float(foot_tilt[td:].max()),post_landing_com_support_margin_m_min=float(margin.min()),
        floor_height_m_min=float(min(bottoms)),selected_self_gap_m_min=float(min(gaps)),
        separating_plane_fallbacks=certificates,unresolved_zero_distance_queries=unresolved)
    series=dict(foot_positions=positions,sole_heights=np.array(soles),com=coms,root_tilt_deg=tilt,foot_tilt_deg=foot_tilt,
                floor_height=np.array(bottoms),self_gap=np.array(gaps),sole_material_xy_speed=speed)
    return result,series


def subframe_check(model,qpos,pairs=None):
    meshes,_,_,_,_,_=opt.foot_geometry(model);vertices=dict(meshes)
    extended=pairs is not None
    pairs=ik.self_collision_pairs(model) if pairs is None else pairs
    index=MeshCollisionIndex(model,meshes,pairs,opt.checked_mesh_distance) if extended else None
    data=mujoco.MjData(model);min_height=np.inf;min_gap=np.inf;unknown=0;certified=0
    for first,second in zip(qpos[:-1],qpos[1:]):
        rotations=Slerp([0.,1.],R.from_quat(np.stack([first[3:7],second[3:7]]),scalar_first=True))([.25,.5,.75]).as_quat(scalar_first=True)
        for u,quaternion in zip([.25,.5,.75],rotations):
            data.qpos[:]=(1-u)*first+u*second;data.qpos[3:7]=quaternion
            mujoco.mj_kinematics(model,data)
            if index is not None:
                index.update(data)
                min_height=min(min_height,index.minimum_height(data))
                min_gap=min(min_gap,.05)
                for a,b,distance,_,certificate in index.query(data,.05):
                    min_gap=min(min_gap,distance);unknown+=int(distance==0);certified+=int(certificate)
            else:
                for g,v in meshes:min_height=min(min_height,float((v@data.geom_xmat[g].reshape(3,3)[2]+data.geom_xpos[g,2]).min()))
                for a,b in pairs:
                    distance,_,certificate=opt.checked_mesh_distance(model,data,a,b,vertices);min_gap=min(min_gap,distance);unknown+=int(distance==0);certified+=int(certificate)
    return dict(samples=3*(len(qpos)-1),minimum_floor_height_m=min_height,minimum_selected_self_gap_m=min_gap,
                unresolved_zero_distance_queries=unknown,separating_plane_fallbacks=certified)


def render_artifacts(model,qpos,series,before_series,plan):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=dict(np.load(ROOT/'outputs/humanoid_light_cmu/88_01_targets.npz',allow_pickle=False))
    scale=opt.json_load(ROOT/'outputs/humanoid_light_cmu/88_01_report.json')['target_leg_scale']
    ik.render(model,qpos,target['human_positions'],target['human_joint_names'].tolist(),50.,OUT/'88_01_optimized_comparison.mp4',human_scale=scale)
    first=imageio.get_reader(ROOT/'outputs/humanoid_light_cmu/88_01_comparison.mp4')
    second=imageio.get_reader(OUT/'88_01_optimized_comparison.mp4')
    count=min(first.count_frames(),second.count_frames())
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    with imageio.get_writer(OUT/'88_01_before_after.mp4',fps=25,codec='libx264',quality=7,macro_block_size=1) as writer:
        for t in range(count):
            canvas=Image.new('RGB',(1600,640),'#e9eef2')
            canvas.paste(Image.fromarray(first.get_data(t)),(0,40));canvas.paste(Image.fromarray(second.get_data(t)),(800,40))
            draw=ImageDraw.Draw(canvas);draw.text((15,7),'Before: sequential IK',fill='#17242d',font=font);draw.text((815,7),'After: trajectory + landing posture',fill='#17242d',font=font)
            writer.append_data(np.asarray(canvas))
    first.close();second.close()
    option=mujoco.MjvOption();option.geomgroup[:]=0;option.geomgroup[[0,2]]=1;camera=mujoco.MjvCamera();data=mujoco.MjData(model)
    canvas=Image.new('RGB',(1500,830),'#e9eef2')
    sample_times=[.72,.96,1.2,plan['touchdown_seconds'],plan['absorption_end_seconds'],2.4]
    with mujoco.Renderer(model,height=400,width=500) as renderer:
        for k,seconds in enumerate(sample_times):
            frame=round(seconds*50);data.qpos[:]=qpos[frame];mujoco.mj_forward(model,data)
            camera.lookat[:]=[qpos[frame,0],qpos[frame,1],.85];camera.distance=3.2;camera.azimuth=30;camera.elevation=-10
            renderer.update_scene(data,camera=camera,scene_option=option);canvas.paste(Image.fromarray(renderer.render()),((k%3)*500,(k//3)*415+15))
            ImageDraw.Draw(canvas).text(((k%3)*500+6,(k//3)*415),f'Optimized {seconds:.2f}s',fill='#17242d')
    canvas.save(OUT/'88_01_optimized_montage.png')
    times=np.arange(len(qpos))/50
    fig,axes=plt.subplots(3,1,figsize=(11,8),constrained_layout=True)
    axes[0].plot(times,before_series['sole_heights'][:,0]*100,'--',label='Before left');axes[0].plot(times,before_series['sole_heights'][:,1]*100,'--',label='Before right')
    axes[0].plot(times,series['sole_heights'][:,0]*100,label='After left');axes[0].plot(times,series['sole_heights'][:,1]*100,label='After right');axes[0].set(ylabel='Sole height (cm)',ylim=(-2,80));axes[0].legend(ncol=2)
    axes[1].plot(times,before_series['root_tilt_deg'],'--',label='Before');axes[1].plot(times,series['root_tilt_deg'],label='After');axes[1].set(ylabel='Pelvis tilt (deg)');axes[1].legend()
    axes[2].plot(times,series['com'][:,0],label='COM X');axes[2].plot(times,series['com'][:,1],label='COM Y');axes[2].set(ylabel='COM position (m)',xlabel='Time (s)');axes[2].legend()
    for axis in axes:
        axis.axvline(plan['touchdown_seconds'],color='black',linewidth=.8,label='Touchdown');axis.axvspan(plan['touchdown_seconds'],times[-1],color='green',alpha=.06);axis.grid(alpha=.2)
    fig.savefig(OUT/'landing_metrics.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-render',action='store_true');args=parser.parse_args()
    motion=load_robot_motion(OUT/'88_01_optimized.npz');qpos=motion.qpos.astype(float);qvel=motion.qvel.astype(float)
    plan=opt.json_load(OUT/'phase_plan.json');td=round(plan['touchdown_seconds']*50);foot_goal=np.asarray(plan['landing_foot_positions'])
    model=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/humanoid_light_conversion/scene.xml'))
    after,series=measurements(model,qpos,foot_goal,td)
    before_q=np.load(ROOT/'outputs/humanoid_light_cmu/88_01_50hz.npz',allow_pickle=False)['qpos'].astype(float)
    before,before_series=measurements(model,before_q,foot_goal,td)
    subframes=subframe_check(model,qpos)
    limits=opt.json_load(ik.OUTPUT/'light_profile.json')['velocity_limits_joint_coordinate_rad_s']
    ratio=0.;excess=0.
    for i,name in enumerate(motion.joint_names[1:]):
        j=model.joint(name);a=qpos[:,7+i];excess=max(excess,float(np.maximum(j.range[0]-a,a-j.range[1]).max()))
        ratio=max(ratio,float(abs(np.diff(a)*50).max()/limits[name]))
    root_rotation=R.from_quat(qpos[:,3:7],scalar_first=True).as_matrix()
    turn=cmu_pitch(root_rotation)
    after.update(joint_limit_excess_rad=max(0.,excess),maximum_interval_joint_speed_to_limit_ratio=ratio,
        terminal_speed_max=float(abs(qvel[-10:]).max()),accumulated_local_pitch_deg=turn)
    actual_contact=np.flatnonzero((np.asarray(series['sole_heights']).max(axis=1)<.002)&(np.arange(len(qpos))>round(1.1*50)))
    after['first_double_support_after_flight_s']=float(actual_contact[0]/50) if len(actual_contact) else None
    checks=dict(landing_feet_within_2mm=after['landing_foot_position_error_m_max']<.002,
        support_material_speed_below_1cm_s=after['landing_sole_material_xy_speed_m_s_max']<.01,
        landing_foot_tilt_below_1deg=after['post_landing_foot_tilt_deg_max']<1,
        upright_at_touchdown=after['touchdown_root_tilt_deg']<2,
        com_inside_support_polygon=after['post_landing_com_support_margin_m_min']>.01,
        joint_ranges_respected=excess<1e-6,joint_speeds_respected=ratio<1.0005,
        no_floor_penetration=after['floor_height_m_min']>-.0001 and subframes['minimum_floor_height_m']>-.0001,
        selected_self_collisions_clear=after['selected_self_gap_m_min']>.0002 and subframes['minimum_selected_self_gap_m']>.0002,
        no_unresolved_mesh_queries=after['unresolved_zero_distance_queries']==0 and subframes['unresolved_zero_distance_queries']==0,
        complete_turn=abs(turn+360)<3,settled_terminal_pose=after['terminal_speed_max']<.01)
    report=dict(status='passed_kinematic_landing_checks' if all(checks.values()) else 'needs_correction',checks=checks,
        phase_plan=plan,before=before,after=after,subframe_checks=subframes,training_ready=False,dynamic_feasibility_verified=False,
        limitations=['No actuator torque, impact-force, friction-cone or angular-momentum feasibility certificate.',
          'Collision checks cover all robot-floor hulls and the listed 20 nonadjacent self pairs.',
          'Original pelvis rotation/tuck and post-landing rebound are adapted to a robot-feasible upright landing and recovery.'],
        source_asf_sha256=ik.digest(cmu_data()/'88.asf'),source_amc_sha256=ik.digest(cmu_data()/'88_01.amc'),
        robot_xml_sha256=ik.digest(ROOT/'outputs/humanoid_light_conversion/humanoid_light.xml'),
        optimizer_sha256=ik.digest(Path(opt.__file__)),validator_sha256=ik.digest(Path(__file__)))
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(OUT/'metrics.npz',**series)
    print(json.dumps(dict(status=report['status'],checks=checks,after=after,subframe_checks=subframes),indent=2),flush=True)
    if not args.no_render:render_artifacts(model,qpos,series,before_series,plan)


def cmu_pitch(rotations):
    return float(np.rad2deg(R.from_matrix(np.swapaxes(rotations[:-1],-1,-2)@rotations[1:]).as_rotvec()[:,1].sum()))


def cmu_data():return ROOT/'src/humanoid_retarget/datasets/cmu/raw'


if __name__=='__main__':main()
