"""Screen the COM-first candidate without relaxing the physical criteria."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

import retarget_humanoid_light_cmu_com_first as opt
import validate_humanoid_light_cmu_dynamics as checks
import validate_humanoid_light_cmu_landing as geometry

ROOT,OUT=opt.ROOT,opt.OUT


def render(model,q,old,series,old_series,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    source=np.load(OUT/'88_01_com_first.npz',allow_pickle=False)
    target=source['com_target'];origin=plan.get('time_origin_seconds',0.);times=origin+np.arange(len(q))*.02
    first_old=round(origin*50)
    old_view=old[first_old:first_old+len(q)]
    fig,axes=plt.subplots(4,1,figsize=(11,10),constrained_layout=True)
    axes[0].plot(times,old_series['com'][first_old:first_old+len(q),2],label='Previous COM')
    axes[0].plot(times,series['com'][:,2],label='COM-first candidate')
    axes[0].plot(times,target[:,2],'--',label='COM target')
    axes[0].plot(times,q[:,2],':',label='Candidate pelvis height')
    axes[0].set(ylabel='Height (m)');axes[0].legend(ncol=2)
    reference=R.from_matrix(source['body_rotation_target'])
    error=(reference.inv()*R.from_quat(q[:,3:7],scalar_first=True)).magnitude()*180/np.pi
    axes[1].plot(times,error,label='Body orientation error')
    axes[1].set(ylabel='Orientation error (deg)');axes[1].legend()
    axes[2].plot(times,old_series['angular_momentum'][first_old:first_old+len(q),1],label='Previous H_y')
    axes[2].plot(times,series['angular_momentum'][:,1],label='Candidate H_y')
    axes[2].set(ylabel='kg m^2 / s');axes[2].legend()
    axes[3].plot(times,series['sole_heights'][:,0],label='Left sole')
    axes[3].plot(times,series['sole_heights'][:,1],label='Right sole')
    axes[3].set(ylabel='Sole height (m)',ylim=(-.05,.35),xlabel='Time (s)');axes[3].legend()
    for ax in axes:
        for key in ['touchdown_seconds','absorption_end_seconds','rebound_takeoff_seconds','rebound_touchdown_seconds']:
            ax.axvline(plan[key],color='black',lw=.5)
        ax.grid(alpha=.2)
    fig.savefig(OUT/'com_body_comparison.png',dpi=140);plt.close(fig)
    actual=np.load(OUT/'physics_rollout.npz',allow_pickle=False)
    start=round(float(actual['start_seconds'])*50)
    option=mujoco.MjvOption();option.geomgroup[:]=0;option.geomgroup[[0,2]]=1
    camera=mujoco.MjvCamera();data=mujoco.MjData(model)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    labels=['Previous candidate','COM / body priority + free footprints','Actual MuJoCo, bounded joint torques']
    with mujoco.Renderer(model,height=440,width=560) as renderer,imageio.get_writer(
        OUT/'com_first_diagnostic.mp4',fps=25,codec='libx264',quality=7,macro_block_size=1) as writer:
        for i in range(max(0,54-first_old),min(125-first_old,len(q))):
            canvas=Image.new('RGB',(1680,512),'#e9eef2');draw=ImageDraw.Draw(canvas)
            for k,pose in enumerate([old_view[i],q[i],actual['qpos'][i-start]]):
                data.qpos[:]=pose;mujoco.mj_forward(model,data)
                camera.lookat[:]=[q[i,0],q[i,1],.75];camera.distance=3.6;camera.azimuth=55.;camera.elevation=-12.
                renderer.update_scene(data,camera=camera,scene_option=option)
                canvas.paste(Image.fromarray(renderer.render()),(k*560,35))
                draw.text((k*560+8,8),labels[k],font=font,fill='#17242d')
            draw.text((8,480),f'{origin+i/50:.2f}s | 0.5x | Experimental candidate; training_ready=False',font=font,fill='#9b2929')
            writer.append_data(np.asarray(canvas))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-render',action='store_true');args=parser.parse_args()
    z=np.load(OUT/'88_01_com_first.npz',allow_pickle=False);q=z['qpos']
    plan=json.loads((OUT/'phase_plan.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/humanoid_light_conversion/scene.xml'))
    report,series=checks.screen(model,q,z['source_contact'],z['contact_forces_world'],z['contact_impulses_world'],phase_plan=plan)
    sub=geometry.subframe_check(model,q)
    report['checks'].update(interpolated_floor_clear=sub['minimum_floor_height_m']>-.001,
        interpolated_selected_self_clear=sub['minimum_selected_self_gap_m']>.0002,
        interpolated_distance_queries_resolved=sub['unresolved_zero_distance_queries']==0)
    report['collocation_passed']=all(report['checks'].values())
    previous=np.load(opt.dyn.OUT/'88_01_candidate.npz',allow_pickle=False)
    old,old_series=checks.screen(model,previous['qpos'],previous['source_contact'],
        previous['contact_forces_world'],previous['contact_impulses_world'])
    origin=plan.get('time_origin_seconds',0.)
    td=round((plan['touchdown_seconds']-origin)*50);abs_end=round((plan['absorption_end_seconds']-origin)*50)
    body_error=(R.from_matrix(z['body_rotation_target']).inv()*R.from_quat(q[:,3:7],scalar_first=True)).magnitude()*180/np.pi
    report['com_body_objectives']=dict(touchdown_com_height_m=float(series['com'][td,2]),
        touchdown_com_target_m=float(z['com_target'][td,2]),
        minimum_absorption_com_height_m=float(series['com'][td:abs_end+1,2].min()),
        absorption_target_m=float(z['com_target'][abs_end,2]),
        touchdown_pelvis_height_m=float(q[td,2]),touchdown_body_error_deg=float(body_error[td]),
        previous_touchdown_com_height_m=float(old_series['com'][67,2]))
    simulated,sim=checks.rollout(model,q,series,start=0. if plan.get('partial_motion',False) else .76,
        initial_velocity=plan.get('initial_velocity'),time_origin=origin)
    np.savez_compressed(OUT/'physics_rollout.npz',**sim)
    np.savez_compressed(OUT/'metrics.npz',**series)
    result=dict(status='pending_additional_physics_validation' if report['collocation_passed'] else 'candidate_rejected',
        after=report,before=old,subframe_checks=sub,forward_rollout=simulated,phase_plan=plan,
        training_ready=False,dynamic_feasibility_verified=False,
        optimizer_sha256=opt.ik.digest(Path(opt.__file__)),validator_sha256=opt.ik.digest(Path(__file__)),
        candidate_sha256=opt.ik.digest(OUT/'88_01_com_first.npz'),
        robot_sha256=opt.ik.digest(ROOT/'outputs/humanoid_light_conversion/humanoid_light.xml'),
        limitations=['Touchdown timing is searched over five grid candidates, not continuously optimized.',
                    'Footprint XY and yaw are continuous decisions; vertical contact points must lie on the ground.',
                    'Impact is a 20 ms approximation; forward simulation uses 1 ms steps.',
                    'Physical rollout uses bounded ideal joint torques; source delay and motor coupling are not modeled.'])
    if plan.get('partial_motion',False):
        result['limitations'].append(
            'This is a landing-window experiment starting from a prescribed aerial state at '
            f'{origin:.2f} s; takeoff and the preceding motion are not validated.')
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if not args.no_render:render(model,q,previous['qpos'],series,old_series,plan)


if __name__=='__main__':main()
