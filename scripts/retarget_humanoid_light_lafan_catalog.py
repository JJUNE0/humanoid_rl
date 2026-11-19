"""Build one reviewed-length LAFAN sample per theme using landmark velocities.

Uses the current preview profile: weak bone pose=1, elbow/knee velocities=6,
hip velocities=6, shoulders=0. Source-coordinate normalization is shared with
the previous experiments. No joint-angle or Cartesian XY target is used.
Height/contact and calibrated hip/pelvis rotation tracking are optional.
These outputs are kinematic candidates, not dynamically verified training data.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

import retarget_humanoid_light_lafan_velocity as velocity

ROOT = velocity.ROOT
OUT = ROOT/'outputs/humanoid_light_lafan_catalog'
DESCRIPTIONS = {
    'walk':'Walking', 'run':'Running', 'sprint':'Sprinting', 'dance':'Dancing',
    'fight':'Fighting', 'fightAndSports':'Fighting / sports', 'jumps':'Jumping',
    'obstacles':'Obstacle locomotion', 'fallAndGetUp':'Falling / getting up',
    'ground':'Crawling / crouching', 'aiming':'Aiming', 'push':'Pushing',
    'pushAndStumble':'Push / stumble / recovery', 'pushAndFall':'Push / fall',
    'multipleActions':'Multiple actions',
}
ORDER = list(DESCRIPTIONS)
REVIEWED = {'walk':'walk1_subject1','dance':'dance1_subject2','fight':'fight1_subject2'}


def inventory():
    root = ROOT/'src/humanoid_retarget/datasets'
    manifest = json.loads((root/'lafan1/raw_manifest.json').read_text())
    grouped = {name:[] for name in ORDER}
    for entry in manifest['files']:
        item = dict(entry)
        item['stem'] = Path(item['path']).stem
        item['duration_seconds'] = (item['frames']-1)/item['fps']
        item['present'] = (root/'lafan1'/item['path']).is_file()
        theme = re.sub(r'\d+_subject\d+$','',item['stem'])
        grouped[theme].append(item)
    cmu = json.loads((root/'cmu/manifest.json').read_text())
    h1 = json.loads((root/'loco_mujoco/mocap/h1/manifest.json').read_text())
    return dict(lafan=dict(sequence_count=len(manifest['files']),
        frame_count=manifest['frame_count'],
        duration_seconds=sum((x['frames']-1)/x['fps'] for x in manifest['files']),
        original_human_bvh=True,themes={name:dict(description=DESCRIPTIONS[name],
            count=len(files),files=files) for name,files in grouped.items()}),
        cmu=dict(original_human_asf_amc=True,clips=cmu['clips']),
        additional_robot_motions=dict(robot='Unitree H1',already_retargeted=True,
            count=h1['sequence_count'],duration_seconds=h1['duration_seconds'],
            files=[x['path'] for x in h1['files']]),
        duplicate_lafan_derivatives='G1/H1 retargeted files are also present; they are not additional human recordings.')


def select_window(theme,item,seconds,extract,utils):
    path = ROOT/'src/humanoid_retarget/datasets/lafan1'/item['path']
    names,positions,_,_,fps = velocity.source.load_human(path,extract,utils)
    hips = positions[:,names.index('Hips')]
    root_velocity = np.gradient(hips,1/fps,axis=0,edge_order=2)
    hands = positions[:,[names.index('LeftHand'),names.index('RightHand')]]
    feet = positions[:,[names.index('LeftFoot'),names.index('RightFoot')]]
    hand_speed = np.linalg.norm(np.gradient(hands-hips[:,None],1/fps,axis=0),axis=-1).mean(axis=-1)
    foot_speed = np.linalg.norm(np.gradient(feet-hips[:,None],1/fps,axis=0),axis=-1).mean(axis=-1)
    root_speed = np.linalg.norm(root_velocity,axis=-1)
    floor = np.quantile(positions[:,[names.index('LeftToe'),names.index('RightToe')],2],.02)
    if theme in ['run','sprint','walk']:
        score = root_speed+.2*foot_speed
        reason = 'high mean locomotion speed'
    elif theme in ['jumps','obstacles']:
        score = np.maximum(feet[:,:,2]-floor,0).mean(axis=-1)+.4*abs(root_velocity[:,2])+.1*foot_speed
        reason = 'elevated feet and vertical movement'
    elif theme in ['fallAndGetUp','pushAndFall','ground']:
        score = np.maximum(np.quantile(hips[:,2],.8)-hips[:,2],0)+.4*abs(root_velocity[:,2])+.08*hand_speed
        reason = 'low body height and vertical movement'
    else:
        score = .5*hand_speed+.25*foot_speed+.25*root_speed
        reason = 'high mean limb and root movement'
    width = round(seconds*fps)+1
    if width>len(score):raise ValueError(f'{path.name} shorter than requested sample')
    window = np.convolve(score,np.ones(width)/width,mode='valid')
    margin = min(round(fps),max(0,(len(window)-1)//4))
    eligible = window[margin:len(window)-margin] if margin else window
    best = margin+int(np.argmax(eligible))
    start = round(best/fps*50)/50
    start = min(start,np.floor((item['duration_seconds']-seconds)*50)/50)
    return max(0.,start),reason,float(floor)


def render_pair(model,q,ref,path,theme,source_start,before=None,candidate_title=None,baseline_title=None,source_label='Original LAFAN',camera_azimuth=10.):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    chains = [['Hips','LeftUpLeg','LeftLeg','LeftFoot','LeftToe'],
              ['Hips','RightUpLeg','RightLeg','RightFoot','RightToe'],
              ['Hips','Spine','Spine1','Spine2','Neck','Head'],
              ['Spine2','LeftShoulder','LeftArm','LeftForeArm','LeftHand'],
              ['Spine2','RightShoulder','RightArm','RightForeArm','RightHand']]
    edges = [(ref['names'].index(a),ref['names'].index(b)) for chain in chains for a,b in zip(chain,chain[1:])]
    human = ref['human'].copy();human[:,:,2] -= ref['source_floor_z_m']
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    option,camera,data = mujoco.MjvOption(),mujoco.MjvCamera(),mujoco.MjData(model)
    # The plane collision is infinite; enlarge its visible extent so clips
    # with several metres of travel do not appear to leave the ground.
    for gid in np.flatnonzero(model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE):
        model.geom_size[gid,:2]=50.
    columns=3 if before is not None else 2
    with mujoco.Renderer(model,height=440,width=560) as renderer,imageio.get_writer(
            path,fps=25,codec='libx264',quality=7,macro_block_size=1) as writer:
        for frame in range(0,len(q)-1,2):
            canvas=Image.new('RGB',(560*columns,512),'#e9eef2');draw=ImageDraw.Draw(canvas)
            for column in range(columns):
                pose=before[frame] if before is not None and column==1 else q[frame]
                data.qpos[:]=pose;mujoco.mj_forward(model,data)
                option.geomgroup[:]=0;option.geomgroup[0]=1
                if column:option.geomgroup[2]=1
                camera.lookat[:]=(pose[:3] if column else human[frame,ref['names'].index('Hips')])
                camera.lookat[2]=.7
                camera.distance,camera.azimuth,camera.elevation=3.,camera_azimuth,-12.
                renderer.update_scene(data,camera=camera,scene_option=option)
                if column==0:
                    for a,b in edges:
                        scene=renderer.scene;g=scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),
                            np.eye(3).ravel(),np.array([1.,.55,.12,1.],np.float32))
                        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.012,human[frame,a],human[frame,b]);scene.ngeom+=1
                canvas.paste(Image.fromarray(renderer.render()),(560*column,35))
            draw.text((8,8),f'{source_label}: {DESCRIPTIONS[theme]}',font=font,fill='#17242d')
            if before is not None:
                draw.text((568,8),baseline_title or 'Before: velocity + weak pose + hips',font=font,fill='#17242d')
            draw.text((560*(columns-1)+8,8),candidate_title or 'Light: velocities + weak pose + hip velocities',font=font,fill='#17242d')
            draw.text((8,480),f'{theme} | source {source_start+frame/50:.2f}s | Kinematic preview; training_ready=False',font=font,fill='#9b2929')
            writer.append_data(np.asarray(canvas))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=5.)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--no-render',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Reuse matching numerical results; regenerate videos')
    parser.add_argument('--height-weight',type=float,default=0.)
    parser.add_argument('--contact-weight',type=float,default=0.)
    parser.add_argument('--slip-weight',type=float,default=0.)
    parser.add_argument('--hip-rotation-weight',type=float,default=0.)
    parser.add_argument('--pelvis-rotation-weight',type=float,default=0.)
    parser.add_argument('--baseline-dir',type=Path,help='Older catalog with identical source windows')
    args=parser.parse_args()
    if args.seconds<1 or not np.isfinite(args.seconds):parser.error('seconds must be finite and at least 1')
    if min(args.height_weight,args.contact_weight,args.slip_weight,args.hip_rotation_weight,args.pelvis_rotation_weight)<0 or not np.isfinite([args.height_weight,args.contact_weight,args.slip_weight,args.hip_rotation_weight,args.pelvis_rotation_weight]).all():
        parser.error('Height/contact weights must be finite and nonnegative')
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    height_contact_enabled=max(args.height_weight,args.contact_weight,args.slip_weight)>0
    rotation_enabled=max(args.hip_rotation_weight,args.pelvis_rotation_weight)>0
    if (height_contact_enabled or rotation_enabled) and output==OUT.resolve():parser.error('Use a separate output-dir to preserve the previous catalog')
    baselines={}
    if args.baseline_dir:
        baselines={x['theme']:x for x in json.loads((args.baseline_dir/'results.json').read_text())['results']}
    data=inventory();(output/'inventory.json').write_text(json.dumps(data,indent=2)+'\n')
    _,extract,utils=velocity.source.prepare()
    model=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/humanoid_light_conversion/scene.xml'))
    selection=[]
    for theme in ORDER:
        files=data['lafan']['themes'][theme]['files']
        preferred=next((x for x in files if x['stem'].endswith('_subject2')),files[0])
        if theme in REVIEWED:
            preferred=next(x for x in files if x['stem']==REVIEWED[theme])
        start,reason,floor=select_window(theme,preferred,args.seconds,extract,utils)
        if theme in REVIEWED:
            start,reason=5.,'same source and start as the previously reviewed example'
        selection.append(dict(theme=theme,description=DESCRIPTIONS[theme],source=preferred['stem'],
            source_sha256=preferred['sha256'],start_seconds=start,duration_seconds=args.seconds,
            selection_reason=reason,source_floor_z_m=floor,
            source_has_unmodeled_context=theme in ['obstacles','aiming','push','pushAndFall','pushAndStumble']))
    (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    results=[];began=time.perf_counter()
    for item in selection:
        theme,name,start=item['theme'],item['source'],item['start_seconds']
        directory=output/theme;directory.mkdir(exist_ok=True)
        command=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/retarget_humanoid_light_lafan_velocity.py'),
            '--clip',f'{name}:{start:g}:{args.seconds:g}','--pose-weight','1','--intermediate-weight','6',
            '--hip-weight','6','--shoulder-weight','0','--output-dir',str(directory),'--no-render','--skip-comparison']
        command.extend(['--height-weight',str(args.height_weight),'--contact-weight',str(args.contact_weight),
                        '--slip-weight',str(args.slip_weight),'--hip-rotation-weight',str(args.hip_rotation_weight),
                        '--pelvis-rotation-weight',str(args.pelvis_rotation_weight)])
        item['command']=command
        try:
            label=f'{name}_{start:g}s_{args.seconds:g}s'
            report_path=directory/f'{label}.json'
            reused=False
            if args.resume and report_path.exists() and (directory/f'{label}.npz').exists():
                prior=json.loads(report_path.read_text())
                reused=(prior['source_sha256']==item['source_sha256'] and
                    prior['optimizer_sha256']==velocity.source.digest(Path(velocity.__file__)) and
                    prior['robot_sha256']==velocity.source.digest(ROOT/'outputs/humanoid_light_conversion/humanoid_light.xml') and
                    prior['source_start_seconds']==start and prior['frames']==round(args.seconds*50)+1 and
                    prior['fps']==50 and prior['substeps']==4 and prior['pose_weight']==1 and
                    prior['pose_gain_per_second']==2 and prior['intermediate_velocity_weight']==6 and
                    prior['hip_velocity_weight']==6 and prior['shoulder_velocity_weight']==0 and
                    prior.get('trunk_height_weight',0)==args.height_weight and
                    prior.get('contact_height_weight',0)==args.contact_weight and
                    prior.get('contact_slip_weight',0)==args.slip_weight and
                    prior.get('hip_rotation_weight',0)==args.hip_rotation_weight and
                    prior.get('pelvis_rotation_weight',0)==args.pelvis_rotation_weight)
            if not reused:
                with (directory/'run.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads((directory/f'{label}.json').read_text())
            row=dict(item,status=report['status'],checks=report['checks'],numerical_result_reused=reused,
                report=str((directory/f'{label}.json').relative_to(output)),
                motion=str((directory/f'{label}.npz').relative_to(output)),
                primary_velocity_rmse_m_s=float(np.sqrt(np.mean(np.square(list(report['velocity_rmse_m_s'].values()))))),
                hip_velocity_rmse_m_s=float(np.sqrt(np.mean(np.square(list(report['hip_velocity_rmse_m_s'].values()))))),
                bone_direction_median_error_deg=report['bone_direction_error_deg']['median'],
                trunk_tilt_p95_error_deg=report['trunk_tilt_error_deg']['p95'],
                rejected_substeps=report['rejected_steps'],qp_failures=report['qp_failures'])
            row['height_contact']=report['height_contact']
            row['rotation']=dict(hip_relative_rotation_error_deg=report['hip_relative_rotation_error_deg'],
                                 pelvis_rotation_error_deg=report['pelvis_rotation_error_deg'])
            row['hip_relative_rotation_median_deg']=report['hip_relative_rotation_error_deg']['median']
            row['pelvis_rotation_median_deg']=report['pelvis_rotation_error_deg']['median']
            before=None;ref=None
            if theme in baselines:
                old=baselines[theme]
                if (old['source']==name and old['start_seconds']==start and old['duration_seconds']==args.seconds):
                    with np.load(args.baseline_dir/old['motion'],allow_pickle=False) as z:before=z['qpos']
                    path=ROOT/'src/humanoid_retarget/datasets/lafan1/raw'/f'{name}.bvh'
                    ref=velocity.references(path,start,args.seconds,50.,model,extract,utils)
                    row['before_height_contact'],_=velocity.height_contact_measurements(model,before,
                        velocity.source.differentiate(model,before,50.),ref)
                    row['before_rotation'],_=velocity.rotation_measurements(model,before,ref)
                    row['baseline_motion']=str((args.baseline_dir/old['motion']).resolve())
            if not args.no_render:
                path=ROOT/'src/humanoid_retarget/datasets/lafan1/raw'/f'{name}.bvh'
                if ref is None:ref=velocity.references(path,start,args.seconds,50.,model,extract,utils)
                ref['source_floor_z_m']=item['source_floor_z_m']*ref['scale']
                with np.load(directory/f'{label}.npz',allow_pickle=False) as z:q=z['qpos']
                render_pair(model,q,ref,directory/'preview.mp4',theme,start,before=before,
                            candidate_title=('After: pelvis + relative hip rotation' if rotation_enabled else
                                             'After: add trunk height + inferred contacts' if height_contact_enabled else None),
                            baseline_title='Before: height/contact, no hip rotation' if rotation_enabled else None)
                row['video']=str((directory/'preview.mp4').relative_to(output))
            results.append(row)
        except Exception as exc:
            results.append(dict(item,status='processing_failed',error=str(exc)))
        print(json.dumps(dict(completed=len(results),total=len(selection),theme=theme,status=results[-1]['status'],elapsed_seconds=time.perf_counter()-began)),flush=True)
        (output/'results.json').write_text(json.dumps(dict(results=results,training_ready=False,dynamic_feasibility_verified=False),indent=2)+'\n')
    with (output/'catalog.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['theme','description','source','start_seconds','duration_seconds','status',
                'primary_velocity_rmse_m_s','hip_velocity_rmse_m_s','bone_direction_median_error_deg','trunk_tilt_p95_error_deg',
                'hip_relative_rotation_median_deg','pelvis_rotation_median_deg']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    if not args.no_render:
        from PIL import Image,ImageDraw,ImageFont
        import imageio.v2 as imageio
        playlist=output/'playlist.txt'
        videos=[x for x in results if 'video' in x]
        playlist.write_text(''.join("file '"+x['video']+"'\n" for x in videos))
        subprocess.run(['ffmpeg','-v','error','-f','concat','-safe','0','-i',str(playlist),'-c','copy',
                        '-movflags','+faststart','-y',str(output/'all_15_themes.mp4')],check=True)
        thumbnail_columns=2 if baselines else 3
        thumbnail_width=1680//thumbnail_columns
        canvas=Image.new('RGB',(1680,280*((len(videos)+thumbnail_columns-1)//thumbnail_columns)),'#e9eef2');draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
        for i,item in enumerate(videos):
            with imageio.get_reader(output/item['video']) as reader:frame=reader.get_data(round(args.seconds*25/2))
            x,y=(i%thumbnail_columns)*thumbnail_width,(i//thumbnail_columns)*280
            canvas.paste(Image.fromarray(frame).resize((thumbnail_width,256)),(x,y+24))
            draw.text((x+6,y+2),f'{i+1:02d} {item["theme"]}',font=font,fill='#17242d')
        canvas.save(output/'contact_sheet.png')
    print('FINISHED',len(results),'samples',flush=True)


if __name__=='__main__':main()
