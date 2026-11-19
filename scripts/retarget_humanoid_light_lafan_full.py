"""Resume-capable full-length conversion of the 77 original LAFAN sequences.

Writes compact qpos/qvel archives, inferred contacts and validation reports.
The original dataset/model and previous experiments are never overwritten.
No reinforcement-learning training is started by this script.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import types

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import mujoco
import numpy as np

import retarget_humanoid_light_lafan_velocity as opt

ROOT=opt.ROOT
OUT=ROOT/'outputs/humanoid_light_lafan_full'
PROFILE=dict(fps=50.,substeps=4,pose_weight=1.,pose_gain=2.,intermediate_weight=6.,
    shoulder_weight=0.,hip_weight=6.,height_weight=12.,contact_weight=80.,slip_weight=6.,
    hip_rotation_weight=6.,pelvis_rotation_weight=2.,joint_limit_weight=6.,
    self_collision_weight=80.,collision_coverage='full')
RUNTIME={}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def initialize_worker():
    # Leave interactive desktop processes priority over the long batch.
    os.nice(5)
    dependencies=opt.source.OUTPUT/'python_deps';sys.path.insert(0,str(dependencies))
    package=types.ModuleType('light_lafan_vendor')
    package.__path__=[str(opt.source.OUTPUT/'upstream/general_motion_retargeting/utils/lafan_vendor')]
    sys.modules['light_lafan_vendor']=package
    RUNTIME.update(model=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/humanoid_light_conversion/scene.xml')),
        profile=json.loads((opt.source.OUTPUT/'light_profile.json').read_text()),
        extract=importlib.import_module('light_lafan_vendor.extract'),
        utils=importlib.import_module('light_lafan_vendor.utils'))


def solver_for(model,profile):
    keys={k:v for k,v in PROFILE.items() if k not in ['fps','substeps']}
    return opt.VelocityIK(model,profile,1/(PROFILE['fps']*PROFILE['substeps']),**keys)


def convert_one(job):
    started=time.perf_counter();name=job['name'];out=Path(job['output'])
    directory=out/'motions';directory.mkdir(exist_ok=True)
    report_path=directory/f'{name}.json';motion_path=directory/f'{name}.npz'
    if report_path.exists() and motion_path.exists():
        report=json.loads(report_path.read_text())
        if report.get('fingerprint')==job['fingerprint'] and report.get('output_sha256')==digest(motion_path):
            return dict(name=name,status=report['status'],frames=report['frames'],reused=True,
                        bytes=motion_path.stat().st_size,seconds=0.)
    work=out/'_work'/name;work.mkdir(parents=True,exist_ok=True)
    state_path=work/'checkpoint.json';poses_path=work/'qpos.npy'
    model=RUNTIME['model'];profile=RUNTIME['profile'];fps=PROFILE['fps'];substeps=PROFILE['substeps']
    source_path=ROOT/'src/humanoid_retarget/datasets/lafan1'/job['source']['path']
    if digest(source_path)!=job['source']['sha256']:raise ValueError(f'Source hash mismatch: {name}')
    duration=np.floor((job['source']['frames']-1)/job['source']['fps']*fps+1e-8)/fps
    if job.get('max_frames') is not None:duration=min(duration,(job['max_frames']-1)/fps)
    ref=opt.references(source_path,0.,duration,fps,model,RUNTIME['extract'],RUNTIME['utils'])
    count=len(ref['times']);solver=solver_for(model,profile)
    frame=0;resumed=False
    if state_path.exists() and poses_path.exists():
        state=json.loads(state_path.read_text())
        if state.get('fingerprint')!=job['fingerprint']:raise ValueError(f'Checkpoint fingerprint mismatch: {name}')
        q=np.lib.format.open_memmap(poses_path,mode='r+')
        if q.shape!=(count,model.nq):raise ValueError('Checkpoint trajectory shape mismatch')
        frame=state['completed_frame'];resumed=True
        for key,value in state['solver_counters'].items():setattr(solver,key,value)
    else:
        q=np.lib.format.open_memmap(poses_path,mode='w+',dtype=np.float64,shape=(count,model.nq))
        q[0]=solver.initialize_height_contact(ref)
    counters=lambda:{key:getattr(solver,key) for key in ['qp_failures','rejected_steps','backtracked_steps','minimum_step_fraction']}
    def checkpoint(completed):
        q.flush();atomic_json(state_path,dict(fingerprint=job['fingerprint'],completed_frame=completed,
            total_frames=count,solver_counters=counters(),elapsed_seconds=time.perf_counter()-started,
            stage='retargeting' if completed<count-1 else 'validating'))
    checkpoint(frame)
    for first in range(frame,count-1):
        pose=np.asarray(q[first]).copy()
        for sub in range(substeps):
            pose=solver.step(pose,**opt.target_at_step(ref,first,sub,substeps,True))
        q[first+1]=pose
        if (first+1)%500==0:checkpoint(first+1)
    checkpoint(count-1)
    report,series=opt.measure(model,q,ref,fps,solver.pairs)
    margin=solver.joint_margin;low=model.jnt_range[1:,0];high=model.jnt_range[1:,1]
    near=(q[:,7:]<low+margin)|(q[:,7:]>high-margin)
    report.update(name=name,frames=count,fps=fps,duration_seconds=(count-1)/fps,
        source=job['source']['path'],source_sha256=job['source']['sha256'],source_start_seconds=0.,
        source_fps=job['source']['fps'],source_frames=job['source']['frames'],
        trailing_source_time_not_on_output_grid_seconds=(job['source']['frames']-1)/job['source']['fps']-(count-1)/fps,
        parameters=PROFILE,fingerprint=job['fingerprint'],profile_fingerprint=job['profile_fingerprint'],
        joint_limit_margin_rad=margin.tolist(),joint_limit_margin_fraction_of_joint_frames=float(near.mean()),
        self_clearance_target_m=solver.self_clearance,collision_pairs=len(solver.pairs),
        collision_coverage='all nonstructural original mesh-hull pairs',
        mesh_distance_method='legacy libccd; nonpositive results repaired only by original-vertex separating-plane certificates',
        structural_collision_exclusions=solver.structural_exclusions,
        frames_resumed_from=frame if resumed else 0,solver_counters=counters(),
        training_ready=False,dynamic_feasibility_verified=False,
        limitations=['Contact probabilities are inferred from source skeletal height/speed, not measured forces.',
                     'No torque/contact-force or momentum feasibility certificate is supplied.',
                     'Original source terrain/partners/props are not reconstructed.'])
    report['checks']=dict(original_joint_ranges=report['joint_limit_excess_rad']<1e-6,
        original_joint_speeds=report['joint_speed_to_limit_max']<1.001,
        interpolated_floor_clear=report['subframes']['minimum_floor_height_m']>-.001,
        interpolated_body_pairs_clear=report['subframes']['minimum_selected_self_gap_m']>.00005,
        interpolated_distance_queries_resolved=report['subframes']['unresolved_zero_distance_queries']==0)
    passed=all(report['checks'].values())
    report['status']='kinematic_candidate_needs_review' if passed else 'kinematic_candidate_needs_correction'
    temporary=motion_path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,qpos=np.asarray(q,dtype=np.float32),qvel=series['qvel'].astype(np.float32),
            frequency=np.array(fps),joint_names=np.array(['root']+[model.joint(i).name for i in range(1,model.njnt)]),
            split_points=np.array([0,count]),source_contact_confidence=ref['contact_confidence'].astype(np.float16),
            contact_body_names=np.array(ref['contact_body_names']),source_sha256=np.array(job['source']['sha256']),
            robot_sha256=np.array(job['robot_sha256']),profile_fingerprint=np.array(job['profile_fingerprint']),
            kinematic_geometry_checks_passed=np.array(passed),training_ready=np.array(False))
    temporary.replace(motion_path);report['output_sha256']=digest(motion_path)
    report['output_bytes']=motion_path.stat().st_size;report['wall_seconds']=time.perf_counter()-started
    atomic_json(report_path,report)
    del q
    poses_path.unlink();state_path.unlink();work.rmdir()
    return dict(name=name,status=report['status'],frames=count,reused=False,
                bytes=motion_path.stat().st_size,seconds=report['wall_seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--max-frames',type=int,help='Smoke tests only; omitted for full conversion')
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    if args.workers<1 or (args.max_frames is not None and args.max_frames<3):parser.error('Invalid worker/frame count')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    opt.source.prepare()
    manifest=json.loads((ROOT/'src/humanoid_retarget/datasets/lafan1/raw_manifest.json').read_text())
    names=args.names
    files=[x for x in manifest['files'] if not names or Path(x['path']).stem in names]
    if args.limit is not None:files=files[:args.limit]
    if not files:parser.error('No source files selected')
    sources=['retarget_humanoid_light_lafan_velocity.py','humanoid_light_collision.py',
        'retarget_humanoid_light_lafan.py','optimize_humanoid_light_cmu_landing.py',
        'validate_humanoid_light_cmu_landing.py','retarget_humanoid_light_lafan_full.py']
    hashes={name:digest(ROOT/'scripts'/name) for name in sources}
    robot=ROOT/'outputs/humanoid_light_conversion/humanoid_light.xml'
    controls=ROOT/'outputs/humanoid_light_retarget/light_profile.json'
    fingerprint=hashlib.sha256(json.dumps(dict(parameters=PROFILE,source_code=hashes,
        robot_sha256=digest(robot),control_profile_sha256=digest(controls),max_frames=args.max_frames),sort_keys=True).encode()).hexdigest()
    profile_path=out/'profile.json'
    if profile_path.exists() and json.loads(profile_path.read_text())['fingerprint']!=fingerprint:
        raise ValueError('Output profile differs; choose a new directory rather than mixing runs')
    atomic_json(profile_path,dict(parameters=PROFILE,fingerprint=fingerprint,source_code_sha256=hashes,
        robot_sha256=digest(robot),control_profile_sha256=digest(controls),workers=args.workers))
    provenance=out/'provenance';provenance.mkdir(exist_ok=True)
    for name in sources:shutil.copy2(ROOT/'scripts'/name,provenance/name)
    jobs=[]
    for source in files:
        name=Path(source['path']).stem
        job_fingerprint=hashlib.sha256((fingerprint+source['sha256']).encode()).hexdigest()
        jobs.append(dict(name=name,source=source,output=str(out),fingerprint=job_fingerprint,
            profile_fingerprint=fingerprint,robot_sha256=digest(robot),max_frames=args.max_frames))
    # Longest first reduces the tail after the other workers have finished.
    jobs.sort(key=lambda x:x['source']['frames'],reverse=True)
    total=sum(min(int((x['source']['frames']-1)/x['source']['fps']*50)+1,args.max_frames or 10**12) for x in jobs)
    started=time.perf_counter();results=[];errors=[]
    def publish():
        atomic_json(out/'manifest.json',dict(dataset='LAFAN1 original human BVH -> Humanoid Light',
            profile_fingerprint=fingerprint,expected_sequences=len(jobs),expected_frames=total,
            completed_sequences=len(results),completed_frames=sum(x['frames'] for x in results),
            failed_sequences=len(errors),results=sorted(results,key=lambda x:x['name']),errors=errors,
            elapsed_seconds=time.perf_counter()-started,training_ready=False,
            dynamic_feasibility_verified=False,run_complete=len(results)+len(errors)==len(jobs)))
    publish()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker) as pool:
        futures={pool.submit(convert_one,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:result=future.result();results.append(result);print(json.dumps(result),flush=True)
            except Exception as exc:
                import traceback
                error=dict(name=job['name'],error=str(exc),traceback=traceback.format_exc());errors.append(error)
                print('FAILED',json.dumps(error),flush=True)
            publish()
    print('FINISHED',json.dumps(dict(completed=len(results),failed=len(errors),frames=sum(x['frames'] for x in results))),flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
