"""Full CMU ASF/AMC conversion using the accepted LAFAN velocity profile.

Resumable CPU jobs, compact 50 Hz exports and explicit geometric checks.
Source timing is supplied by the official CMU index or a real C3D header.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

import retarget_humanoid_light_lafan_full as shared
import humanoid_light_cmu_velocity_reference as adapter

ROOT=shared.ROOT
OUT=ROOT/'outputs/humanoid_light_cmu_full'
PROFILE=shared.PROFILE


def convert_one(job):
    began=time.perf_counter();name=job['name']
    directory=Path(job['output'])/('motions' if job['source'].get('timing_verified',True) else 'motions_unverified_timing')
    directory.mkdir(exist_ok=True)
    motion_path=directory/f'{name}.npz';report_path=motion_path.with_suffix('.json')
    if motion_path.exists() and report_path.exists():
        old=json.loads(report_path.read_text())
        if old.get('fingerprint')==job['fingerprint'] and old.get('output_sha256')==shared.digest(motion_path):
            return dict(name=name,frames=old['frames'],status=old['status'],bytes=motion_path.stat().st_size,reused=True,seconds=0.,
                        motion=str(motion_path.relative_to(job['output'])))
    source=job['source'];asf=ROOT/source['asf'];amc=ROOT/source['amc']
    if shared.digest(asf)!=source['asf_sha256'] or shared.digest(amc)!=source['amc_sha256']:
        raise ValueError('Original source checksum mismatch')
    header=ROOT/source['c3d_header'] if source.get('c3d_header') else None
    if header is not None and source.get('c3d_header_sha256') and shared.digest(header)!=source['c3d_header_sha256']:
        raise ValueError('Original C3D timing header checksum mismatch')
    motion=adapter.cmu.load_asfamc(asf,amc,header,fps=source['fps'])
    expected=int(np.floor((len(motion.frame_ids)-1)/motion.fps*PROFILE['fps']+1e-8))+1
    if expected<3:
        skipped=Path(job['output'])/'skipped';skipped.mkdir(exist_ok=True)
        shared.atomic_json(skipped/f'{name}.json',dict(name=name,source=source,
            fingerprint=job['fingerprint'],profile_fingerprint=job['profile_fingerprint'],
            status='source_too_short_for_velocity',converted=False,training_ready=False,
            reason='Fewer than three samples on the 50 Hz grid; no motion or duration was fabricated'))
        return dict(name=name,frames=0,status='source_too_short_for_velocity',bytes=0,reused=False,
                    seconds=time.perf_counter()-began,motion=None)
    source_audit=adapter.cmu.check_source_fk(motion)
    source_audit['source_fps_hz']=source_audit.pop('source_fps_from_c3d_header')
    source_audit['timing_evidence']=source['timing_evidence']
    model=shared.RUNTIME['model'];fps=PROFILE['fps'];substeps=PROFILE['substeps']
    ref=adapter.references(motion,model,fps);count=len(ref['times'])
    solver=shared.solver_for(model,shared.RUNTIME['profile'])
    work=Path(job['output'])/'_work'/name;work.mkdir(parents=True,exist_ok=True)
    poses_path=work/'qpos.npy';state_path=work/'checkpoint.json'
    frame=0
    if poses_path.exists() and state_path.exists():
        state=json.loads(state_path.read_text())
        if state['fingerprint']!=job['fingerprint']:raise ValueError('Checkpoint profile mismatch')
        q=np.lib.format.open_memmap(poses_path,mode='r+')
        if q.shape!=(count,model.nq):raise ValueError('Checkpoint shape mismatch')
        frame=state['completed_frame']
        for key,value in state['solver_counters'].items():setattr(solver,key,value)
    else:
        q=np.lib.format.open_memmap(poses_path,mode='w+',dtype=np.float64,shape=(count,model.nq))
        q[0]=solver.initialize_height_contact(ref)
    counters=lambda:{key:getattr(solver,key) for key in ['qp_failures','rejected_steps','backtracked_steps','minimum_step_fraction']}
    def checkpoint(completed):
        q.flush();shared.atomic_json(state_path,dict(fingerprint=job['fingerprint'],completed_frame=completed,
            total_frames=count,solver_counters=counters(),elapsed_seconds=time.perf_counter()-began,
            stage='validating' if completed==count-1 else 'retargeting'))
    checkpoint(frame)
    for first in range(frame,count-1):
        pose=np.asarray(q[first]).copy()
        for sub in range(substeps):pose=solver.step(pose,**shared.opt.target_at_step(ref,first,sub,substeps,True))
        q[first+1]=pose
        if (first+1)%500==0:checkpoint(first+1)
    checkpoint(count-1)
    report,series=shared.opt.measure(model,q,ref,fps,solver.pairs)
    actual_pelvis=series['actual_pelvis_rotation']
    integral=lambda rotations:np.rad2deg(R.from_matrix(np.swapaxes(rotations[:-1],-1,-2)@rotations[1:]).as_rotvec().sum(0)).tolist()
    margins=solver.joint_margin;low=model.jnt_range[1:,0];high=model.jnt_range[1:,1]
    report.update(name=name,description=source['description'],source=source,source_validation=source_audit,
        frames=count,fps=fps,duration_seconds=(count-1)/fps,source_frames=len(motion.frame_ids),
        source_fps=motion.fps,source_start_seconds=0.,scale=ref['scale'],parameters=PROFILE,
        fingerprint=job['fingerprint'],profile_fingerprint=job['profile_fingerprint'],
        robot_sha256=job['robot_sha256'],frames_resumed_from=frame,solver_counters=counters(),
        joint_limit_margin_rad=margins.tolist(),
        joint_limit_margin_fraction_of_joint_frames=float(((q[:,7:]<low+margins)|(q[:,7:]>high-margins)).mean()),
        self_clearance_target_m=solver.self_clearance,collision_pairs=len(solver.pairs),
        structural_collision_exclusions=solver.structural_exclusions,
        mesh_distance_method='legacy libccd with original-vertex separating-plane certificates',
        height_calibration=ref['height_calibration'],neutral_source_trunk_height_m=ref['neutral_source_trunk_height_m'],
        trunk_orientation_method=ref['trunk_orientation_method'],
        trunk_height_local_offset_m=ref['trunk_height_local_offset_m'].tolist(),
        source_pelvis_local_rotation_integral_deg=integral(ref['pelvis_rotation']),
        achieved_pelvis_local_rotation_integral_deg=integral(actual_pelvis),
        source_inverted_pelvis_frames=int((ref['pelvis_rotation'][:,2,2]<-.5).sum()),
        achieved_inverted_pelvis_frames=int((actual_pelvis[:,2,2]<-.5).sum()),
        training_ready=False,dynamic_feasibility_verified=False,
        limitations=['Contact cues are inferred from skeletal height/speed, not measured forces.',
                     'No torque, contact-force or momentum feasibility certificate.',
                     'Original terrain, objects and interaction partners are not reconstructed.'])
    report['checks']=dict(original_joint_ranges=report['joint_limit_excess_rad']<1e-6,
        original_joint_speeds=report['joint_speed_to_limit_max']<1.001,
        interpolated_floor_clear=report['subframes']['minimum_floor_height_m']>-.001,
        interpolated_body_pairs_clear=report['subframes']['minimum_selected_self_gap_m']>.00005,
        interpolated_distance_queries_resolved=report['subframes']['unresolved_zero_distance_queries']==0)
    geometry_passed=all(report['checks'].values())
    report['checks']['source_timing_verified']=source.get('timing_verified',True)
    passed=all(report['checks'].values())
    report['status']='kinematic_candidate_needs_review' if passed else 'kinematic_candidate_needs_correction'
    temporary=motion_path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,qpos=np.asarray(q,dtype=np.float32),qvel=series['qvel'].astype(np.float32),
            frequency=np.array(fps),joint_names=np.array(['root']+[model.joint(i).name for i in range(1,model.njnt)]),
            split_points=np.array([0,count]),source_contact_confidence=ref['contact_confidence'].astype(np.float16),
            contact_body_names=np.array(ref['contact_body_names']),source_airborne=ref['source_airborne'],
            source_asf_sha256=np.array(source['asf_sha256']),source_amc_sha256=np.array(source['amc_sha256']),
            source_fps=np.array(source['fps']),source_timing_verified=np.array(source.get('timing_verified',True)),
            robot_sha256=np.array(job['robot_sha256']),profile_fingerprint=np.array(job['profile_fingerprint']),
            kinematic_geometry_checks_passed=np.array(geometry_passed),kinematic_checks_passed=np.array(passed),training_ready=np.array(False))
    temporary.replace(motion_path)
    report['output_sha256']=shared.digest(motion_path);report['output_bytes']=motion_path.stat().st_size
    report['wall_seconds']=time.perf_counter()-began;shared.atomic_json(report_path,report)
    if job.get('render'):
        import retarget_humanoid_light_lafan_catalog as renderer
        renderer.DESCRIPTIONS[name]=source['description']
        renderer.render_pair(model,q,ref,directory/f'{name}.mp4',name,0.,
            candidate_title='Light: velocity features + limits + body clearance',source_label='Original CMU')
    del q
    poses_path.unlink();state_path.unlink();work.rmdir()
    return dict(name=name,frames=count,status=report['status'],bytes=motion_path.stat().st_size,reused=False,seconds=report['wall_seconds'],
                motion=str(motion_path.relative_to(job['output'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'src/humanoid_retarget/datasets/cmu/full/manifest.json')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--names',nargs='+')
    parser.add_argument('--render',action='store_true',help='Only for small previews; full corpus stores no full-length videos')
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    manifest=json.loads(args.manifest.read_text());sources=manifest['motions']
    if args.names:sources=[source for source in sources if source['name'] in args.names]
    if not sources:parser.error('No source motions')
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    files=[ROOT/'scripts'/name for name in ['retarget_humanoid_light_cmu_full.py','humanoid_light_cmu_velocity_reference.py',
        'retarget_humanoid_light_cmu.py','retarget_humanoid_light_lafan_full.py','retarget_humanoid_light_lafan_velocity.py',
        'retarget_humanoid_light_lafan.py','humanoid_light_collision.py','optimize_humanoid_light_cmu_landing.py',
        'validate_humanoid_light_cmu_landing.py']]+[ROOT/'src/humanoid_retarget/datasets/cmu/asfamc.py']
    hashes={str(path.relative_to(ROOT)):shared.digest(path) for path in files}
    robot_sha=shared.digest(ROOT/'outputs/humanoid_light_conversion/humanoid_light.xml')
    calibration=ROOT/'outputs/humanoid_light_retarget/light_profile.json'
    values=dict(parameters=PROFILE,source_code_sha256=hashes,robot_sha256=robot_sha,
        calibration_sha256=shared.digest(calibration))
    fingerprint=hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()
    values['source_manifest_sha256']=shared.digest(args.manifest)
    profile_path=output/'profile.json'
    if profile_path.exists() and json.loads(profile_path.read_text())['fingerprint']!=fingerprint:
        raise ValueError('Existing output uses another profile; choose a separate output directory')
    shared.atomic_json(profile_path,dict(values,fingerprint=fingerprint,workers=args.workers))
    provenance=output/'provenance';provenance.mkdir(exist_ok=True)
    for path in files:shutil.copy2(path,provenance/path.name)
    shutil.copy2(calibration,provenance/'control_profile.json');shutil.copy2(args.manifest,provenance/'source_manifest.json')
    jobs=[dict(name=source['name'],source=source,output=str(output),profile_fingerprint=fingerprint,
        fingerprint=hashlib.sha256((fingerprint+json.dumps(source,sort_keys=True)).encode()).hexdigest(),
        robot_sha256=robot_sha,render=args.render) for source in sources]
    jobs.sort(key=lambda item:(item['source']['frames']-1)/item['source']['fps'],reverse=True)
    counts=[int(np.floor((s['frames']-1)/s['fps']*PROFILE['fps']+1e-8))+1 for s in sources]
    expected=sum(count for count in counts if count>=3)
    results=[];errors=[];began=time.perf_counter()
    def publish():
        shared.atomic_json(output/'manifest.json',dict(dataset='CMU ASF/AMC -> Humanoid Light',
            profile_fingerprint=fingerprint,expected_sequences=len(jobs),expected_frames=expected,
            completed_sequences=len(results),completed_frames=sum(item['frames'] for item in results),
            converted_sequences=sum(item['frames']>0 for item in results),
            skipped_sequences=sum(item['frames']==0 for item in results),
            failed_sequences=len(errors),results=sorted(results,key=lambda row:row['name']),errors=errors,
            elapsed_seconds=time.perf_counter()-began,training_ready=False,dynamic_feasibility_verified=False,
            run_complete=len(results)+len(errors)==len(jobs)))
    publish()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=shared.initialize_worker) as pool:
        futures={pool.submit(convert_one,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:result=future.result();results.append(result);print(json.dumps(result),flush=True)
            except Exception as exc:
                import traceback
                error=dict(name=job['name'],error=str(exc),traceback=traceback.format_exc());errors.append(error)
                print('FAILED',json.dumps(error),flush=True)
            publish()
    print('FINISHED',json.dumps(dict(completed=len(results),failed=len(errors),frames=sum(r['frames'] for r in results))),flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
