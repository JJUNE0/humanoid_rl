"""Verify compact full LAFAN exports and write a sequence-level quality index."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    if not manifest['run_complete'] or manifest['failed_sequences']:
        raise ValueError('Conversion is incomplete or has failed sequences')
    raw=json.loads((ROOT/'src/humanoid_retarget/datasets/lafan1/raw_manifest.json').read_text())['files']
    sources={Path(f['path']).stem:f for f in raw}
    results={r['name']:r for r in manifest['results']}
    if set(results)!=set(sources):raise ValueError('Full original source set differs from converted source set')
    profile=json.loads((directory/'profile.json').read_text())
    for name,expected in profile['source_code_sha256'].items():
        if sha256(directory/'provenance'/name)!=expected:raise ValueError(f'Code snapshot mismatch: {name}')
    calibration=directory/'provenance/control_profile.json'
    if sha256(calibration)!=profile['control_profile_sha256']:raise ValueError('Calibration snapshot mismatch')
    robot=ROOT/'outputs/humanoid_light_conversion/humanoid_light.xml'
    if sha256(robot)!=profile['robot_sha256']:raise ValueError('Robot model changed during conversion')
    model=mujoco.MjModel.from_xml_path(str(robot.parent/'scene.xml'))
    joint_names=['root']+[model.joint(i).name for i in range(1,model.njnt)]
    contacts=['left_ankle_roll_link','right_ankle_roll_link','left_wrist_link','right_wrist_link',
              'left_knee_link','right_knee_link','pelvis_link','torso_roll_link']
    rows=[];failures=Counter();total_bytes=0;max_quaternion_norm_error=0.;total_frames=0
    for name,source in sorted(sources.items()):
        path=directory/'motions'/f'{name}.npz';report=json.loads(path.with_suffix('.json').read_text())
        if sha256(path)!=report['output_sha256']:raise ValueError(f'Output checksum mismatch: {name}')
        original=ROOT/'src/humanoid_retarget/datasets/lafan1'/source['path']
        if sha256(original)!=source['sha256']:raise ValueError(f'Original source checksum mismatch: {name}')
        count=int(np.floor((source['frames']-1)/source['fps']*50+1e-8))+1
        with np.load(path,allow_pickle=False) as motion:
            q=motion['qpos'];v=motion['qvel'];confidence=motion['source_contact_confidence']
            assert q.shape==(count,model.nq),(name,q.shape,count)
            assert v.shape==(count,model.nv),(name,v.shape)
            assert confidence.shape==(count,8),(name,confidence.shape)
            assert q.dtype==np.float32 and v.dtype==np.float32
            assert np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(confidence).all(),name
            assert confidence.min()>=0 and confidence.max()<=1,name
            assert float(motion['frequency'])==50.,name
            assert motion['joint_names'].tolist()==joint_names,name
            assert motion['contact_body_names'].tolist()==contacts,name
            assert motion['split_points'].tolist()==[0,count],name
            assert not bool(motion['training_ready']),name
            assert bool(motion['kinematic_geometry_checks_passed'])==all(report['checks'].values()),name
            assert str(motion['source_sha256'])==source['sha256']==report['source_sha256'],name
            assert str(motion['robot_sha256'])==profile['robot_sha256'],name
            assert str(motion['profile_fingerprint'])==manifest['profile_fingerprint']==report['profile_fingerprint'],name
            assert report['frames']==count and report['fps']==50.,name
            quaternion_error=float(abs(np.linalg.norm(q[:,3:7].astype(float),axis=1)-1).max())
            assert quaternion_error<2e-6,(name,quaternion_error)
            max_quaternion_norm_error=max(max_quaternion_norm_error,quaternion_error)
            excess=max(0.,float((model.jnt_range[1:,0]-q[:,7:]).max()),float((q[:,7:]-model.jnt_range[1:,1]).max()))
            # Check stored float32 positions as well as the full precision solve.
            assert excess<2e-6,(name,excess)
            stationary=np.max(abs(np.diff(q.astype(float),axis=0)),axis=1)<1e-9
            edges=np.diff(np.r_[False,stationary,False].astype(int));lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
            longest=int(lengths.max()) if len(lengths) else 0
        failed=[key for key,ok in report['checks'].items() if not ok];failures.update(failed)
        solver=report['solver_counters'];steps=(count-1)*profile['parameters']['substeps'];geom=report['subframes'];height=report['height_contact']
        row=dict(name=name,motion_type=re.sub(r'\d+$','',name.split('_')[0]),frames=count,duration_seconds=(count-1)/50,
            bytes=path.stat().st_size,status=report['status'],failed_checks=';'.join(failed),
            joint_limit_excess_rad=report['joint_limit_excess_rad'],joint_speed_to_limit_max=report['joint_speed_to_limit_max'],
            minimum_interpolated_floor_height_m=geom['minimum_floor_height_m'],
            minimum_interpolated_body_gap_m=geom['minimum_selected_self_gap_m'],
            unresolved_distance_queries=geom['unresolved_zero_distance_queries'],
            qp_failure_fraction=solver['qp_failures']/steps,rejected_step_fraction=solver['rejected_steps']/steps,
            backtracked_step_fraction=solver['backtracked_steps']/steps,
            stationary_frame_fraction=float(stationary.mean()),longest_stationary_interval_seconds=longest/50,
            median_hip_rotation_error_deg=report['hip_relative_rotation_error_deg']['median'],
            median_pelvis_rotation_error_deg=report['pelvis_rotation_error_deg']['median'],
            trunk_height_rmse_m=height['trunk_height_error_rmse_m'],
            hover_during_inferred_contact_fraction=height['hover_during_inferred_contact_fraction'])
        rows.append(row);total_bytes+=path.stat().st_size;total_frames+=count
    assert total_frames==manifest['expected_frames']==manifest['completed_frames']
    index=directory/'sequence_index.csv'
    with index.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(sequences=len(rows),motion_types=dict(Counter(row['motion_type'] for row in rows)),
        frames=total_frames,frequency_hz=50.,motion_hours=sum(row['duration_seconds'] for row in rows)/3600,
        original_source_hours=sum((s['frames']-1)/s['fps'] for s in sources.values())/3600,
        compressed_motion_bytes=total_bytes,status_counts=dict(Counter(row['status'] for row in rows)),
        failed_check_counts=dict(failures),max_stored_quaternion_norm_error=max_quaternion_norm_error,
        check_pass_counts={key:len(rows)-failures[key] for key in report['checks']},
        all_77_original_source_hashes_verified=True,all_output_hashes_verified=True,
        all_source_code_and_calibration_snapshots_verified=True,
        all_archive_shapes_and_finite_values_verified=True,all_original_joint_ranges_verified=True,
        collision_pairs_per_sequence=375,structural_pair_exclusions=31,
        joint_limit_margin_max_degrees=5.,body_clearance_target_m=.008,
        high_stationary_sequences=[row['name'] for row in rows if row['longest_stationary_interval_seconds']>1.],
        sequence_index=str(index.relative_to(ROOT)),profile_fingerprint=manifest['profile_fingerprint'],
        training_ready=False,dynamic_feasibility_verified=False,
        notes=['Stored contacts are inferred skeletal cues, not measured support forces.',
               'Stationary intervals can be true pauses or solver stalls; inspect sequence reports.',
               'Unresolved mesh-distance queries remain failures, not certified collisions or clearances.',
               'Geometry screening samples quarter-frame interpolation, not continuous-time collision certification.',
               'All files are full-length candidates; geometry pass does not certify physical tracking.'])
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,nargs='?',default=ROOT/'outputs/humanoid_light_lafan_full')
    args=parser.parse_args();summarize(args.directory.resolve())


if __name__=='__main__':main()
