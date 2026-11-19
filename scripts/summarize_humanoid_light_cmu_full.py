"""Verify every full-CMU output and build a sequence-level quality index."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def summarize(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    if not manifest['run_complete'] or manifest['failed_sequences']:
        raise ValueError('The conversion is incomplete or has processing failures')
    provenance=directory/'provenance';profile=json.loads((directory/'profile.json').read_text())
    inventory=json.loads((provenance/'source_manifest.json').read_text())
    if digest(provenance/'source_manifest.json')!=profile['source_manifest_sha256']:
        raise ValueError('Source inventory checksum mismatch')
    for relative,expected in profile['source_code_sha256'].items():
        if digest(provenance/Path(relative).name)!=expected:raise ValueError(f'Code snapshot mismatch: {relative}')
    if digest(provenance/'control_profile.json')!=profile['calibration_sha256']:
        raise ValueError('Control calibration snapshot mismatch')
    calibration=json.loads((provenance/'control_profile.json').read_text())
    robot=ROOT/'outputs/humanoid_light_conversion/humanoid_light.xml'
    if digest(robot)!=profile['robot_sha256']:raise ValueError('Robot model changed during conversion')
    model=mujoco.MjModel.from_xml_path(str(robot.parent/'scene.xml'))
    joint_names=['root']+[model.joint(i).name for i in range(1,model.njnt)]
    speed_limits=np.array([calibration['velocity_limits_joint_coordinate_rad_s'][model.joint(i).name]
                           for i in range(1,model.njnt)])
    contacts=['left_ankle_roll_link','right_ankle_roll_link','left_wrist_link','right_wrist_link',
              'left_knee_link','right_knee_link','pelvis_link','torso_roll_link']
    sources={row['name']:row for row in inventory['motions']}
    results={row['name']:row for row in manifest['results']}
    if set(sources)!=set(results):raise ValueError('Input/output sequence sets differ')
    rows=[];skipped=[];failures=Counter();frames=0;total_bytes=0;sample_count=0
    max_quaternion_error=0.;checked_sources=set()
    for name,source in sorted(sources.items()):
        result=results[name]
        for field in ['asf','amc']:
            original=ROOT/source[field]
            if original not in checked_sources:
                if digest(original)!=source[field+'_sha256']:raise ValueError(f'Original source checksum mismatch: {name}/{field}')
                checked_sources.add(original)
        expected=int(np.floor((source['frames']-1)/source['fps']*50+1e-8))+1
        if result['frames']==0:
            report=json.loads((directory/'skipped'/f'{name}.json').read_text())
            if expected>=3 or report['status']!='source_too_short_for_velocity':raise ValueError(f'Unexpected skipped source: {name}')
            skipped.append(dict(name=name,source_frames=source['frames'],duration_seconds=source['duration_seconds'],reason=report['reason']))
            continue
        relative=result['motion'];path=directory/relative
        report=json.loads(path.with_suffix('.json').read_text())
        if digest(path)!=report['output_sha256']:raise ValueError(f'Output checksum mismatch: {name}')
        timing=source.get('timing_verified',True)
        if ('motions_unverified_timing' in path.parts)==timing:
            raise ValueError(f'Incorrect timing directory: {name}')
        with np.load(path,allow_pickle=False) as motion:
            q=motion['qpos'];v=motion['qvel'];confidence=motion['source_contact_confidence']
            assert q.shape==(expected,model.nq) and v.shape==(expected,model.nv),name
            assert confidence.shape==(expected,8) and motion['source_airborne'].shape==(expected,),name
            assert q.dtype==v.dtype==np.float32,name
            assert np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(confidence).all(),name
            assert confidence.min()>=0 and confidence.max()<=1,name
            assert float(motion['frequency'])==50. and float(motion['source_fps'])==source['fps'],name
            assert bool(motion['source_timing_verified'])==timing,name
            assert motion['joint_names'].tolist()==joint_names and motion['contact_body_names'].tolist()==contacts,name
            assert motion['split_points'].tolist()==[0,expected],name
            assert str(motion['source_asf_sha256'])==source['asf_sha256'],name
            assert str(motion['source_amc_sha256'])==source['amc_sha256'],name
            assert str(motion['robot_sha256'])==profile['robot_sha256'],name
            assert str(motion['profile_fingerprint'])==manifest['profile_fingerprint']==report['profile_fingerprint'],name
            assert not bool(motion['training_ready']) and not report['training_ready'],name
            assert bool(motion['kinematic_checks_passed'])==all(report['checks'].values()),name
            geometry_passed=all(ok for key,ok in report['checks'].items() if key!='source_timing_verified')
            assert bool(motion['kinematic_geometry_checks_passed'])==geometry_passed,name
            excess=max(0.,float((model.jnt_range[1:,0]-q[:,7:]).max()),float((q[:,7:]-model.jnt_range[1:,1]).max()))
            assert excess<2e-6,(name,excess)
            assert np.max(abs(v[:,6:])/speed_limits)<1.001,name
            error=float(abs(np.linalg.norm(q[:,3:7].astype(float),axis=1)-1).max())
            assert error<2e-6,(name,error)
            max_quaternion_error=max(max_quaternion_error,error)
            stationary=np.max(abs(np.diff(q.astype(float),axis=0)),axis=1)<1e-9
            edges=np.diff(np.r_[False,stationary,False].astype(int))
            lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
            longest=float(lengths.max()/50) if len(lengths) else 0.
        assert report['frames']==expected and report['fps']==50.,name
        assert report['collision_pairs']==375 and len(report['structural_collision_exclusions'])==31,name
        checks=report['checks'];failed=[key for key,ok in checks.items() if not ok];failures.update(failed)
        steps=(expected-1)*4;solver=report['solver_counters'];geometry=report['subframes']
        assert geometry['samples']==3*(expected-1),name
        sample_count+=geometry['samples'];frames+=expected;total_bytes+=path.stat().st_size
        rows.append(dict(name=name,description=source['description'],motion=relative,subject=source['subject'],
            source_timing_verified=timing,source_fps=source['fps'],frames=expected,duration_seconds=(expected-1)/50,
            bytes=path.stat().st_size,status=report['status'],failed_checks=';'.join(failed),
            joint_limit_excess_rad=report['joint_limit_excess_rad'],joint_speed_to_limit_max=report['joint_speed_to_limit_max'],
            minimum_interpolated_floor_height_m=geometry['minimum_floor_height_m'],
            minimum_interpolated_body_gap_m=geometry['minimum_selected_self_gap_m'],
            unresolved_distance_queries=geometry['unresolved_zero_distance_queries'],
            qp_failure_fraction=solver['qp_failures']/steps,rejected_step_fraction=solver['rejected_steps']/steps,
            backtracked_step_fraction=solver['backtracked_steps']/steps,longest_stationary_interval_seconds=longest,
            stationary_frame_fraction=float(stationary.mean()),
            median_pelvis_rotation_error_deg=report['pelvis_rotation_error_deg']['median'],
            median_hip_rotation_error_deg=report['hip_relative_rotation_error_deg']['median'],
            trunk_tilt_p95_error_deg=report['trunk_tilt_error_deg']['p95'],
            trunk_height_rmse_m=report['height_contact']['trunk_height_error_rmse_m'],
            source_local_pitch_integral_deg=report['source_pelvis_local_rotation_integral_deg'][1],
            achieved_local_pitch_integral_deg=report['achieved_pelvis_local_rotation_integral_deg'][1]))
    assert frames==manifest['completed_frames']==manifest['expected_frames']
    assert len(rows)==manifest['converted_sequences'] and len(skipped)==manifest['skipped_sequences']
    index=directory/'sequence_index.csv'
    with index.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(attempted_sequences=len(sources),converted_sequences=len(rows),skipped_sequences=skipped,
        original_source_subjects=len({source['subject'] for source in sources.values()}),
        verified_timing_sequences=sum(row['source_timing_verified'] for row in rows),
        assumed_120hz_timing_sequences=sum(not row['source_timing_verified'] for row in rows),
        frames=frames,frequency_hz=50.,total_candidate_motion_hours=sum(row['duration_seconds'] for row in rows)/3600,
        verified_timing_motion_hours=sum(row['duration_seconds'] for row in rows if row['source_timing_verified'])/3600,
        assumed_timing_motion_hours=sum(row['duration_seconds'] for row in rows if not row['source_timing_verified'])/3600,
        compressed_motion_bytes=total_bytes,status_counts=dict(Counter(row['status'] for row in rows)),
        failed_check_counts=dict(failures),check_pass_counts={key:len(rows)-failures[key] for key in checks},
        interpolated_samples_screened=sample_count,max_stored_quaternion_norm_error=max_quaternion_error,
        all_original_source_and_output_hashes_verified=True,all_code_and_calibration_snapshots_verified=True,
        all_archive_shapes_finite_values_and_original_joint_ranges_verified=True,
        collision_pairs_per_sequence=375,structural_pair_exclusions=31,body_clearance_target_m=.008,
        profile_fingerprint=profile['fingerprint'],verification_script_sha256=digest(Path(__file__)),
        training_ready=False,dynamic_feasibility_verified=False,
        notes=['Unverified timing exports are in motions_unverified_timing and explicitly assume 120 Hz.',
               'Contacts are inferred skeletal cues, not measured forces.',
               'Ground uses a per-clip foot-height quantile; hanging, handstand-only or prop-supported clips may need separate ground/context annotation.',
               'source_airborne is the legacy feet-above-floor heuristic; hands or other body parts may still support the motion.',
               'Quarter-frame geometry screening does not certify continuous collision clearance or physical tracking.',
               'Long stationary intervals may be true pauses or solver stalls; inspect individual reports.',
               'Reported local rotation integrals are diagnostics, not a complete-flip certificate.'])
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,nargs='?',default=ROOT/'outputs/humanoid_light_cmu_full')
    args=parser.parse_args();summarize(args.directory.resolve())


if __name__=='__main__':main()
