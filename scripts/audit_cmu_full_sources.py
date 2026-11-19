"""Check every CMU source format and feature array before the long IK batch."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import numpy as np
import retarget_humanoid_light_lafan_full as shared
import humanoid_light_cmu_velocity_reference as adapter


def audit(source):
    try:
        root=shared.ROOT;header=root/source['c3d_header'] if source.get('c3d_header') else None
        motion=adapter.cmu.load_asfamc(root/source['asf'],root/source['amc'],header,fps=source['fps'])
        if len(motion.frame_ids)!=source['frames']:raise ValueError('Inventory/decoder frame count mismatch')
        expected=int(np.floor((len(motion.frame_ids)-1)/motion.fps*50+1e-8))+1
        if expected<3:
            return dict(name=source['name'],ok=True,frames=0,source_frames=len(motion.frame_ids),
                source_fps=motion.fps,status='source_too_short_for_velocity',
                reason='Fewer than three 50 Hz samples; original source preserved')
        reference=adapter.references(motion,shared.RUNTIME['model'])
        if len(reference['times'])!=expected:raise ValueError('Resampling shape mismatch')
        source_fk=adapter.cmu.check_source_fk(motion)
        return dict(name=source['name'],ok=True,frames=expected,source_frames=len(motion.frame_ids),
            source_fps=motion.fps,scale=reference['scale'],
            independent_fk_max_error_m=source_fk['max_endpoint_error_m'],
            max_pelvis_angular_speed_rad_s=float(np.linalg.norm(reference['pelvis_angular_velocity'],axis=-1).max()))
    except Exception as exc:
        import traceback
        return dict(name=source['name'],ok=False,error=str(exc),traceback=traceback.format_exc())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=shared.ROOT/'src/humanoid_retarget/datasets/cmu/full/manifest.json')
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--output',type=Path,default=shared.ROOT/'outputs/cmu_full_preparation/source_audit.json')
    args=parser.parse_args();sources=json.loads(args.manifest.read_text())['motions']
    results=[];began=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=shared.initialize_worker) as pool:
        futures=[pool.submit(audit,source) for source in sources]
        for future in as_completed(futures):
            row=future.result();results.append(row)
            if not row['ok']:print('FAILED',json.dumps(row),flush=True)
            if len(results)%100==0 or len(results)==len(sources):
                print('PROGRESS',len(results),len(sources),'failed',sum(not x['ok'] for x in results),flush=True)
                shared.atomic_json(args.output,dict(expected=len(sources),completed=len(results),
                    failures=[x for x in results if not x['ok']],results=sorted(results,key=lambda x:x['name']),
                    source_manifest_sha256=shared.digest(args.manifest),elapsed_seconds=time.perf_counter()-began))
    print('FINISHED',len(results),'failed',sum(not x['ok'] for x in results),flush=True)
    if any(not x['ok'] for x in results):raise SystemExit(1)


if __name__=='__main__':main()
