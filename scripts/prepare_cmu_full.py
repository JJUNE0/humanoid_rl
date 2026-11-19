"""Inventory the official CMU ASF/AMC bulk archive and fill catalog gaps.

The official bulk ZIP is used instead of crawling all individual motions.
Only files missing from that archive and 512-byte timing headers without
catalog frame rates are requested separately. Original files remain intact.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import shutil
import sys
import time
from urllib.request import Request,urlopen
import zipfile
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from humanoid_retarget.datasets.cmu.asfamc import c3d_header

OUT=ROOT/'src/humanoid_retarget/datasets/cmu/full'
META=ROOT/'outputs/cmu_full_preparation'
BASE='http://mocap.cs.cmu.edu'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def get(url,path,header_only=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return
    for attempt in range(3):
        try:
            request=Request(url,headers={'Range':'bytes=0-511'} if header_only else {})
            with urlopen(request,timeout=30) as response:
                data=response.read(512) if header_only else response.read()
            if data.lstrip().startswith(b'<'):raise ValueError('HTML response instead of motion data')
            if header_only and (len(data)!=512 or data[1]!=0x50):raise ValueError('Invalid C3D header')
            temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(data);temporary.replace(path)
            return
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=OUT/'archives/allasfamc.zip')
    parser.add_argument('--index',type=Path,default=META/'official_index.json')
    parser.add_argument('--unverified-fps',type=float,
                        help='Explicit assumed FPS for archived trials lacking official timing; exports are flagged unverified')
    args=parser.parse_args();archive=args.archive.resolve()
    if not archive.exists():
        partial=archive.with_suffix('.zip.part')
        if partial.exists() and partial.stat().st_size==1083549385 and zipfile.is_zipfile(partial):partial.replace(archive)
        else:raise FileNotFoundError('Complete the official allasfamc.zip download first')
    raw=OUT/'raw';raw.mkdir(parents=True,exist_ok=True)
    metadata=OUT/'metadata';metadata.mkdir(exist_ok=True)
    official=json.loads(args.index.read_text())['trials'];byname={row['name']:row for row in official}
    shutil.copy2(args.index,metadata/'official_index.json')
    for name in ['faq.html','search.html','subjects.html','archive_headers.json']:
        if (META/name).exists():shutil.copy2(META/name,metadata/name)
    print('ARCHIVE',archive.stat().st_size,'bytes',flush=True)
    with zipfile.ZipFile(archive) as source:
        files=[info for info in source.infolist() if Path(info.filename).suffix.lower() in ['.asf','.amc']]
        needed=sum(info.file_size for info in files)
        if shutil.disk_usage(OUT).free<needed+2*1024**3:raise ValueError('Insufficient disk space for original sources')
        for info in files:
            parts=PurePosixPath(info.filename)
            if parts.is_absolute() or '..' in parts.parts:raise ValueError('Unsafe archive path')
            path=raw.joinpath(*parts.parts);path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                if path.stat().st_size!=info.file_size:raise ValueError(f'Existing source size differs: {path}')
                with path.open('rb') as stream:
                    crc=0
                    while blob:=stream.read(1024**2):crc=zlib.crc32(blob,crc)
                if crc!=info.CRC:raise ValueError(f'Existing source CRC differs: {path}')
                continue
            temporary=path.with_suffix(path.suffix+'.tmp')
            with source.open(info) as src,temporary.open('wb') as dst:shutil.copyfileobj(src,dst,1024**2)
            temporary.replace(path)
    amcs={path.stem:path for path in raw.rglob('*.amc')}
    asfs={path.stem:path for path in raw.rglob('*.asf')}
    missing=sorted(set(byname)-set(amcs));requests=[]
    for name in missing:
        row=byname[name];subject=f"{row['subject']:02}"
        directory=raw/'supplement'/subject
        if subject not in asfs:
            asfs[subject]=directory/f'{subject}.asf'
            requests.append((f'{BASE}/subjects/{subject}/{subject}.asf',asfs[subject],False))
        amcs[name]=directory/f'{name}.amc'
        requests.append((BASE+row['amc_path'],amcs[name],False))
    unlisted=sorted(set(amcs)-set(byname))
    headers={name:OUT/'headers'/f'{name}.c3d.header' for name in unlisted}
    for name in unlisted:
        subject=name.split('_')[0]
        if args.unverified_fps is None:
            requests.append((f'{BASE}/subjects/{subject}/{name}.c3d',headers[name],True))
    failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(get,*item):item for item in requests}
        for number,future in enumerate(as_completed(futures),1):
            item=futures[future]
            try:future.result()
            except Exception as exc:failures.append(dict(url=item[0],path=str(item[1].relative_to(ROOT)),error=str(exc)))
            if number%10==0 or number==len(futures):print('SUPPLEMENTS',number,len(futures),'errors',len(failures),flush=True)
    motions=[];unavailable=[]
    for name,path in sorted(amcs.items()):
        subject=name.split('_')[0];skeleton=asfs.get(subject)
        if not path.exists() or skeleton is None or not skeleton.exists():
            unavailable.append(dict(name=name,reason='Original ASF or AMC unavailable'));continue
        original=path.read_bytes();frames=re.findall(rb'^\s*(\d+)\s*$',original,re.M)
        if not frames:unavailable.append(dict(name=name,reason='No AMC frame records'));continue
        row=byname.get(name)
        if row is not None:
            fps=row['fps'];description=row['description'];evidence='CMU official all-motion index: framerate column'
            verified=True
        else:
            header=headers[name]
            if not header.exists():
                if args.unverified_fps is None:
                    unavailable.append(dict(name=name,reason='Missing verified timing; original AMC preserved'));continue
                fps=args.unverified_fps;verified=False;evidence='Explicit assumed FPS; no original timing found in official index, individual C3D files or bulk C3D indexes'
            else:
                timing=c3d_header(header)
                if timing['frames']!=len(frames):
                    unavailable.append(dict(name=name,reason='C3D and AMC frame counts disagree',amc_frames=len(frames),c3d=timing));continue
                fps=timing['fps'];verified=True;evidence='Original 512-byte C3D header'
            description='Unlisted trial in official bulk archive'
        entry=dict(name=name,subject=int(subject),description=description,frames=len(frames),fps=fps,
            duration_seconds=(len(frames)-1)/fps,asf=str(skeleton.relative_to(ROOT)),amc=str(path.relative_to(ROOT)),
            asf_sha256=digest(skeleton),amc_sha256=hashlib.sha256(original).hexdigest(),timing_evidence=evidence,
            timing_verified=verified)
        if row is None and headers[name].exists():
            entry.update(c3d_header=str(headers[name].relative_to(ROOT)),c3d_header_sha256=digest(headers[name]))
        motions.append(entry)
    manifest=dict(dataset='CMU Graphics Lab Motion Capture Database',official_archive_url=BASE+'/allasfamc.zip',
        archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,bulk_motion_count=len(amcs)-len(missing),
        official_index_count=len(official),union_motion_count=len(amcs),supplemental_amc_count=len(missing),
        original_bytes=sum(p.stat().st_size for p in list(amcs.values())+list(asfs.values()) if p.exists()),
        duration_seconds=sum(row['duration_seconds'] for row in motions),frame_count=sum(row['frames'] for row in motions),
        sequence_count=len(motions),motions=motions,unavailable=unavailable,download_errors=failures,
        verified_timing_sequences=sum(row['timing_verified'] for row in motions),
        assumed_timing_sequences=sum(not row['timing_verified'] for row in motions),
        official_index_sha256=digest(args.index),original_motion_bytes_untouched=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PREPARED',json.dumps({key:manifest[key] for key in ['sequence_count','union_motion_count','frame_count','duration_seconds','original_bytes']}),flush=True)
    print('UNAVAILABLE',json.dumps(unavailable),flush=True)


if __name__=='__main__':main()
