"""Make a compact CMU showcase from saved full-batch qpos, with raw humans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont

import retarget_humanoid_light_cmu_full as batch
import retarget_humanoid_light_lafan_catalog as renderer

ROOT=batch.ROOT
PREFERRED=[('Walking','07_01'),('Running','09_01'),('Modern dance','05_01'),
    ('Soccer kick','10_01'),('Getting up','140_01'),('Jumping','118_01'),
    ('Backflip','87_03'),('Backflip, second take','87_04'),('Cartwheels','87_05'),
    ('Backflip','88_01'),('Cartwheel into backflip','88_05'),
    ('Breakdance with flips','85_14'),('Flip / one-hand stand','89_03'),('Cartwheel','90_02')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=batch.OUT)
    parser.add_argument('--seconds',type=float,default=5.)
    args=parser.parse_args();directory=args.dataset.resolve();output=directory/'previews/showcase'
    clips=output/'clips';clips.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((directory/'manifest.json').read_text())
    if not manifest['run_complete'] or manifest['failed_sequences']:raise ValueError('Finish the conversion first')
    inventory=json.loads((directory/'provenance/source_manifest.json').read_text())
    profile=json.loads((directory/'profile.json').read_text())
    if batch.shared.digest(ROOT/'outputs/humanoid_light_conversion/humanoid_light.xml')!=profile['robot_sha256']:
        raise ValueError('Robot differs from the conversion model')
    if batch.shared.digest(ROOT/'outputs/humanoid_light_retarget/light_profile.json')!=profile['calibration_sha256']:
        raise ValueError('Calibration differs from the conversion')
    sources={row['name']:row for row in inventory['motions']}
    results={row['name']:row for row in manifest['results'] if row['frames']>0}
    chosen=[(label,name) for label,name in PREFERRED if name in results and sources[name].get('timing_verified',True)]
    # Add one actual catalog-labelled punching motion instead of guessing an ID.
    punching=next((name for name in sorted(results) if sources[name].get('timing_verified',True)
        and re.search(r'boxing|punch',sources[name]['description'],re.I)),None)
    if punching:chosen.insert(4,('Punching / boxing',punching))
    batch.shared.initialize_worker();model=batch.shared.RUNTIME['model'];rendered=[]
    for label,name in chosen:
        source=sources[name];path=directory/results[name]['motion']
        report=json.loads(path.with_suffix('.json').read_text())
        if batch.shared.digest(path)!=report['output_sha256']:raise ValueError(f'Output checksum mismatch: {name}')
        motion=batch.adapter.cmu.load_asfamc(ROOT/source['asf'],ROOT/source['amc'],
            ROOT/source['c3d_header'] if source.get('c3d_header') else None,fps=source['fps'])
        reference=batch.adapter.references(motion,model)
        with np.load(path,allow_pickle=False) as archive:poses=archive['qpos']
        count=min(len(poses),round(args.seconds*50)+1)
        activity=np.linalg.norm(reference['linear_velocity'],axis=-1).mean(1)+.25*np.linalg.norm(reference['angular_velocity'],axis=-1)
        first=int(np.argmax(np.convolve(activity,np.ones(count)/count,mode='valid')))
        view=dict(names=reference['names'],human=reference['human'][first:first+count],
                  source_floor_z_m=reference['source_floor_z_m'])
        renderer.DESCRIPTIONS[name]=f'{label} ({name})'
        video=clips/f'{name}.mp4'
        renderer.render_pair(model,poses[first:first+count],view,video,name,first/50,
            candidate_title='Light: latest full CMU conversion',source_label='Original CMU',
            camera_azimuth=75. if label.startswith('Backflip') else 45.)
        duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration',
            '-of','default=noprint_wrappers=1:nokey=1',str(video)],text=True).strip())
        rendered.append(dict(label=label,name=name,description=source['description'],source_start_seconds=first/50,
            export_first_frame=first,export_frames=count,video=str(video.relative_to(output)),
            video_duration_seconds=duration,motion_sha256=report['output_sha256'],
            profile_fingerprint=report['profile_fingerprint'],checks=report['checks']))
        (output/'selection.json').write_text(json.dumps(rendered,indent=2)+'\n')
        print('RENDERED',len(rendered),len(chosen),name,label,flush=True)
    playlist=output/'playlist.txt';playlist.write_text(''.join("file '"+row['video']+"'\n" for row in rendered))
    metadata=[';FFMETADATA1','title=CMU / Humanoid Light - full conversion showcase'];position=0
    for row in rendered:
        end=position+round(row['video_duration_seconds']*1000)
        metadata.extend(['[CHAPTER]','TIMEBASE=1/1000',f'START={position}',f'END={end}','title='+row['label']+' '+row['name']]);position=end
    chapters=output/'chapters.ffmetadata';chapters.write_text('\n'.join(metadata)+'\n')
    combined=output/'cmu_showcase.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','concat','-safe','0','-i',str(playlist),'-f','ffmetadata',
        '-i',str(chapters),'-map_metadata','1','-map_chapters','1','-c','copy','-movflags','+faststart','-y',str(combined)],check=True)
    canvas=Image.new('RGB',(1680,280*((len(rendered)+2)//3)),'#e9eef2');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    for i,row in enumerate(rendered):
        with imageio.get_reader(output/row['video']) as reader:
            frame=reader.get_data(max(0,int(row['video_duration_seconds']*25/2)))
        x,y=i%3*560,i//3*280
        canvas.paste(Image.fromarray(frame).resize((560,256)),(x,y+24))
        draw.text((x+6,y+2),f'{i+1:02d} {row["label"]} {row["name"]}',font=font,fill='#17242d')
    canvas.save(output/'contact_sheet.png')
    (output/'summary.json').write_text(json.dumps(dict(video=combined.name,sequences=len(rendered),
        duration_seconds=position/1000,bytes=combined.stat().st_size,source_timing_verified=True),indent=2)+'\n')
    print('FINISHED',combined,flush=True)


if __name__=='__main__':main()
