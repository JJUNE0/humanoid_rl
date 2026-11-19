"""Render representative windows directly from the full converted dataset.

No IK or physics rollout is run. Source references are normalized from time
zero to match the full trajectory's heading, scale and origin exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import retarget_humanoid_light_lafan_catalog as catalog
import retarget_humanoid_light_lafan_full as batch

ROOT = batch.ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=batch.OUT)
    parser.add_argument('--selection', type=Path,
                        default=ROOT/'outputs/humanoid_light_lafan_hip_rotation/selection.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--theme', action='append', choices=catalog.ORDER)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = (args.output_dir or dataset/'previews').resolve()
    clips = output/'clips'
    clips.mkdir(parents=True, exist_ok=True)
    profile = json.loads((dataset/'profile.json').read_text())
    robot = ROOT/'outputs/humanoid_light_conversion/humanoid_light.xml'
    if batch.digest(robot) != profile['robot_sha256']:
        raise ValueError('Robot differs from the model used for conversion')
    if batch.digest(batch.opt.source.OUTPUT/'light_profile.json') != profile['control_profile_sha256']:
        raise ValueError('Human/robot calibration differs from the full conversion')
    selection = json.loads(args.selection.read_text())
    if args.theme:
        selection = [item for item in selection if item['theme'] in args.theme]
    batch.initialize_worker()
    model = batch.RUNTIME['model']
    rendered = []
    for item in selection:
        name, theme = item['source'], item['theme']
        motion = dataset/'motions'/f'{name}.npz'
        report = json.loads(motion.with_suffix('.json').read_text())
        if batch.digest(motion) != report['output_sha256']:
            raise ValueError(f'Motion checksum mismatch: {name}')
        source = ROOT/'src/humanoid_retarget/datasets/lafan1/raw'/f'{name}.bvh'
        if batch.digest(source) != report['source_sha256']:
            raise ValueError(f'Source checksum mismatch: {name}')
        with np.load(motion, allow_pickle=False) as archive:
            fps = float(archive['frequency'])
            poses = archive['qpos']
        if fps != 50.:
            raise ValueError('The preview renderer expects the 50 Hz export')
        first = round(item['start_seconds']*fps)
        count = round(item['duration_seconds']*fps)+1
        if first+count > len(poses):
            raise ValueError(f'Window exceeds exported motion: {name}')
        # Using references(start=window_start) would rebase the source heading
        # and XY origin; the exported robot instead started at source time 0.
        reference = batch.opt.references(source, 0., (len(poses)-1)/fps, fps, model,
                                         batch.RUNTIME['extract'], batch.RUNTIME['utils'])
        view_reference = dict(names=reference['names'],
            human=reference['human'][first:first+count],
            source_floor_z_m=reference['source_floor_z_m'])
        path = clips/f'{theme}.mp4'
        temporary = clips/f'{theme}.tmp.mp4'
        catalog.render_pair(model, poses[first:first+count], view_reference, temporary,
            theme, item['start_seconds'], candidate_title='Full export: joint limits + body clearance')
        temporary.replace(path)
        rendered.append(dict(theme=theme, description=item['description'], source=name,
            source_start_seconds=first/fps, duration_seconds=item['duration_seconds'],
            first_export_frame=first, export_frames=count,
            motion_sha256=report['output_sha256'], profile_fingerprint=report['profile_fingerprint'],
            video=str(path.relative_to(output)), video_sha256=batch.digest(path),
            whole_sequence_checks=report['checks']))
        print(json.dumps(dict(completed=len(rendered), total=len(selection), theme=theme)), flush=True)
        (output/'selection.json').write_text(json.dumps(rendered, indent=2)+'\n')

    playlist = output/'playlist.txt'
    playlist.write_text(''.join("file '"+row['video']+"'\n" for row in rendered))
    metadata = [';FFMETADATA1', 'title=LAFAN1 / Humanoid Light - full dataset preview']
    position = 0
    for row in rendered:
        end = position+round(row['duration_seconds']*1000)
        metadata.extend(['[CHAPTER]', 'TIMEBASE=1/1000', f'START={position}', f'END={end}',
                         'title='+row['description']])
        position = end
    chapters = output/'chapters.ffmetadata'
    chapters.write_text('\n'.join(metadata)+'\n')
    movie = output/'all_15_themes.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(playlist),
        '-f', 'ffmetadata', '-i', str(chapters), '-map_metadata', '1', '-map_chapters', '1',
        '-c', 'copy', '-movflags', '+faststart', '-y', str(movie)], check=True)
    canvas = Image.new('RGB', (1680, 280*((len(rendered)+2)//3)), '#e9eef2')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    for index, item in enumerate(rendered):
        with imageio.get_reader(output/item['video']) as reader:
            frame = reader.get_data(round(item['duration_seconds']*25/2))
        x, y = index%3*560, index//3*280
        canvas.paste(Image.fromarray(frame).resize((560, 256)), (x, y+24))
        draw.text((x+6, y+2), f'{index+1:02d} {item["theme"]}', font=font, fill='#17242d')
    canvas.save(output/'contact_sheet.png')
    (output/'summary.json').write_text(json.dumps(dict(video=movie.name,
        themes=len(rendered), duration_seconds=position/1000,
        profile_fingerprint=profile['fingerprint'], movie_sha256=batch.digest(movie),
        note='Original human versus saved full-dataset qpos; kinematic playback.'), indent=2)+'\n')
    print('FINISHED', str(movie), flush=True)


if __name__ == '__main__':
    main()
