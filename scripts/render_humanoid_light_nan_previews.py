"""Render original retargeted motions next to captured random-control physics."""

import os
os.environ.setdefault("MUJOCO_GL","egl")
from pathlib import Path
import json
import hashlib
import subprocess

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/humanoid_light_nan_scan/videos"
CAPTURE=OUT/"physics_capture"
FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
LABEL={"nan":"NaN","divergence":"속도·가속도 폭주","finite":"관측 구간 내 유한","capacity_overflow":"제약 용량 초과"}


def main():
    selected=json.loads((OUT/"selection.json").read_text())
    results={r["key"]:r for r in map(json.loads,(CAPTURE/"results.jsonl").read_text().splitlines())}
    model=mujoco.MjModel.from_binary_path(str(CAPTURE/"model.mjb"))
    model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    model.vis.headlight.ambient[:]=.5;model.vis.headlight.diffuse[:]=.7
    for gid in np.flatnonzero(model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE):
        model.geom_size[gid,:2]=40.
        model.geom_rgba[gid]=[.84,.87,.9,1.]
    option=mujoco.MjvOption();option.geomgroup[:]=0;option.geomgroup[0]=1;option.geomgroup[2]=1
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=False
    camera=mujoco.MjvCamera();camera.distance=2.8;camera.azimuth=125.;camera.elevation=-13.
    data=mujoco.MjData(model)
    title=ImageFont.truetype(FONT,26);font=ImageFont.truetype(FONT,21);small=ImageFont.truetype(FONT,17)
    clips=[];metadata=[];screens=[]
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        for number,row in enumerate(selected,1):
            key=row["key"];name=Path(row["path"]).stem;start=row["frame"]
            current=results[key]
            with np.load(ROOT/row["path"],allow_pickle=False) as source:
                source_q=source["qpos"]
            with np.load(CAPTURE/f"trace_{key.replace(':','_')}.npz",allow_pickle=False) as trace:
                q=trace["qpos"];v=trace["qvel"];time=trace["time"]
            good=np.isfinite(q).all(1)&np.isfinite(v).all(1)&(np.abs(v[:,6:]).max(1)<1000)
            invalid=np.flatnonzero(~good)
            stop=int(invalid[0]) if len(invalid) else len(q)
            stop=max(1,stop)
            origin=q[0,:2].copy()
            source_origin=source_q[start,:2].copy()
            lead=min(50,start)
            first=start-lead
            count=min(lead+100,len(source_q)-first)
            path=OUT/f"{number:02d}_{name}_frame{start}.mp4"
            last_image=None
            with imageio.get_writer(path,fps=25,codec="libx264",quality=8,
                                    macro_block_size=1,ffmpeg_params=["-threads","2","-movflags","+faststart"]) as writer:
                for frame in range(count):
                    index=first+frame;elapsed=(index-start)/50
                    trace_index=max(0,int(np.searchsorted(time,max(elapsed,0.),side="right")-1))
                    frozen=trace_index>=stop
                    trace_index=min(trace_index,stop-1)
                    poses=[source_q[index].copy(),q[trace_index].copy()]
                    poses[0][:2]-=source_origin;poses[1][:2]-=origin
                    canvas=Image.new("RGB",(1280,640),"#eef2f6")
                    draw=ImageDraw.Draw(canvas)
                    draw.text((14,8),f"{number:02d}  {name}  /  frame {start}",font=title,fill="#15283a")
                    draw.text((654,12),f"이번 물리 재생: {LABEL[current['outcome']]}",font=font,
                              fill="#a5273b" if current["outcome"]!="finite" else "#24674b")
                    draw.text((14,57),"리타겟 원본 · 저장된 모션 재생",font=font,fill="#245f76")
                    draw.text((654,57),"5ms 물리 · 무작위 명령 (학습 정책 아님)",font=font,fill="#7a3147")
                    for col,pose in enumerate(poses):
                        data.qpos[:]=pose
                        mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_camlight(model,data)
                        camera.lookat[:]=[pose[0],pose[1],.65]
                        renderer.update_scene(data,camera=camera,scene_option=option)
                        canvas.paste(Image.fromarray(renderer.render()),(640*col,96))
                    draw=ImageDraw.Draw(canvas)
                    if elapsed<0:
                        draw.rounded_rectangle((670,510,1230,560),radius=7,fill="#32475c")
                        draw.text((686,520),"물리 실험 시작 전: 초기 자세",font=font,fill="white")
                    elif frozen:
                        draw.rounded_rectangle((670,500,1230,565),radius=7,fill="#94283e")
                        draw.text((684,505),"수치 폭주 직전의 유한 프레임에서 정지",font=font,fill="white")
                        draw.text((684,535),f"재생 정지 시각 {time[stop-1]:.3f}s",font=small,fill="white")
                    draw.rectangle((0,580,1280,586),fill="#cbd6df")
                    marker=int(1280*frame/max(count-1,1));draw.rectangle((0,580,marker,586),fill="#337c9b")
                    draw.text((14,592),f"원본 frame {index}  /  시작점 기준 {elapsed:+.2f}s",font=small,fill="#253d52")
                    draw.text((654,592),"원본과 물리 재생 모두 0.5배속",font=small,fill="#253d52")
                    draw.text((14,617),"왼쪽은 원본 데이터의 움직임, 오른쪽은 무작위 명령에 대한 반응입니다.",font=small,fill="#526878")
                    image=np.asarray(canvas);writer.append_data(image)
                    if frame==min(count-1,lead+40):last_image=canvas.copy()
            screens.append(last_image)
            clips.append(path)
            metadata.append(dict(name=name,source_path=row["path"],source_frame=start,
                                 source_first_frame=first,source_last_frame=first+count-1,
                                 source_sha256=hashlib.sha256((ROOT/row["path"]).read_bytes()).hexdigest(),
                                 scan_outcome=row["outcome"],recaptured_outcome=current["outcome"],
                                 recaptured_failure_time_s=current.get("failure_time_s"),
                                 playback_speed=.5,video=path.name,
                                 frozen_before_invalid_physics=bool(len(invalid)),
                                 note="Saved mocap qpos on left; recaptured random-control GPU physics on right. No learned policy."))
            print(json.dumps(metadata[-1]),flush=True)
    playlist=OUT/"video_playlist.txt"
    playlist.write_text("".join(f"file '{p.name}'\n" for p in clips))
    combined=OUT/"source_vs_random_physics.mp4"
    subprocess.run(["ffmpeg","-v","error","-f","concat","-safe","0","-i",str(playlist),
                    "-c","copy","-movflags","+faststart","-y",str(combined)],check=True)
    source_only=OUT/"suspect_motions_source.mp4"
    source_filter=("crop=640:640:0:0,drawbox=x=0:y=612:w=640:h=28:color=0xeef2f6:t=fill,"
                   f"drawtext=fontfile={FONT}:text='리타겟 원본 모션 / 0.5배속':x=14:y=616:fontsize=17:fontcolor=0x526878")
    subprocess.run(["ffmpeg","-v","error","-i",str(combined),"-vf",source_filter,
                    "-c:v","libx264","-threads","2","-crf","20","-pix_fmt","yuv420p",
                    "-movflags","+faststart","-y",str(source_only)],check=True)
    sheet=Image.new("RGB",(1280,640),"#eef2f6")
    for i,im in enumerate(screens):sheet.paste(im.resize((640,320)),((i%2)*640,(i//2)*320))
    sheet.save(OUT/"contact_sheet.jpg",quality=92)
    (OUT/"video_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+"\n")
    print("FINISHED",combined,source_only,flush=True)


if __name__=="__main__":main()
