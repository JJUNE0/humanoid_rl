"""Optimize a CMU backflip with explicit contact and terminal-pose targets.

Bounded sparse trajectory least squares: all poses are optimized together with
velocity/acceleration terms and fixed landing foot targets. Root orientation
is parameterized as a local SO(3) correction to a continuous full-turn prior.
The aerial COM prior is ballistic; this is still kinematic optimization, with
no actuator torque/contact-force or angular-momentum feasibility certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from scipy import sparse

import retarget_humanoid_light_cmu as cmu
import retarget_humanoid_light_lafan as ik

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/humanoid_light_cmu_landing"
sys.path.insert(0, str(ROOT / "src"))
from humanoid_retarget.datasets.cmu import load_asfamc


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def right_jacobian(v):
    angle = np.linalg.norm(v)
    k = skew(v)
    if angle < 1e-5: return np.eye(3)-.5*k+k@k/6
    return np.eye(3)-(1-np.cos(angle))/angle**2*k+(angle-np.sin(angle))/angle**3*(k@k)


def right_jacobian_inverse(v):
    angle = np.linalg.norm(v)
    k = skew(v)
    if angle < 1e-5: return np.eye(3)+.5*k+k@k/12
    return np.eye(3)+.5*k+(1-.5*angle/np.tan(.5*angle))/angle**2*(k@k)


def smoothstep(t):
    u = np.clip(t, 0., 1.)
    return u*u*(3-2*u)


def checked_mesh_distance(model,data,a,b,vertices,cutoff=.05):
    """Repair an inconsistent zero only with a vertex separation certificate.

    Some MuJoCo native CCD mesh queries return exactly zero with a nonzero
    witness segment (upstream issue 3383). Retry after a common translation,
    then prove separation on the ORIGINAL vertex sets before accepting a gap.
    A positive projection gap is a conservative lower bound, not a heuristic
    decision to ignore a contact. True/unresolved intersections remain flagged.
    """
    points=np.empty(6)
    distance=mujoco.mj_geomDistance(model,data,a,b,cutoff,points)
    if distance!=0.:return distance,points,False
    original_a,original_b=data.geom_xpos[a].copy(),data.geom_xpos[b].copy()
    va=vertices[a]@data.geom_xmat[a].reshape(3,3).T+original_a
    vb=vertices[b]@data.geom_xmat[b].reshape(3,3).T+original_b
    try:
        for shift in [[1e-7,0,0],[0,1e-7,0],[0,0,1e-7],[0,0,-1],[.1,.2,.3]]:
            data.geom_xpos[a]=original_a+shift;data.geom_xpos[b]=original_b+shift
            witness=np.empty(6)
            value=mujoco.mj_geomDistance(model,data,a,b,cutoff,witness)
            normal=witness[3:]-witness[:3]
            if value<=0 or np.linalg.norm(normal)<1e-10:continue
            normal/=np.linalg.norm(normal)
            ia,ib=int(np.argmax(va@normal)),int(np.argmin(vb@normal))
            gap=float((vb[ib]-va[ia])@normal)
            if gap>1e-6:return gap,np.r_[va[ia],vb[ib]],True
    finally:
        data.geom_xpos[a]=original_a;data.geom_xpos[b]=original_b
    return distance,points,False


def foot_geometry(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    meshes, feet = ik.geometry(model)
    ids = [model.body(side+"_ankle_roll_link").id for side in ["left", "right"]]
    pos, rot = data.xpos[ids].copy(), data.xmat[ids].reshape(2,3,3).copy()
    bottoms = []
    for i, side in enumerate(["left", "right"]):
        world = feet[side][1] @ rot[i].T+pos[i]
        bottoms.append(float(world[:,2].min()))
    root = model.qpos0.copy()
    root[2] -= min(bottoms)-.0007
    pos[:,2] -= min(bottoms)-.0007
    return meshes, feet, ids, root, pos, rot


def ground_pose(model, foot_pos, foot_rot, knee_hint, com_offset=.0, initial=None):
    """Solve an upright, two-foot stance with a feasible amount of knee bend."""
    data = mujoco.MjData(model)
    ids = [model.body(s+"_ankle_roll_link").id for s in ["left", "right"]]
    q = model.qpos0.copy() if initial is None else initial.copy()
    center = foot_pos.mean(axis=0)
    hint = np.zeros(model.nu)
    for side in ["left", "right"]:
        for suffix, value in [("hip_pitch", -.5*knee_hint), ("knee", knee_hint), ("ankle_pitch", -.5*knee_hint)]:
            hint[model.joint(side+"_"+suffix+"_joint").qposadr[0]-7] = value
    lower = np.r_[center[:2]-.4, .4, model.jnt_range[1:,0]+1e-5]
    upper = np.r_[center[:2]+.4, .9, model.jnt_range[1:,1]-1e-5]
    yaw = np.arctan2(foot_rot[0,1,2],foot_rot[0,0,2])
    # Foot body axes differ from the root axes: the ground reference is built
    # without rotating the original model's heading in this experiment.
    root_q = np.array([1.,0,0,0])
    def residual(x):
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = x[:3], root_q, x[3:]
        mujoco.mj_fwdPosition(model,data)
        feet_r = data.xmat[ids].reshape(2,3,3)
        return np.r_[600*(data.xpos[ids]-foot_pos).ravel(),
            200*R.from_matrix(np.swapaxes(foot_rot,-1,-2)@feet_r).as_rotvec().ravel(),
            2*(x[3:]-hint), 8*(x[2]-(.704-.09*knee_hint)),
            25*(data.subtree_com[1,:2]-center[:2]-[com_offset,0])]
    x0 = np.r_[q[:3],q[7:]]
    x0[:2] += center[:2]-foot_geometry(model)[4].mean(axis=0)[:2] if initial is None else 0.
    fit=least_squares(residual,np.clip(x0,lower+1e-8,upper-1e-8),bounds=(lower,upper),max_nfev=100,
                       ftol=1e-9,xtol=1e-9,gtol=1e-9)
    residual(fit.x)
    q[:3],q[3:7],q[7:] = fit.x[:3],root_q,fit.x[3:]
    return q


def poses_to_parameters(qpos, base_rotation):
    x = np.empty((len(qpos),32))
    x[:,:3],x[:,6:] = qpos[:,:3],qpos[:,7:]
    x[:,3:6] = R.from_matrix(np.swapaxes(base_rotation,-1,-2)@R.from_quat(qpos[:,3:7],scalar_first=True).as_matrix()).as_rotvec()
    return x


def parameters_to_poses(x, base_rotation):
    q=np.empty((len(x),33))
    q[:,:3],q[:,7:] = x[:,:3],x[:,6:]
    q[:,3:7]=R.from_matrix(base_rotation@R.from_rotvec(x[:,3:6]).as_matrix()).as_quat(scalar_first=True)
    for t in range(1,len(q)):
        if np.dot(q[t-1,3:7],q[t,3:7])<0:q[t,3:7]*=-1
    return q


def make_plan(model, times, root_source, rotation_source, human, human_names, old_flight):
    """One main jump, then continuous double support and recovery."""
    _, robot_feet, ids, standing, p0, r0 = foot_geometry(model)
    spans=ik.contact_intervals(old_flight)
    first,end=max(spans,key=lambda pair:pair[1]-pair[0])
    takeoff=float(times[max(0,first-2)])
    feet_source=np.stack([human[:,human_names.index(s+"Toe"),2] for s in ["Left","Right"]],axis=1)
    floor=np.quantile(feet_source,.02)
    touch=end
    while touch<len(times)-1 and feet_source[touch].min()>floor+.025:touch+=1
    touchdown=float(times[touch])
    center=np.mean(np.stack([human[touch,human_names.index(s+"Foot")] for s in ["Left","Right"]]),axis=0)
    # Preserve the measured horizontal displacement, using the source's fitted scale.
    scale=float(json_load(ROOT/"outputs/humanoid_light_cmu/88_01_report.json")["target_leg_scale"])
    landing_center=center[:2]*scale
    p1=p0.copy();p1[:,:2]+=landing_center-p0.mean(axis=0)[:2]
    q_touch=ground_pose(model,p1,r0,.9,.025)
    q_absorb=ground_pose(model,p1,r0,1.15,.0,initial=q_touch)
    q_stand=ground_pose(model,p1,r0,0.,.0,initial=q_touch)
    q_launch=ground_pose(model,p0,r0,.2,0.)
    q_crouch=ground_pose(model,p0,r0,.9,0.)
    q_reference=np.tile(standing,(len(times),1))
    for t,seconds in enumerate(times):
        if seconds<takeoff:
            if seconds<takeoff-.18:
                a,b,u=standing,q_crouch,smoothstep((seconds-(takeoff-.5))/.32)
            else:a,b,u=q_crouch,q_launch,smoothstep((seconds-(takeoff-.18))/.18)
        elif seconds>=touchdown:
            elapsed=seconds-touchdown
            if elapsed<.12:a,b,u=q_touch,q_absorb,smoothstep(elapsed/.12)
            else:a,b,u=q_absorb,q_stand,smoothstep((elapsed-.12)/.65)
        else:
            a,b,u=q_launch,q_touch,(seconds-takeoff)/(touchdown-takeoff)
        q_reference[t,:3]=(1-u)*a[:3]+u*b[:3]
        q_reference[t,7:]=(1-u)*a[7:]+u*b[7:]
    progress=smoothstep((times-(takeoff-.06))/(touchdown-(takeoff-.06)))
    rotation=R.from_rotvec(np.outer(progress,[0.,-2*np.pi,0.])).as_matrix()
    foot_p=np.tile(p0,(len(times),1,1));foot_p[times>=touchdown]=p1
    contacts=np.tile(((times<takeoff)|(times>=touchdown))[:,None],(1,2))
    data=mujoco.MjData(model)
    def com(q):
        data.qpos[:]=q;mujoco.mj_fwdPosition(model,data);return data.subtree_com[1].copy()
    c0,c1=com(q_launch),com(q_touch)
    duration=touchdown-takeoff
    velocity=(c1-c0)/duration;velocity[2]+=.5*9.81*duration
    com_target=np.empty((len(times),3))
    for t,seconds in enumerate(times):
        if takeoff<seconds<touchdown:
            tau=seconds-takeoff;com_target[t]=c0+velocity*tau+[0,0,-.5*9.81*tau*tau]
        else:com_target[t]=com(q_reference[t])
    bone_pairs=ik.bone_correspondences()
    source_directions=np.stack([human[:,human_names.index(h2)]-human[:,human_names.index(h1)] for _,h1,h2,_,_ in bone_pairs],axis=1)
    source_directions/=np.linalg.norm(source_directions,axis=-1,keepdims=True)
    remap=rotation@np.swapaxes(rotation_source,-1,-2)
    bone_target=np.einsum('tij,tbj->tbi',remap,source_directions)
    # Shift from human tuck to a reachable robot landing before first contact.
    for t,seconds in enumerate(times):
        data.qpos[:]=q_reference[t];mujoco.mj_kinematics(model,data)
        ground_dirs=[]
        for _,_,_,b1,b2 in bone_pairs:
            v=data.xpos[model.body(b2).id]-data.xpos[model.body(b1).id];ground_dirs.append(v/np.linalg.norm(v))
        blend=max(1-smoothstep((seconds-(takeoff-.18))/.18), smoothstep((seconds-(touchdown-.2))/.2))
        bone_target[t]=(1-blend)*bone_target[t]+blend*np.array(ground_dirs)
        bone_target[t]/=np.linalg.norm(bone_target[t],axis=-1,keepdims=True)
    plan={"clip":"88_01","source_touchdown_heuristic": "first low toe/MTP height on descent after longest flight span",
          "takeoff_seconds":takeoff,"touchdown_seconds":touchdown,"absorption_end_seconds":touchdown+.12,
          "recovery_end_seconds":touchdown+.77,"landing_foot_positions":p1.tolist(),
          "post_landing_contact":"continuous double support; source rebound replaced by absorption/recovery",
          "root_orientation":"continuous full-turn prior ending upright at touchdown; optimized local corrections",
          "airborne_com":"ballistic reference with g=9.81 between feasible endpoint COMs",
          "kinematic_only":True,"dynamic_feasibility_verified":False,"robot_limits_modified":False}
    return dict(times=times,root=q_reference[:,:3].copy(),rotation=rotation,joint_reference=q_reference[:,7:].copy(),
        foot_pos=foot_p,foot_rot=np.tile(r0,(len(times),1,1,1)),contacts=contacts,com=com_target,
        bone_target=bone_target,q_reference=q_reference,metadata=plan)


def json_load(path):
    return json.loads(Path(path).read_text())


def seed_trajectory(model, plan):
    """Fit bounded aerial limb poses away from the floor, without sticky IK."""
    data=mujoco.MjData(model)
    lo,hi=model.jnt_range[1:,0]+1e-5,model.jnt_range[1:,1]-1e-5
    pairs=[(model.body(a).id,model.body(b).id) for _,_,_,a,b in ik.bone_correspondences()]
    q=plan['q_reference'].copy()
    to,td=plan['metadata']['takeoff_seconds'],plan['metadata']['touchdown_seconds']
    previous=q[0,7:].copy()
    for t,seconds in enumerate(plan['times']):
        if not to<seconds<td:continue
        data.qpos[:3]=[0,0,2]
        data.qpos[3:7]=R.from_matrix(plan['rotation'][t]).as_quat(scalar_first=True)
        target=plan['bone_target'][t]
        def evaluate(theta,jacobian=False):
            data.qpos[7:]=theta;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
            residual=[];rows=[]
            for i,(a,b) in enumerate(pairs):
                v=data.xpos[b]-data.xpos[a];length=np.linalg.norm(v);unit=v/length
                residual.extend(4*(unit-target[i]))
                if jacobian:
                    ja,jb=np.empty((3,model.nv)),np.empty((3,model.nv))
                    mujoco.mj_jacBody(model,data,ja,None,a);mujoco.mj_jacBody(model,data,jb,None,b)
                    rows.extend(4*(np.eye(3)-np.outer(unit,unit))@(jb[:,6:]-ja[:,6:])/length)
            residual.extend(.2*(theta-previous))
            if jacobian:rows.extend(.2*np.eye(model.nu))
            return np.asarray(rows) if jacobian else np.asarray(residual)
        result=least_squares(evaluate,np.clip(previous,lo+1e-7,hi-1e-7),jac=lambda x:evaluate(x,True),
            bounds=(lo,hi),max_nfev=50,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        q[t,7:]=previous=result.x
        q[t,3:7]=data.qpos[3:7]
        data.qpos[:]=q[t];mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        q[t,:3]+=plan['com'][t]-data.subtree_com[1]
        plan['joint_reference'][t]=q[t,7:]
    plan['root']=q[:,:3].copy()
    return q


class TrajectoryProblem:
    def __init__(self,model,plan):
        self.model,self.plan=model,plan
        self.data=mujoco.MjData(model)
        self.n=len(plan['times']);self.dt=plan['times'][1]-plan['times'][0]
        self.meshes,self.foot_meshes,self.feet,_,p0,r0=foot_geometry(model)
        self.mesh_by_id=dict(self.meshes)
        self.bones=[(model.body(a).id,model.body(b).id) for _,_,_,a,b in ik.bone_correspondences()]
        self.pairs=ik.self_collision_pairs(model)
        points=[]
        for i,side in enumerate(['left','right']):
            world=self.foot_meshes[side][1]@r0[i].T+p0[i]
            points.extend(world[world[:,2]<world[:,2].min()+.004,:2])
        self.hull=ConvexHull(np.asarray(points)).equations
        self.center0=p0.mean(axis=0)[:2]
        self.speed=np.array([json_load(ik.OUTPUT/'light_profile.json')['velocity_limits_joint_coordinate_rad_s'][model.joint(i).name] for i in range(1,model.njnt)])
        n=self.n
        d2=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csr')
        self.smoothing=sparse.kron(d2,sparse.diags(np.r_[np.full(3,30.),np.full(3,5.),np.full(26,4.)]),format='csr')
        d1=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format='csr')
        self.difference=sparse.kron(d1,sparse.eye(32),format='csr')
        self.velocity_bounds=np.tile(np.r_[np.full(6,np.inf),self.speed*self.dt],n-1)
        self.cache_x=None
        self.calls=0;self.started=time.perf_counter()

    def evaluate(self,flat):
        if self.cache_x is not None and np.array_equal(flat,self.cache_x):return self.cache_r,self.cache_j
        x=flat.reshape(self.n,32);q=parameters_to_poses(x,self.plan['rotation'])
        residuals=[];blocks=[];geometry_rows=[];geo_blocks=[];geo_values=[];model,data=self.model,self.data
        row_offset=0
        for t in range(self.n):
            data.qpos[:]=q[t];mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
            chain=np.eye(32);chain[3:6,3:6]=right_jacobian(x[t,3:6])
            values=[];rows=[];cache={};frame_geo=[];frame_clearance=[]
            def add(value,jac,scale=1.):
                value=np.atleast_1d(value);jac=np.atleast_2d(jac)
                values.extend(scale*value)
                rows.extend(np.asarray(scale)[:,None]*jac if np.ndim(scale) else scale*jac)
            def body_jac(bid):
                if bid not in cache:
                    jp,jr=np.empty((3,32)),np.empty((3,32))
                    mujoco.mj_jacBody(model,data,jp,jr,bid)
                    cache[bid]=(jp@chain,jr@chain)
                return cache[bid]
            weights=self.plan.get('weights',{})
            add(x[t,:3]-self.plan['root'][t],np.eye(32)[:3],weights.get('root_position',2.))
            if 'rotation_target' in self.plan:
                target_delta=R.from_matrix(self.plan['rotation'][t].T@self.plan['rotation_target'][t]).as_rotvec()
            else:
                target_delta=R.from_matrix(self.plan['rotation'][t].T).as_rotvec() if self.plan['contacts'][t].any() else np.zeros(3)
            add(x[t,3:6]-target_delta,np.eye(32)[3:6],weights.get('root_rotation',12.))
            joint_weight=weights.get('joint_air',.5) if not self.plan['contacts'][t].any() else weights.get('joint_ground',2.)
            add(x[t,6:]-self.plan['joint_reference'][t],np.eye(32)[6:],joint_weight)
            for i,(a,b) in enumerate(self.bones):
                v=data.xpos[b]-data.xpos[a];length=np.linalg.norm(v);unit=v/length
                derivative=(np.eye(3)-np.outer(unit,unit))@(body_jac(b)[0]-body_jac(a)[0])/length
                add(unit-self.plan['bone_target'][t,i],derivative,weights.get('bone_direction',3.))
            for side,bid in enumerate(self.feet):
                contact=self.plan['contacts'][t,side]
                jp,jw=body_jac(bid)
                add(data.xpos[bid]-self.plan['foot_pos'][t,side],jp,weights.get('contact_position',800.) if contact else 0.)
                actual=data.xmat[bid].reshape(3,3)
                error=R.from_matrix(self.plan['foot_rot'][t,side].T@actual).as_rotvec()
                derivative=right_jacobian_inverse(error)@actual.T@jw
                add(error,derivative,weights.get('contact_orientation',250.) if contact else 0.)
            jc=np.empty((3,32));mujoco.mj_jacSubtreeCom(model,data,jc,1);jc=jc@chain
            com=data.subtree_com[1]
            com_weight=weights.get('com_air',60.) if not self.plan['contacts'][t].any() else weights.get('com_ground',10.)
            add(com-self.plan['com'][t],jc,com_weight)
            offset=self.plan['foot_pos'][t].mean(axis=0)[:2]-self.center0
            hull_value=self.hull[:,:2]@(com[:2]-offset)+self.hull[:,2]+.012
            active=(hull_value>0)&self.plan['contacts'][t].all()
            add(np.where(active,hull_value,0),np.where(active[:,None],self.hull[:,:2]@jc[:2],0),200.)
            for gid,vertices in self.meshes:
                world=vertices@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
                point=world[world[:,2].argmin()];height=point[2]-.0007
                if height<.02:
                    jp=np.empty((3,32));mujoco.mj_jac(model,data,jp,None,point,model.geom_bodyid[gid]);gradient=(jp@chain)[2]
                else:gradient=np.zeros(32)
                if height<.02:
                    frame_geo.append(gradient);frame_clearance.append(point[2]-.0005)
                if height<0:geometry_rows.append(row_offset+len(values))
                add(min(height,0),gradient if height<0 else np.zeros(32),2500.)
            for a,b in self.pairs:
                distance,witness,_=checked_mesh_distance(model,data,a,b,self.mesh_by_id)
                if distance<.02:
                    normal=witness[3:]-witness[:3];length=np.linalg.norm(normal)
                    if length>1e-10:
                        normal*=np.sign(distance if distance else 1.)/length
                        ja,jb=np.empty((3,32)),np.empty((3,32))
                        mujoco.mj_jac(model,data,ja,None,witness[:3],model.geom_bodyid[a]);mujoco.mj_jac(model,data,jb,None,witness[3:],model.geom_bodyid[b])
                        gradient=normal@(jb-ja)@chain
                    else:
                        ba,bb=model.geom_bodyid[a],model.geom_bodyid[b]
                        normal=data.xpos[bb]-data.xpos[ba];normal/=max(np.linalg.norm(normal),1e-9)
                        gradient=normal@(body_jac(bb)[0]-body_jac(ba)[0])
                else:gradient=np.zeros(32)
                if distance<.02:
                    frame_geo.append(gradient);frame_clearance.append(distance-.002)
                if distance<.0025:geometry_rows.append(row_offset+len(values))
                add(min(distance-.0025,0),gradient if distance<.0025 else np.zeros(32),500.)
            residuals.append(np.asarray(values));blocks.append(sparse.csr_matrix(np.asarray(rows)))
            geo_blocks.append(sparse.csr_matrix(np.asarray(frame_geo).reshape(-1,32)));geo_values.extend(frame_clearance)
            row_offset+=len(values)
        r0=np.concatenate(residuals);j0=sparse.block_diag(blocks,format='csr')
        smooth=self.smoothing@flat
        delta=self.difference@flat
        excess=np.maximum(abs(delta)-self.velocity_bounds,0)
        active=excess>0
        velocity_jac=sparse.diags(100*np.sign(delta)*active)@self.difference
        residual=np.r_[r0,smooth,100*excess]
        jac=sparse.vstack([j0,self.smoothing,velocity_jac],format='csr')
        self.cache_x=flat.copy();self.cache_r,self.cache_j=residual,jac
        self.active_geometry_rows=np.asarray(geometry_rows,dtype=int)
        self.geometry_jacobian=sparse.block_diag(geo_blocks,format='csc')
        self.geometry_clearance=np.asarray(geo_values)
        self.calls+=1
        if self.calls%10==0:print(f"TRAJECTORY evaluation {self.calls} cost={.5*residual@residual:.3f} elapsed={time.perf_counter()-self.started:.1f}s",flush=True)
        return residual,jac

    def fun(self,x):return self.evaluate(x)[0]
    def jac(self,x):return self.evaluate(x)[1]


def validate_jacobian(problem,x):
    r,j=problem.evaluate(x)
    random=np.random.default_rng(83)
    columns=np.r_[np.arange(32),random.integers(0,len(x),16)]
    errors=[]
    for column in columns:
        test=x.copy();test[column]+=1e-6
        numeric=(problem.fun(test)-r)/1e-6
        exact=j[:,int(column)].toarray().ravel()
        errors.append(np.linalg.norm(numeric-exact)/max(1.,np.linalg.norm(numeric)))
    report={'columns':len(columns),'relative_error_max':float(max(errors)), 'relative_error_p95':float(np.quantile(errors,.95))}
    if report['relative_error_p95']>.005:raise AssertionError(f"Trajectory Jacobian failed: {report}")
    print('JACOBIAN',json.dumps(report),flush=True)
    return report


def polish_constraints(problem,x,lower,upper,iterations):
    """Sparse SQP polishing with hard joint bounds and frame-speed limits."""
    sys.path.insert(0,str(OUT/'python_deps'))
    if not (OUT/'python_deps/osqp').is_dir():
        import subprocess
        subprocess.run(['uv','pip','install','--target',str(OUT/'python_deps'),'--no-deps','osqp==1.1.3','joblib==1.6.0'],check=True)
    import osqp
    identity=sparse.eye(len(x),format='csc')
    finite=np.flatnonzero(np.isfinite(problem.velocity_bounds))
    velocity=problem.difference[finite].tocsc()
    speed_bounds=problem.velocity_bounds[finite]*.9995
    history=[];best=None;best_cost=np.inf
    for step in range(iterations):
        residual,jacobian=problem.evaluate(x)
        hessian=sparse.triu(jacobian.T@jacobian+.2*identity,format='csc')
        gradient=np.asarray(jacobian.T@residual).ravel()
        trust=np.tile(np.r_[np.full(3,.01),np.full(29,.035)],problem.n)
        matrices=[identity,velocity]
        lows=[np.maximum(lower-x,-trust),-speed_bounds-velocity@x]
        highs=[np.minimum(upper-x,trust),speed_bounds-velocity@x]
        rows=problem.geometry_jacobian
        if rows.shape[0]:
            matrices.append(rows);lows.append(-problem.geometry_clearance+1e-7);highs.append(np.full(rows.shape[0],np.inf))
        solver=osqp.OSQP()
        solver.setup(P=hessian,q=gradient,A=sparse.vstack(matrices,format='csc'),l=np.concatenate(lows),u=np.concatenate(highs),
            eps_abs=1e-6,eps_rel=1e-6,max_iter=12000,polishing=True,verbose=False)
        result=solver.solve(raise_error=False)
        if result.info.status_val not in (1,2):
            history.append({'iteration':step,'status':result.info.status});break
        delta=result.x
        candidates=[]
        for fraction in [1.,.5,.25,.125]:
            candidate=np.clip(x+fraction*delta,lower+1e-10,upper-1e-10)
            if np.max(abs(velocity@candidate)-speed_bounds)>2e-6:continue
            next_residual=problem.fun(candidate)
            cost=float(.5*next_residual@next_residual)
            clearance=float(problem.geometry_clearance.min()) if len(problem.geometry_clearance) else .05
            candidates.append((cost,candidate,clearance,fraction))
        if not candidates:
            history.append({'iteration':step,'status':'no numerically speed-feasible QP step'});break
        cost,candidate,clearance,fraction=min(candidates,key=lambda value:value[0])
        if clearance>=-2e-5 and cost<best_cost:best=candidate.copy();best_cost=cost
        history.append({'iteration':step,'cost':cost,'minimum_geometry_clearance_m':clearance,'step_fraction':fraction,
            'max_step':float(abs(delta).max()),'geometry_rows':rows.shape[0],'qp_status':result.info.status})
        print('SQP',json.dumps(history[-1]),flush=True)
        x=candidate
        if abs(delta).max()<1e-5:break
    if best is not None:
        history.append({'selected_best_feasible_cost':best_cost});x=best
    return x,history


def optimize(model,plan,seed,max_nfev,polish_iterations=12,resume=None):
    problem=TrajectoryProblem(model,plan)
    x=poses_to_parameters(seed,plan['rotation']).ravel()
    low=np.tile(np.r_[[-2,-1,.25],np.full(3,-.7),model.jnt_range[1:,0]+1e-5],len(seed))
    high=np.tile(np.r_[[1,1,2.2],np.full(3,.7),model.jnt_range[1:,1]-1e-5],len(seed))
    x=np.clip(x,low+1e-7,high-1e-7)
    jacobian_report=validate_jacobian(problem,x) if resume is None else {'previous_seed_check':str(OUT/'solver_least_squares.json')}
    started=time.perf_counter()
    if resume is None:
        result=least_squares(problem.fun,x,jac=problem.jac,bounds=(low,high),tr_solver='lsmr',
            max_nfev=max_nfev,ftol=1e-6,xtol=1e-7,gtol=1e-5,x_scale='jac',
            tr_options={'atol':1e-6,'btol':1e-6,'maxiter':200},verbose=0)
        x=result.x
        ls=dict(success=bool(result.success),message=result.message,nfev=result.nfev,cost=float(result.cost),optimality=float(result.optimality))
    else:
        previous=np.load(resume,allow_pickle=False)['qpos']
        x=poses_to_parameters(previous,plan['rotation']).ravel()
        x=np.clip(x,low+1e-8,high-1e-8)
        ls=dict(resumed_from=str(resume))
    x,polish=polish_constraints(problem,x,low,high,polish_iterations)
    q=parameters_to_poses(x.reshape(-1,32),plan['rotation'])
    return q,problem,dict(least_squares=ls,sqp_polish=polish,wall_seconds=time.perf_counter()-started,jacobian_check=jacobian_report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-nfev',type=int,default=100)
    parser.add_argument('--no-render',action='store_true')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--polish-iterations',type=int,default=12)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    motion=load_asfamc(cmu.DATA/'raw/88.asf',cmu.DATA/'raw/88_01.amc',cmu.DATA/'raw/88_01.c3d.header')
    model=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/humanoid_light_conversion/scene.xml'))
    source_root,source_rotation,_,_,flight,human,names,times,scale=cmu.targets(motion,model,50.)
    plan=make_plan(model,times,source_root,source_rotation['pelvis_link'],human,names,flight)
    print('PHASES',json.dumps(plan['metadata']),flush=True)
    seed=seed_trajectory(model,plan)
    np.savez_compressed(OUT/'88_01_seed.npz',qpos=seed)
    (OUT/'phase_plan.json').write_text(json.dumps(plan['metadata'],indent=2)+'\n')
    q,problem,solver=optimize(model,plan,seed,args.max_nfev,args.polish_iterations,args.resume)
    v=ik.differentiate(model,q,50.)
    np.savez_compressed(OUT/'88_01_optimized.npz',qpos=q.astype(np.float32),qvel=v.astype(np.float32),
        frequency=np.asarray(50.),joint_names=np.asarray(['root']+[model.joint(i).name for i in range(1,model.njnt)]),
        split_points=np.array([0,len(q)]),source_contact=plan['contacts'],training_ready=np.asarray(False))
    (OUT/'solver.json').write_text(json.dumps(solver,indent=2)+'\n')
    print('SOLVER',json.dumps(solver),flush=True)
    if not args.no_render:ik.render(model,q,human,names,50.,OUT/'88_01_optimized_comparison.mp4',human_scale=scale)


if __name__=='__main__':main()
