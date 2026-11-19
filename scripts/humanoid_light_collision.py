"""Conservative AABB culling and structural exclusions for original mesh hulls."""
from __future__ import annotations

import mujoco
import numpy as np


class MeshCollisionIndex:
    def __init__(self,model,meshes,pairs,distance):
        self.model,self.meshes,self.pairs,self.distance=model,meshes,list(pairs),distance
        self.vertices=dict(meshes)
        self.geom_ids=np.array([g for g,_ in meshes])
        low=np.array([v.min(0) for _,v in meshes]);high=np.array([v.max(0) for _,v in meshes])
        self.local_centers=(low+high)/2;self.local_extents=(high-low)/2
        order={g:i for i,(g,_) in enumerate(meshes)}
        self.pair_indices=np.array([(order[a],order[b]) for a,b in self.pairs],dtype=int).reshape(-1,2)
        self.world=None
        self.local_normals=[]
        for gid,vertices in meshes:
            mesh=int(model.geom_dataid[gid]);start=int(model.mesh_faceadr[mesh]);count=int(model.mesh_facenum[mesh])
            triangles=vertices[model.mesh_face[start:start+count]]
            normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            lengths=np.linalg.norm(normals,axis=1);normals=normals[lengths>1e-12]/lengths[lengths>1e-12,None]
            self.local_normals.append(np.unique(np.round(normals,10),axis=0))

    def update(self,data):
        # Transform each mesh's enclosing local box, not every vertex. This
        # remains conservative under rotation and is much cheaper at 200 Hz.
        rotation=data.geom_xmat[self.geom_ids].reshape(-1,3,3)
        center=np.einsum('gij,gj->gi',rotation,self.local_centers)+data.geom_xpos[self.geom_ids]
        extent=np.einsum('gij,gj->gi',abs(rotation),self.local_extents)
        self.low,self.high=center-extent,center+extent
        self.exact={}
        self.separating_normals={}

    def checked_distance(self,data,ia,ib,cutoff):
        """Use legacy CCD; repair a nonpositive result only with a proof.

        Native CCD in MuJoCo 3.11 can report deep penetration even when the
        original hulls have a positive separating-plane gap. Legacy libccd
        avoids this reproduced failure. Neither backend's nonpositive result
        is discarded without an independent original-vertex certificate.
        """
        first,second=self.meshes[ia][0],self.meshes[ib][0]
        witness=np.empty(6);flags=self.model.opt.disableflags
        try:
            self.model.opt.disableflags=flags|int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
            value=mujoco.mj_geomDistance(self.model,data,first,second,cutoff,witness)
        finally:self.model.opt.disableflags=flags
        if value>0:return value,witness,False
        va=self.exact_world(data,ia)[0];vb=self.exact_world(data,ib)[0]
        _,native_witness,_=self.distance(self.model,data,first,second,self.vertices,cutoff)
        axes=[np.eye(3),self.local_normals[ia]@data.geom_xmat[first].reshape(3,3).T,
              self.local_normals[ib]@data.geom_xmat[second].reshape(3,3).T]
        for points in [witness,native_witness]:
            normal=points[3:]-points[:3];length=np.linalg.norm(normal)
            if length>1e-12:axes.append((normal/length)[None])
        axes=np.vstack(axes);pa=va@axes.T;pb=vb@axes.T
        forward=pb.min(0)-pa.max(0);backward=pa.min(0)-pb.max(0)
        gaps=np.maximum(forward,backward);index=int(np.argmax(gaps));gap=float(gaps[index])
        if gap>1e-6:
            normal=axes[index] if forward[index]>=backward[index] else -axes[index]
            a=va[int(np.argmax(va@normal))];b=vb[int(np.argmin(vb@normal))]
            self.separating_normals[(first,second)]=normal
            return gap,np.r_[a,b],True
        return value,witness,False

    def exact_world(self,data,index):
        if index not in self.exact:
            gid,vertices=self.meshes[index]
            world=vertices@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
            self.exact[index]=(world,world.min(0),world.max(0))
        return self.exact[index]

    def near_floor(self,data,cutoff=.03):
        for index in np.flatnonzero(self.low[:,2]<cutoff):
            gid,_=self.meshes[index]
            yield gid,self.exact_world(data,index)[0]

    def minimum_height(self,data):
        minimum=np.inf
        for index in np.argsort(self.low[:,2]):
            if self.low[index,2]>=minimum:break
            gid,vertices=self.meshes[index]
            minimum=min(minimum,float((vertices@data.geom_xmat[gid].reshape(3,3)[2]+data.geom_xpos[gid,2]).min()))
        return minimum

    def query(self,data,cutoff=.03):
        a,b=self.pair_indices.T
        gaps=np.maximum(np.maximum(self.low[a]-self.high[b],self.low[b]-self.high[a]),0.)
        lower=np.linalg.norm(gaps,axis=1)
        for index in np.flatnonzero(lower<cutoff):
            ia,ib=self.pair_indices[index]
            _,la,ha=self.exact_world(data,ia);_,lb,hb=self.exact_world(data,ib)
            exact_gap=np.maximum(np.maximum(la-hb,lb-ha),0.)
            if np.linalg.norm(exact_gap)>=cutoff:continue
            first,second=self.pairs[index]
            value,witness,certificate=self.checked_distance(data,ia,ib,cutoff)
            yield first,second,value,witness,certificate


def body_collision_pairs(model,meshes,distance):
    """All collider pairs except rigid/adjacent structural neighbors.

    A small number of non-parent ancestor pairs overlap in the original
    neutral model (ankle housings and the waist assembly). These are recorded
    explicitly. Unrelated bodies are never exempted based on a neutral pose.
    """
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    vertices=dict(meshes)
    world={g:v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g] for g,v in meshes}
    def ancestors(body):
        chain=[]
        while body:
            chain.append(body);body=int(model.body_parentid[body])
        return chain
    chains={int(model.geom_bodyid[g]):ancestors(int(model.geom_bodyid[g])) for g,_ in meshes}
    pairs,excluded=[],[]
    for i,(a,_) in enumerate(meshes):
        for b,_ in meshes[i+1:]:
            ba,bb=int(model.geom_bodyid[a]),int(model.geom_bodyid[b])
            reason=None;neutral_distance=None
            if model.body_weldid[ba]==model.body_weldid[bb]:reason='same rigid assembly'
            elif model.body_parentid[ba]==bb or model.body_parentid[bb]==ba:reason='directly joined bodies'
            else:
                chain=chains[bb] if ba in chains[bb] else chains[ba] if bb in chains[ba] else None
                ancestor=ba if ba in chains[bb] else bb
                if chain is not None and chain.index(ancestor)<=2:
                    separation=np.maximum(np.maximum(world[a].min(0)-world[b].max(0),
                                                       world[b].min(0)-world[a].max(0)),0.)
                    if np.linalg.norm(separation)<.003:
                        neutral_distance,_,_=distance(model,data,a,b,vertices,.01)
                        if neutral_distance<.003:reason='overlapping short ancestor joint housing in original neutral model'
            if reason:
                excluded.append(dict(first=model.body(ba).name,second=model.body(bb).name,
                    reason=reason,neutral_distance_m=None if neutral_distance is None else float(neutral_distance)))
            else:pairs.append((a,b))
    return pairs,excluded
