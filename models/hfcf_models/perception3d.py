"""Compact CPU 3D perception references, explicitly not upstream checkpoint replicas.

PointPillars: point decoration -> pillar pooling -> BEV scatter -> dense anchor heads.
CenterPoint: voxel pooling -> dense 3D convolution -> height compression -> center heads.
BEVFusion: calibrated camera lift/splat + LiDAR BEV -> fused center/map heads.
"""
import torch
from torch import nn
from torch.nn import functional as F


class BEVHeads(nn.Module):
    def __init__(self, channels, classes, center=True):
        super().__init__()
        self.center = center
        self.classes = nn.Conv2d(channels, classes, 1)
        self.boxes = nn.Conv2d(channels, 9 if center else 7, 1)
        self.direction = None if center else nn.Conv2d(channels,2,1)

    def forward(self,x):
        out = {'heatmap_logits' if self.center else 'class_logits':self.classes(x), 'box_regression':self.boxes(x)}
        if self.direction is not None:
            out['direction_logits'] = self.direction(x)
        return out


class Grid(nn.Module):
    def __init__(self, grid_size=16, extent=8.):
        super().__init__()
        if grid_size < 2 or extent <= 0:
            raise ValueError('grid_size >= 2 and extent > 0 are required')
        self.grid_size, self.extent = grid_size, extent

    def xy(self,points):
        return ((points[:,:2]+self.extent)/(2*self.extent)*self.grid_size).floor().long()

    def valid(self, points, indexes):
        return torch.isfinite(points).all(dim=1) & (indexes>=0).all(dim=1) & (indexes<self.grid_size).all(dim=1)


class PillarEncoder(Grid):
    def __init__(self, point_dim=4, hidden_dim=32, grid_size=16, extent=8., **_):
        super().__init__(grid_size,extent)
        self.point_dim = point_dim
        self.point_net = nn.Sequential(nn.Linear(point_dim+5,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU())
        self.channels = hidden_dim

    def forward(self, points):
        if points.ndim != 3 or points.shape[-1] != self.point_dim:
            raise ValueError('points must have shape B,N,point_dim')
        batches = []
        for cloud in points:
            xy = self.xy(cloud)
            valid = self.valid(cloud,xy)
            cloud,xy = cloud[valid],xy[valid]
            image = points.new_zeros((self.grid_size*self.grid_size,self.channels))
            if len(cloud):
                keys=xy[:,1]*self.grid_size+xy[:,0]
                unique,inverse=torch.unique(keys,return_inverse=True)
                sums=cloud.new_zeros((len(unique),3)).index_add(0,inverse,cloud[:,:3])
                counts=torch.bincount(inverse,minlength=len(unique)).to(cloud.dtype).unsqueeze(1)
                cluster_offset=cloud[:,:3]-sums[inverse]/counts[inverse]
                center=(xy.to(cloud.dtype)+.5)*(2*self.extent/self.grid_size)-self.extent
                decorated=torch.cat([cloud,cluster_offset,cloud[:,:2]-center],dim=-1)
                encoded=self.point_net(decorated)
                pooled=encoded.new_full((len(unique),self.channels),-torch.inf)
                pooled=pooled.scatter_reduce(0,inverse[:,None].expand_as(encoded),encoded,reduce='amax',include_self=True)
                image=image.index_copy(0,unique,pooled)
            batches.append(image.T.reshape(self.channels,self.grid_size,self.grid_size))
        return torch.stack(batches)


class PointPillars(nn.Module):
    """Compact pillar network with one anchor per cell; raw heads, no box decoder/NMS."""
    def __init__(self, point_dim=4,hidden_dim=32,classes=3,grid_size=16,extent=8.,**_):
        super().__init__()
        self.pillars=PillarEncoder(point_dim,hidden_dim,grid_size,extent)
        self.backbone=nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,3,padding=1),nn.GELU(),nn.Conv2d(hidden_dim,hidden_dim,3,padding=1),nn.GELU())
        self.heads=BEVHeads(hidden_dim,classes,center=False)

    def encode(self,points):
        return self.backbone(self.pillars(points))

    def forward(self,points):
        return self.heads(self.encode(points))


class CenterPoint(Grid):
    """Compact dense-voxel substitute for the upstream sparse backbone; center heads."""
    def __init__(self,point_dim=4,hidden_dim=32,classes=3,grid_size=16,extent=8.,z_bins=4,**_):
        super().__init__(grid_size,extent)
        self.point_dim,self.z_bins=point_dim,z_bins
        self.voxel_net=nn.Sequential(nn.Conv3d(point_dim,hidden_dim,3,padding=1),nn.GELU(),nn.Conv3d(hidden_dim,hidden_dim,3,padding=1),nn.GELU())
        self.backbone=nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,3,padding=1),nn.GELU())
        self.heads=BEVHeads(hidden_dim,classes)

    def encode(self,points):
        if points.ndim!=3 or points.shape[-1]!=self.point_dim:
            raise ValueError('points must have shape B,N,point_dim')
        grids=[]
        for cloud in points:
            xy=self.xy(cloud)
            z=((cloud[:,2]+self.extent)/(2*self.extent)*self.z_bins).floor().long()
            valid=self.valid(cloud,xy)&(z>=0)&(z<self.z_bins)
            cloud,xy,z=cloud[valid],xy[valid],z[valid]
            cells=self.z_bins*self.grid_size*self.grid_size
            sums=points.new_zeros((cells,self.point_dim))
            counts=points.new_zeros((cells,1))
            keys=(z*self.grid_size+xy[:,1])*self.grid_size+xy[:,0]
            sums=sums.index_add(0,keys,cloud)
            counts=counts.index_add(0,keys,torch.ones(len(keys),1,device=cloud.device,dtype=cloud.dtype))
            voxel=(sums/counts.clamp_min(1)).T.reshape(self.point_dim,self.z_bins,self.grid_size,self.grid_size)
            grids.append(voxel)
        volume=self.voxel_net(torch.stack(grids))
        return self.backbone(volume.amax(dim=2))

    def forward(self,points):
        return self.heads(self.encode(points))


class BEVFusion(Grid):
    """Small calibrated RGB-depth lift/splat and LiDAR BEV fusion reference."""
    def __init__(self,in_channels=3,point_dim=4,hidden_dim=32,classes=3,grid_size=16,extent=8.,depth_bins=4,**_):
        super().__init__(grid_size,extent)
        self.camera=nn.Sequential(nn.Conv2d(in_channels,hidden_dim,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(hidden_dim,hidden_dim,3,stride=2,padding=1),nn.GELU())
        self.depth=nn.Conv2d(hidden_dim,depth_bins,1)
        self.register_buffer('depth_values',torch.linspace(1.,extent,depth_bins))
        self.lidar=PointPillars(point_dim,hidden_dim,classes,grid_size,extent)
        self.fuse=nn.Sequential(nn.Conv2d(2*hidden_dim,hidden_dim,3,padding=1),nn.GELU())
        self.heads=BEVHeads(hidden_dim,classes)
        self.map_head=nn.Conv2d(hidden_dim,classes,1)

    def camera_bev(self,images,intrinsics,camera_to_lidar):
        if images.ndim!=5:
            raise ValueError('BEVFusion images must have shape B,V,C,H,W')
        b,v,c,h,w=images.shape
        if intrinsics.shape!=(b,v,3,3) or camera_to_lidar.shape!=(b,v,4,4):
            raise ValueError('Provide intrinsics B,V,3,3 and camera_to_lidar B,V,4,4 in a shared calibrated frame')
        x=self.camera(images.reshape(b*v,c,h,w))
        _,channels,fh,fw=x.shape
        probabilities=self.depth(x).softmax(dim=1)
        yy,xx=torch.meshgrid(torch.linspace(.5,h-.5,fh,device=x.device),torch.linspace(.5,w-.5,fw,device=x.device),indexing='ij')
        pixels=torch.stack([xx,yy,torch.ones_like(xx)],dim=-1).reshape(-1,3)
        outputs=[]
        for batch in range(b):
            accumulated=x.new_zeros((self.grid_size*self.grid_size,channels))
            mass=x.new_zeros((self.grid_size*self.grid_size,1))
            for view in range(v):
                idx=batch*v+view
                rays=pixels@torch.linalg.inv(intrinsics[batch,view]).T
                xyz=rays[None]*self.depth_values[:,None,None]
                transform=camera_to_lidar[batch,view]
                lidar_xyz=xyz@transform[:3,:3].T+transform[:3,3]
                xy=self.xy(lidar_xyz.reshape(-1,3))
                valid=self.valid(lidar_xyz.reshape(-1,3),xy)
                weights=probabilities[idx].reshape(-1,1)
                features=x[idx].flatten(1).T[None].expand(len(self.depth_values),-1,-1).reshape(-1,channels)
                keys=xy[:,1]*self.grid_size+xy[:,0]
                accumulated=accumulated.index_add(0,keys[valid],(features*weights)[valid])
                mass=mass.index_add(0,keys[valid],weights[valid])
            outputs.append((accumulated/mass.clamp_min(1e-6)).T.reshape(channels,self.grid_size,self.grid_size))
        return torch.stack(outputs)

    def forward(self,images,points,intrinsics,camera_to_lidar):
        camera=self.camera_bev(images,intrinsics,camera_to_lidar)
        lidar=self.lidar.encode(points)
        x=self.fuse(torch.cat([camera,lidar],dim=1))
        return {**self.heads(x),'map_logits':self.map_head(x)}
