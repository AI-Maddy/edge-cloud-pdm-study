"""Small shape-correct tensors for architecture construction/gradient checks only."""
import torch
from .factory import canonical


def sample_inputs(name,cfg,batch=2):
    name=canonical(name)
    p=cfg.get('parameters',{})
    f=p.get('input_dim',16)
    channels=p.get('in_channels',3)
    if name in ('cnn','yolo','yolo11','rtdetr'):
        return {'images':torch.randn(batch,channels,64,64)}
    if name=='dinov2':
        size=p.get('image_size',224)
        return {'images':torch.randn(batch,3,size,size)}
    if name=='clip':
        size=p['vision_config'].get('image_size',224)
        ids=torch.randint(1,100,(batch,8));ids[:,-1]=p['text_config']['eos_token_id']
        return {'images':torch.randn(batch,3,size,size),'input_ids':ids,'attention_mask':torch.ones_like(ids)}
    if name=='eva02':
        return {'images':torch.randn(batch,3,224,224)}
    if name in ('pedestrian_model','battery_model','risk_model'):
        return {'features':torch.randn(batch,f)}
    if name in ('autoencoder','simclr','byol'):
        return {'x':torch.randn(batch,f)}
    if name=='transformer':
        return {'src':torch.randn(batch,5,f),'tgt':torch.randn(batch,3,f)}
    if name=='transformer_encoder':
        return {'sequence':torch.randn(batch,5,f)}
    if name=='gnn':
        return {'nodes':torch.randn(batch,5,f),'adjacency':torch.ones(batch,5,5)-torch.eye(5)[None]}
    if name=='cnn_transformer_hybrid':
        return {'frames':torch.randn(batch,3,channels,16,16)}
    if name in ('pointpillars','centerpoint','bevfusion'):
        points=torch.randn(batch,64,p.get('point_dim',4))
        if name!='bevfusion':return {'points':points}
        intrinsics=torch.tensor([[8.,0.,8.],[0.,8.,8.],[0.,0.,1.]]).reshape(1,1,3,3).repeat(batch,2,1,1)
        extrinsics=torch.eye(4).reshape(1,1,4,4).repeat(batch,2,1,1)
        return {'images':torch.randn(batch,2,3,16,16),'points':points,'intrinsics':intrinsics,'camera_to_lidar':extrinsics}
    raise ValueError('No synthetic smoke contract for '+name)


def tensor_summary(value):
    if isinstance(value,torch.Tensor):
        if not torch.isfinite(value).all():raise ValueError('Model produced nonfinite outputs')
        return {'shape':list(value.shape),'dtype':str(value.dtype)}
    if isinstance(value,dict):return {str(k):tensor_summary(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [tensor_summary(v) for v in value]
    if value is None:return None
    return str(type(value).__name__)


def tensors(value):
    if isinstance(value,torch.Tensor):return [value]
    if isinstance(value,dict):return [t for v in value.values() for t in tensors(v)]
    if isinstance(value,(tuple,list)):return [t for v in value for t in tensors(v)]
    return []
