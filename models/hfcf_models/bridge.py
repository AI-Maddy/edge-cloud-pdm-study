"""Checkpoint-aware inference and integration with the existing scripts."""
from pathlib import Path
import torch
from .factory import build_model,canonical,load_config


def load_for_inference(name,config=None,checkpoint=None,allow_untrained=False):
    cfg=config or {}
    training_status='initialized-not-trained'
    if checkpoint:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        cfg={**payload['config'],**cfg}
        training_status=payload.get('training_status','unknown')
    elif cfg.get('local_pretrained'):
        training_status='external-pretrained-not-domain-validated'
    if not allow_untrained and training_status in ('initialized-not-trained','synthetic-smoke-training','unknown'):
        raise ValueError('Inference requires trained local weights; set allow_untrained=true only for explicit shape/synthetic checks')
    model=build_model(name,cfg,checkpoint).eval()
    return model,training_status


def tensor_inputs(inputs):
    return {k:torch.as_tensor(v,dtype=torch.long if k in ('input_ids','attention_mask') else torch.bool if k=='padding_mask' else torch.float32) for k,v in inputs.items()}


def json_tensors(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:json_tensors(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_tensors(v) for v in value]
    return value


def image_batch(paths,name,config):
    from PIL import Image
    from torchvision.transforms import Compose,Resize,ToTensor,Normalize,InterpolationMode
    key=canonical(name)
    params=config['parameters']
    if key=='clip':
        size=params['vision_config']['image_size'];mean=(.48145466,.4578275,.40821073);std=(.26862954,.26130258,.27577711)
    else:
        size=params.get('image_size',224 if key=='eva02' else 64);mean=(.485,.456,.406);std=(.229,.224,.225)
    transform=Compose([Resize((size,size),interpolation=InterpolationMode.BICUBIC),ToTensor(),Normalize(mean,std)])
    tensors=[]
    for path in paths:
        with Image.open(path) as image:tensors.append(transform(image.convert('RGB')))
    return torch.stack(tensors)


def record_embeddings(data,cfg,kind):
    rows=data.get('records',[]) if isinstance(data,dict) else data
    if not rows:raise ValueError('Embedding inference needs nonempty records')
    name=canonical(cfg['model'])
    supported=('cnn','dinov2','clip','eva02','autoencoder','simclr','byol','transformer_encoder','cnn_transformer_hybrid','gnn')
    if name not in supported:raise ValueError('Model has no generic record-embedding contract: '+name)
    model,status=load_for_inference(name,cfg.get('model_config'),cfg.get('checkpoint'),cfg.get('allow_untrained',False))
    with torch.no_grad():
        if name in ('cnn','dinov2','clip','eva02'):
            if all('image_path' in r for r in rows):x=image_batch([r['image_path'] for r in rows],name,model.hfcf_config)
            else:x=torch.as_tensor([r['image'] for r in rows],dtype=torch.float32)
            embedding=model.encode_images(x) if name=='clip' else model(x)
        elif name in ('autoencoder','simclr','byol'):
            x=torch.as_tensor([r['features'] for r in rows],dtype=torch.float32)
            embedding=model.encode(x) if name=='autoencoder' else model(x)
        elif name=='transformer_encoder':
            embedding=model(torch.as_tensor([r['sequence'] for r in rows],dtype=torch.float32))
        elif name=='cnn_transformer_hybrid':
            embedding=model(torch.as_tensor([r['frames'] for r in rows],dtype=torch.float32))
        else:
            nodes=torch.as_tensor([r['nodes'] for r in rows],dtype=torch.float32)
            adjacency=torch.as_tensor([r['adjacency'] for r in rows],dtype=torch.float32)
            embedding=model(nodes,adjacency).mean(dim=1)
    if not torch.isfinite(embedding).all():raise ValueError('Embedding inference produced nonfinite values')
    return {'records':[{**r,'embedding':embedding[i].tolist(),kind+'_embedding':embedding[i].tolist(),'embedding_model':name,'training_status':status} for i,r in enumerate(rows)],'mode':'local-model-inference','training_status':status}


def train_record_representations(data,cfg):
    from train_models import train
    rows=data.get('records',[]) if isinstance(data,dict) else data
    if not rows:raise ValueError('Training requires nonempty records')
    name=canonical(cfg['model'])
    first=[r[cfg.get('field','features')] for r in rows]
    payload={'features':first,'first':first}
    if name in ('simclr','byol'):
        if not all('second_features' in r for r in rows):raise ValueError('SimCLR/BYOL require second_features for each paired observation')
        payload['second']=[r['second_features'] for r in rows]
    override=cfg.get('model_config',{})
    override={**override,'parameters':{'input_dim':len(first[0]),**override.get('parameters',{})}}
    model,losses=train(name,payload,override,int(cfg.get('steps',20)),float(cfg.get('learning_rate',.001)),int(cfg.get('seed',7)))
    model.eval()
    with torch.no_grad():
        x=torch.as_tensor(first,dtype=torch.float32)
        z=model.encode(x) if name=='autoencoder' else model(x)
    training_status='synthetic-smoke-training' if isinstance(data,dict) and data.get('synthetic',False) else 'user-data-trained-not-independently-validated'
    if cfg.get('save_checkpoint'):
        path=Path(cfg['save_checkpoint'])
        if path.exists() and not cfg.get('overwrite_checkpoint'):raise ValueError('Checkpoint exists; set overwrite_checkpoint=true or choose a new path')
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'model_id':name,'config':model.hfcf_config,'state_dict':model.state_dict(),'training_status':training_status},path)
    return {'records':[{**r,'embedding':z[i].tolist()} for i,r in enumerate(rows)],'model':name,'losses':losses,'training_status':training_status,'checkpoint':cfg.get('save_checkpoint')}


def record_predictions(data,cfg):
    """Expose raw neural detector heads, without claiming decoding/NMS is performed."""
    rows=data.get('records',[]) if isinstance(data,dict) else data
    model,status=load_for_inference(cfg['model'],cfg.get('model_config'),cfg.get('checkpoint'),cfg.get('allow_untrained',False))
    out=[]
    with torch.no_grad():
        for row in rows:
            if 'model_inputs' not in row:raise ValueError('Registry detector inference needs record.model_inputs with batched tensors; outputs are raw heads')
            value=model(**tensor_inputs(row['model_inputs']))
            out.append({**row,'model_outputs':json_tensors(value),'training_status':status,'output_kind':'raw-neural-heads-not-decoded-detections'})
    return {'records':out,'mode':'local-model-raw-inference','training_status':status}
