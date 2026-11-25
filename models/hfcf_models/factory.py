"""Offline, explicit model construction; defaults never fetch pretrained weights."""
import importlib
import json
import os
from pathlib import Path
import torch
from torch import nn
from . import neural, perception3d

ROOT=Path(__file__).resolve().parents[1]
ALIASES={'yolov11':'yolo11','yolo11n':'yolo11','rt-detr':'rtdetr','eva-02':'eva02','transformer encoder':'transformer_encoder','cnn transformer hybrid':'cnn_transformer_hybrid','autoencoders':'autoencoder'}


def canonical(name):
    key=name.lower().strip().replace(' ','_')
    return ALIASES.get(name.lower().strip(),ALIASES.get(key,key))


def list_models():
    return json.loads((ROOT/'catalog.json').read_text())['models']


def load_config(name):
    key=canonical(name)
    applications=json.loads((ROOT/'application-models.json').read_text())['models']
    models={m['id']:m for m in [*list_models(),*applications]}
    if key not in models:
        raise ValueError('Unknown model '+name+'; choose '+', '.join(models))
    return json.loads((ROOT/models[key]['config']).read_text())


def dependency(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ValueError(f'Missing dependency {name}; install models/requirements.txt or the documented upstream environment') from exc


class UltralyticsModule(nn.Module):
    def __init__(self,architecture,kind):
        super().__init__()
        package=dependency('ultralytics')
        self.network=(package.RTDETR if kind=='rtdetr' else package.YOLO)(architecture).model

    def forward(self,images):
        return self.network(images)


class DINOEncoder(nn.Module):
    def __init__(self,config,local_pretrained=None):
        super().__init__()
        package=dependency('transformers')
        self.network=(package.Dinov2Model.from_pretrained(local_pretrained,local_files_only=True) if local_pretrained else package.Dinov2Model(package.Dinov2Config(**config)))

    def forward(self,images):
        return self.network(pixel_values=images).last_hidden_state[:,0]


class CLIPNetwork(nn.Module):
    def __init__(self,config,local_pretrained=None):
        super().__init__()
        package=dependency('transformers')
        self.network=(package.CLIPModel.from_pretrained(local_pretrained,local_files_only=True) if local_pretrained else package.CLIPModel(package.CLIPConfig(**config)))

    def forward(self,images,input_ids,attention_mask=None):
        output=self.network(pixel_values=images,input_ids=input_ids,attention_mask=attention_mask)
        return {'image_embeddings':output.image_embeds,'text_embeddings':output.text_embeds,'logits_per_image':output.logits_per_image}

    def encode_images(self,images):
        # Use the vision tower and projection directly for compatibility across HF versions.
        pooled=self.network.vision_model(pixel_values=images).pooler_output
        return torch.nn.functional.normalize(self.network.visual_projection(pooled),dim=-1)


class EVAEncoder(nn.Module):
    def __init__(self,architecture):
        super().__init__()
        self.network=dependency('timm').create_model(architecture,pretrained=False,num_classes=0)

    def forward(self,images):
        return self.network(images)


def build_model(name,config=None,checkpoint=None):
    cfg=load_config(name)
    if config:
        cfg={**cfg,**config,'parameters':{**cfg.get('parameters',{}),**config.get('parameters',{})}}
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    backend=cfg['backend']
    params=cfg.get('parameters',{})
    if cfg.get('local_pretrained') and not Path(cfg['local_pretrained']).is_dir():
        raise ValueError('local_pretrained must name an existing local Hugging Face directory')
    if backend=='native':
        cls=getattr(neural,cfg['class'],None) or getattr(perception3d,cfg['class'],None)
        if cls is None:
            raise ValueError('Unknown native model class')
        model=cls(**params)
    elif backend=='ultralytics':
        # Only installed YAML architectures or explicit local YAMLs: never a .pt hub name.
        architecture=cfg['architecture']
        if not architecture.endswith(('.yaml','.yml')):
            raise ValueError('Ultralytics construction needs a YAML architecture; use checkpoint for local saved state')
        model=UltralyticsModule(architecture,canonical(name))
    elif backend=='transformers-dinov2':
        model=DINOEncoder(params,cfg.get('local_pretrained'))
    elif backend=='transformers-clip':
        model=CLIPNetwork(params,cfg.get('local_pretrained'))
    elif backend=='timm':
        model=EVAEncoder(cfg['architecture'])
    elif backend=='mmdetection3d':
        # Official models require their original local config hierarchy and compiled ops.
        path=cfg.get('upstream_config')
        if not path or not Path(path).is_file():
            raise ValueError('Official MMDetection3D build requires upstream_config pointing to a complete local config hierarchy')
        dependency('mmdet3d.utils').register_all_modules()
        config_class=dependency('mmengine.config').Config
        upstream=config_class.fromfile(path)
        model=dependency('mmdet3d.registry').MODELS.build(upstream.model)
    else:
        raise ValueError('Unknown backend '+backend)
    if checkpoint:
        path=Path(checkpoint)
        if not path.is_file():
            raise ValueError('checkpoint must be an existing local file')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('model_id')!=canonical(name):
            raise ValueError('Checkpoint model_id does not match requested model')
        model.load_state_dict(saved['state_dict'],strict=True)
    model.hfcf_config=cfg
    return model
