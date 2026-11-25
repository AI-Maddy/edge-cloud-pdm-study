"""Small trainable instances of model families named in the research notes.

These are reference architectures, not claims to reproduce a particular pretrained model.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32, embedding_dim=16, **_):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, hidden_dim, 3, padding=1), nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden_dim, embedding_dim))

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError('CNN expects B,C,H,W images')
        return self.encoder(images)


class PositionalEncoding(nn.Module):
    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
        frequencies = torch.exp(torch.arange(0, x.shape[2], 2, device=x.device, dtype=x.dtype) * (-math.log(10000.) / x.shape[2]))
        phase = positions * frequencies
        pe = torch.zeros_like(x[0])
        pe[:, 0::2] = phase.sin()
        pe[:, 1::2] = phase[:, :pe[:, 1::2].shape[1]].cos()
        return x + pe[None]


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, embedding_dim=16, heads=4, layers=2, **_):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.position = PositionalEncoding()
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim*4, dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, sequence, padding_mask=None):
        if sequence.ndim != 3:
            raise ValueError('TransformerEncoder expects B,T,F sequence')
        x = self.encoder(self.position(self.input(sequence)), src_key_padding_mask=padding_mask)
        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).to(x.dtype)
            if (valid.sum(dim=1) == 0).any():
                raise ValueError('Every sequence needs at least one unmasked token')
            pooled = (x*valid).sum(dim=1) / valid.sum(dim=1)
        else:
            pooled = x.mean(dim=1)
        return self.output(pooled)


class Transformer(nn.Module):
    """Encoder-decoder Transformer; consumes source and target sequences."""
    def __init__(self, input_dim=16, hidden_dim=32, embedding_dim=16, heads=4, layers=2, **_):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.position = PositionalEncoding()
        self.transformer = nn.Transformer(hidden_dim, heads, layers, layers, hidden_dim*4, dropout=0., batch_first=True)
        self.output = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, src, tgt):
        causal = torch.triu(torch.ones(tgt.shape[1], tgt.shape[1], device=tgt.device, dtype=torch.bool), diagonal=1)
        x = self.transformer(self.position(self.input(src)), self.position(self.input(tgt)), tgt_mask=causal)
        return self.output(x)


class GNN(nn.Module):
    """Dense two-layer GCN with symmetric normalization and self loops."""
    def __init__(self, input_dim=16, hidden_dim=32, embedding_dim=16, **_):
        super().__init__()
        self.first = nn.Linear(input_dim, hidden_dim, bias=False)
        self.second = nn.Linear(hidden_dim, embedding_dim, bias=False)

    def forward(self, nodes, adjacency):
        if nodes.ndim != 3 or adjacency.shape != (nodes.shape[0], nodes.shape[1], nodes.shape[1]):
            raise ValueError('GNN expects nodes B,N,F and adjacency B,N,N')
        if (adjacency < 0).any() or not torch.allclose(adjacency, adjacency.transpose(-1,-2)):
            raise ValueError('GCN adjacency must be nonnegative and symmetric')
        a = adjacency + torch.eye(nodes.shape[1], device=nodes.device)[None]
        degree = a.sum(dim=-1).clamp_min(1e-8).rsqrt()
        a = degree.unsqueeze(-1)*a*degree.unsqueeze(-2)
        return a @ self.second(F.gelu(a @ self.first(nodes)))


class CNNTransformerHybrid(nn.Module):
    """Frame CNN followed by temporal Transformer encoding."""
    def __init__(self, in_channels=3, hidden_dim=32, embedding_dim=16, heads=4, layers=2, **_):
        super().__init__()
        self.cnn = CNN(in_channels, hidden_dim, hidden_dim)
        self.temporal = TransformerEncoder(hidden_dim, hidden_dim, embedding_dim, heads, layers)

    def forward(self, frames):
        if frames.ndim != 5:
            raise ValueError('Hybrid expects B,T,C,H,W')
        b,t,c,h,w = frames.shape
        features = self.cnn(frames.reshape(b*t,c,h,w)).reshape(b,t,-1)
        return self.temporal(features)


class Autoencoder(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, embedding_dim=8, **_):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embedding_dim))
        self.decoder = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def mlp_encoder(input_dim, hidden_dim, embedding_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embedding_dim))


class SimCLR(nn.Module):
    """Paired-view NT-Xent objective with configurable MLP or CNN encoder."""
    def __init__(self, input_dim=16, hidden_dim=32, embedding_dim=16, projection_dim=16, temperature=.2, backbone='mlp', in_channels=3, **_):
        super().__init__()
        if temperature <= 0 or backbone not in ('mlp','cnn'):
            raise ValueError('temperature must be positive and backbone mlp/cnn')
        self.encoder = CNN(in_channels, hidden_dim, embedding_dim) if backbone == 'cnn' else mlp_encoder(input_dim, hidden_dim, embedding_dim)
        self.projector = mlp_encoder(embedding_dim, hidden_dim, projection_dim)
        self.temperature = temperature

    def forward(self, x):
        return self.encoder(x)

    def loss(self, first, second):
        if first.shape != second.shape or len(first) < 2:
            raise ValueError('SimCLR requires at least two aligned paired observations')
        n = len(first)
        z = F.normalize(self.projector(self.encoder(torch.cat([first,second]))), dim=-1)
        logits = z @ z.T / self.temperature
        logits = logits.masked_fill(torch.eye(2*n, device=z.device, dtype=torch.bool), -torch.inf)
        labels = (torch.arange(2*n, device=z.device)+n) % (2*n)
        return F.cross_entropy(logits, labels)


class BYOL(nn.Module):
    """Online/predictor and stopped-gradient EMA target; symmetric paired-view loss."""
    def __init__(self, input_dim=16, hidden_dim=32, embedding_dim=16, projection_dim=16, momentum=.99, backbone='mlp', in_channels=3, **_):
        super().__init__()
        if not 0 <= momentum < 1 or backbone not in ('mlp','cnn'):
            raise ValueError('momentum must lie in [0,1), backbone mlp/cnn')
        self.encoder = CNN(in_channels,hidden_dim,embedding_dim) if backbone == 'cnn' else mlp_encoder(input_dim,hidden_dim,embedding_dim)
        self.projector = mlp_encoder(embedding_dim,hidden_dim,projection_dim)
        self.predictor = mlp_encoder(projection_dim,hidden_dim,projection_dim)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.target_projector = copy.deepcopy(self.projector).requires_grad_(False)
        self.momentum = momentum

    def forward(self, x):
        return self.encoder(x)

    def loss(self, first, second):
        if first.shape != second.shape:
            raise ValueError('BYOL paired views must have matching shapes')
        a = F.normalize(self.predictor(self.projector(self.encoder(first))),dim=-1)
        b = F.normalize(self.predictor(self.projector(self.encoder(second))),dim=-1)
        with torch.no_grad():
            ta = F.normalize(self.target_projector(self.target_encoder(first)),dim=-1)
            tb = F.normalize(self.target_projector(self.target_encoder(second)),dim=-1)
        return (2-2*(a*tb).sum(-1)).mean()/2 + (2-2*(b*ta).sum(-1)).mean()/2

    @torch.no_grad()
    def update_target(self):
        for online,target in [(self.encoder,self.target_encoder),(self.projector,self.target_projector)]:
            for p,t in zip(online.parameters(),target.parameters()):
                t.mul_(self.momentum).add_(p,alpha=1-self.momentum)


class SensorTaskModel(nn.Module):
    """Unspecified pedestrian/battery/risk role instantiated as a numeric MLP head.

    This is an explicit application baseline, not an image detector or a trained
    battery degradation model. Callers choose inputs, labels, and loss semantics.
    """
    def __init__(self,input_dim=16,hidden_dim=32,outputs=1,**_):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,outputs))

    def forward(self,features):return self.network(features)
