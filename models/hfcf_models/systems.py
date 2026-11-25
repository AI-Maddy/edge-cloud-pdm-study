"""Reference realizations of conceptual models; no convergence/performance theorem.

The research notes do not specify complete architectures for these concepts. This
module makes explicit, testable implementation choices and reuses the scripts runtime.
"""
from dataclasses import dataclass, field
import importlib
from pathlib import Path
import sys
import numpy as np


def runtime(module):
    scripts=str(Path(__file__).resolve().parents[2]/'scripts')
    if scripts not in sys.path:sys.path.insert(0,scripts)
    return importlib.import_module('hfcf.'+module)


class ContextAwareFederation:
    """CAFIG/CAFC/HCAFC grouping coordinator using the implemented clustering stages."""
    def __init__(self,method='hdbscan',**config):
        self.method,self.config=method,config
        self.schema=None

    def fit_assign(self,records):
        f=runtime('features');r=runtime('federation')
        encoded=f.encode({'records':records},{})
        self.schema=encoded['schema']
        clusters=f.clustering(encoded,{**self.config,'field':'context_embedding'},self.method)
        return r.assign(clusters,{})

    def encode(self,records):
        if self.schema is None:raise ValueError('Fit the context schema before transforming later records')
        return runtime('features').encode({'records':records},{'schema':self.schema})


class KnowledgeGraph:
    """Explicit similarity relationships; no automatic causal semantics."""
    def __init__(self):self.graph={'nodes':[],'edges':[],'records':[]}

    def build(self,knowledge,threshold=.8):
        self.graph=runtime('knowledge').regional({'records':knowledge},{'threshold':threshold})
        return self.graph

    def merge(self,graphs):
        self.graph=runtime('knowledge').fuse({'graphs':[self.graph,*graphs]}, {})
        return self.graph


class TrustRelevanceModel:
    def trust(self,knowledge,**config):
        return runtime('knowledge').trust({'records':knowledge},config)

    def relevance(self,nodes,knowledge,**config):
        return runtime('knowledge').relevance({'nodes':nodes,'knowledge':knowledge},config)

    def distribute(self,pairs,**config):
        return runtime('knowledge').disseminate(pairs,config)


@dataclass
class DigitalTwin:
    """Time-ordered observed state/history twin; does not fabricate future predictions."""
    twin_id:str
    history:list=field(default_factory=list)

    def update(self,timestamp,state):
        if self.history and timestamp<=self.history[-1]['timestamp']:
            raise ValueError('Twin updates must have strictly increasing timestamps')
        x=np.asarray(state,dtype=float)
        if x.ndim!=1 or not x.size or not np.isfinite(x).all():raise ValueError('Twin state must be a nonempty finite vector')
        if self.history and len(x)!=len(self.history[-1]['state']):raise ValueError('Twin state dimension changed')
        self.history.append({'timestamp':timestamp,'state':x.tolist()})
        return self.history[-1]


class WorldModel:
    """Fitted ridge-linear state transition baseline; not a foundation world model."""
    def fit(self,states,actions,next_states,ridge=1e-3):
        s,a,y=[np.asarray(v,dtype=float) for v in (states,actions,next_states)]
        if any(x.ndim!=2 or not np.isfinite(x).all() for x in (s,a,y)) or not len(s) or len(s)!=len(a) or len(a)!=len(y) or y.shape[1]!=s.shape[1]:
            raise ValueError('Finite aligned 2D state/action/next-state matrices are required')
        if ridge<0:raise ValueError('ridge must be nonnegative')
        x=np.c_[s,a,np.ones(len(s))]
        self.state_dim,self.action_dim=s.shape[1],a.shape[1]
        self.weights=np.linalg.solve(x.T@x+ridge*np.eye(x.shape[1]),x.T@y)
        return self

    def predict(self,states,actions):
        if not hasattr(self,'weights'):raise ValueError('Fit WorldModel before predicting')
        s,a=np.asarray(states,dtype=float),np.asarray(actions,dtype=float)
        if s.ndim!=2 or a.ndim!=2 or s.shape[1]!=self.state_dim or a.shape[1]!=self.action_dim or len(s)!=len(a):
            raise ValueError('State/action input dimensions do not match the fitted model')
        return np.c_[s,a,np.ones(len(s))]@self.weights


class HFIPStateMachine:
    """Offline protocol progression, independent of network transport."""
    states=['REGISTERED','OBSERVING','EXTRACTING','FEDERATING','LEARNING','KNOWLEDGE_CREATED','DISSEMINATING','ADAPTING','IMPROVED_INFERENCE']

    def __init__(self):self.state=self.states[0]

    def advance(self,target):
        index=self.states.index(self.state)
        expected=self.states[index+1] if index+1<len(self.states) else 'OBSERVING'
        if target!=expected:raise ValueError(f'Expected transition {self.state} -> {expected}')
        self.state=target
        return self.state

    def objects(self,records,kind):
        return runtime('federation').serialize({'records':records},{'object_type':kind})


class CognitiveFabric:
    """FHFI/HFCF/SO-HFCS reference coordinator; numeric simulation, not a learned backbone."""
    def __init__(self,domain='automotive',**config):self.domain,self.config=domain,config

    def run(self,records=None,synthetic=False):
        return runtime('demo').run({'records':records or []},{**self.config,'synthetic':synthetic},self.domain)

    def convergence(self,previous,current,**config):
        return runtime('evaluation').convergence({'previous':previous,'current':current},config)

    def stability(self,history,**config):
        return runtime('evaluation').stability({'history':history},config)

    def optimize(self,candidates,**config):
        return runtime('evaluation').optimize({'candidates':candidates},config)
