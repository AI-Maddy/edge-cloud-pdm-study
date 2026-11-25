"""Constructible clustering estimators and explicit federation/grouping helpers."""
import numpy as np
from .systems import runtime,ContextAwareFederation


class GeoHashEncoder:
    alphabet='0123456789bcdefghjkmnpqrstuvwxyz'

    def __init__(self,precision=6):
        if not 1<=precision<=12:raise ValueError('precision must be 1..12')
        self.precision=precision

    def encode(self,latitude,longitude):
        if not -90<=latitude<=90 or not -180<=longitude<=180:raise ValueError('latitude/longitude outside valid ranges')
        ranges=[[-180.,180.],[-90.,90.]];values=[longitude,latitude]
        bit=0;character=0;result=''
        for index in range(self.precision*5):
            dimension=index%2;lo,hi=ranges[dimension];middle=(lo+hi)/2
            high=values[dimension]>=middle
            character=(character<<1)|int(high)
            ranges[dimension]=[middle,hi] if high else [lo,middle]
            bit+=1
            if bit==5:result+=self.alphabet[character];bit=0;character=0
        return result

    def transform(self,lat_lon):return [self.encode(float(lat),float(lon)) for lat,lon in lat_lon]


class FeatureFedAvg:
    def aggregate(self,models):return runtime('federation').aggregate({'records':models},{})


class GraphCommunities:
    def __init__(self,method='louvain',**config):self.method,self.config=method,config
    def fit_predict(self,graph):return runtime('features').scene_clusters(graph,{'method':self.method,**self.config})


def build_algorithm(name,**config):
    key=name.lower().replace('-','').replace('_','').replace(' ','')
    if key=='geohash':return GeoHashEncoder(**config)
    if key=='fedavg':return FeatureFedAvg()
    if key in ('leiden','louvain'):return GraphCommunities(key,**config)
    if key in ('hcafc','cafc','cafig'):return ContextAwareFederation(**config)
    from sklearn import cluster
    constructors={'kmeans':cluster.KMeans,'hdbscan':cluster.HDBSCAN,'dbscan':cluster.DBSCAN,'spectralclustering':cluster.SpectralClustering}
    if key not in constructors:raise ValueError('Unsupported algorithm: '+name)
    return constructors[key](**config)
