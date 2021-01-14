import psycopg2
import geopandas as gpd
import pandas as pd
import numpy as np
import json

osmnx_ways = gpd.read_file('/Project0551/jingyi/fmm/example/data/harbin.tmp/edges.shp')

def get_lengths():
    fid = osmnx_ways.fid.values
    lengths = osmnx_ways.length.values * 1000
    return np.r_[0., lengths]

def get_dict_u():
    fid = osmnx_ways.fid.values
    dict_u = {}
    dict_u[0] = 0
    for road in fid:
        dict_u[road + 1] = road + 1 
    return dict_u, len(dict_u)

def get_highway(s):
    types = s
    if s[0] == '[':
        types = json.loads(s.replace("'", '"'))
        types = types[0]
    return types
    
def get_dict_s1():
    """
    s1: highway
    """
    
    highways = osmnx_ways.highway.apply(get_highway)
    types = np.sort(highways.unique())
    type_id = {}
    for i in range(len(types)):
        type_id[types[i]] = i
    df = pd.concat([osmnx_ways.fid, highways.apply(lambda t: type_id.get(t))],
                   axis=1, keys=['fid', 'type_id'])
    dict_s1 = {}
    dict_s1[0] = 0
    for _, row in df.iterrows():
        dict_s1[row.fid + 1] = row.type_id
    return dict_s1, len(highways.unique())

def get_dict_s2():
    """
    s2: number of lanes
    """
    def get_lane(s):
        types = s
        if s is None:
            types = '0'
        elif s[0] == '[':
            types = json.loads(s.replace("'", '"'))
            types = types[0]
        return types
    lanes = osmnx_ways.lanes.apply(get_lane)
    df = pd.concat([osmnx_ways.fid, lanes.apply(lambda x: int(x))],
                   axis=1, keys=['fid', 'lane'])
    dict_s2 = {}
    dict_s2[0] = 0
    for _, row in df.iterrows():
        dict_s2[row.fid + 1] = row.lane
    return dict_s2, len(lanes.unique())

def get_dict_s3():
    """
    s3: one way or not
    """
    oneway = osmnx_ways.oneway
    df = pd.concat([osmnx_ways.fid, oneway.apply(lambda x: int(x))],
                   axis=1, keys=['fid', 'oneway'])
    dict_s3 = {}
    dict_s3[0] = 0
    for _, row in df.iterrows():
        dict_s3[row.fid + 1] = row.oneway
    return dict_s3, len(oneway.unique())

    
    