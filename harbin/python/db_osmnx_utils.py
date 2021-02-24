# import psycopg2
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import scipy.sparse as sp
import torch
import networkx as nx
import dgl
import math

def gps2webmercator(lon, lat):
    """
    Converting GPS coordinate to Web Mercator coordinate
    """
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

def gps2webmercator_lon(lon):
    """
    Converting GPS coordinate to Web Mercator coordinate
    """
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    return semimajoraxis * east

def gps2webmercator_lat(lat):
    """
    Converting GPS coordinate to Web Mercator coordinate
    """
    semimajoraxis = 6378137.0
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return 3189068.5 * math.log((1 + t) / (1 - t))


# osmnx_ways = gpd.read_file('/Project0551/jingyi/fmm/example/data/harbin.tmp/edges.shp')
osmnx_ways = gpd.read_file('/Project0551/jingyi/fmm/example/data/chengdu.tmp/edges.shp')

print('hello')
lon_min = 126.506130
lat_min = 45.657920
lon_max = 126.771862
lat_max = 45.830905
minx, miny = gps2webmercator(lon_min, lat_min)
maxx, maxy = gps2webmercator(lon_max, lat_max)
lon_range = maxx - minx
lat_range = maxy - miny


def get_map_mask():
    mask = torch.zeros((1, 1, 138, 148))
    geo = osmnx_ways.geometry.values
    for pos in geo:
        pos_str = pos.__str__()
        loc = pos_str[12:-1].split(',')
        loc_start = loc[0].split(" ")
        loc_end = loc[1].strip().split(" ")
        lon_start = float(loc_start[0])
        lat_start = float(loc_start[1])
        lon_end = float(loc_end[0])
        lat_end = float(loc_end[1])
        for i in range(30):
            cur_lon = lon_start + (lon_end - lon_start) / 30.0 * i
            cur_lat = lat_start + (lat_end - lat_start) / 30.0 * i
            x, y  = gps2webmercator(cur_lon, cur_lat)
            x = math.floor((x - minx) / (lon_range / 148))
            if x == 148:
                x = 147
            y = math.floor((y - miny) / (lat_range / 138))
            if y == 138:
                y = 137
            mask[0][0][y][x] = 1

    return mask
            
        

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
#     df = pd.concat([osmnx_ways.fid, oneway.apply(lambda x: int(x))],
#                    axis=1, keys=['fid', 'oneway'])
    # For chengdu, the field values are string
    df = pd.concat([osmnx_ways.fid, oneway.apply(lambda x: 1 if x=='True' else 0)],
                   axis=1, keys=['fid', 'oneway'])
    dict_s3 = {}
    dict_s3[0] = 0
    for _, row in df.iterrows():
        dict_s3[row.fid + 1] = row.oneway
    return dict_s3, len(oneway.unique())

def get_adj():
    adj = get_scipy_adj()
    
     # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj
    
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    
def get_scipy_adj():
    node_dict = {}
    for index, row in osmnx_ways.iterrows():
        if row['u'] in node_dict:
            node_dict[row['u']].append(row['fid'])
        else:
            node_dict[row['u']] = [row['fid']]
        if row['v'] in node_dict:
            node_dict[row['v']].append(row['fid'])
        else:
            node_dict[row['v']] = [row['fid']]
    edge_1 = []
    edge_2 = []
    edge_set = set()
    for node, fid_list in node_dict.items():
        for i in range(len(fid_list)):
            for j in range(i):
                if (fid_list[i], fid_list[j]) in edge_set or (fid_list[j], fid_list[i]) in edge_set:
                    continue
                else:
                    edge_set.add((fid_list[i], fid_list[j]))
                    edge_1.append(fid_list[i])
                    edge_2.append(fid_list[j])
                    edge_2.append(fid_list[i])
                    edge_1.append(fid_list[j])
    adj = sp.coo_matrix((np.ones(len(edge_1)), (edge_1, edge_2)))
    return adj

def get_graph():
    adj = get_scipy_adj()
    G = nx.from_scipy_sparse_matrix(adj)
    return G

def get_graph_dgl(device=None):
    adj = get_scipy_adj()
    G = dgl.from_scipy(adj, device=device)
    return G


        
    
        
        
        
        
    