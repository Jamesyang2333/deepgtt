# import psycopg2
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import scipy.sparse as sp
import torch
import networkx as nx
import dgl

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