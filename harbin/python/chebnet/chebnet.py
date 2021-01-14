import warnings
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import os
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (ChebConv, graclus, GCNConv, 
                                max_pool, max_pool_x, global_mean_pool)
from toolz.curried import *
import toolz
# from utils import *
from sklearn import metrics
from tqdm import tqdm

from pandas import DataFrame, Series
from networkx import Graph

def get_edge_index(G: Graph) -> torch.LongTensor:
    """
    Create pytorch geometric edge_index from a networkx graph.
    """
    edge_index = toolz.functoolz.pipe(G.edges(), map(list), list, torch.LongTensor)
    return edge_index.t().contiguous()

def get_x(df: DataFrame, num_nodes: int) -> torch.FloatTensor:
    """
    Get pytorch geometric input feature from observation dataframe.

    Inputs
        df: The observation dataframe with edgeid being attached. 
    Returns
        x (num_nodes, num_features): Input feature tensor. 
    """
    node_obs = {u: [v] for (u, v) in zip(df.edgeid.values, df.speed_mph_mean.values)}
    ## (num_nodes, 1)
    return torch.FloatTensor([get(u, node_obs, [0]) for u in range(num_nodes)])

def get_data(G: Graph, obs: DataFrame, unobs: DataFrame) -> Data:
    edge_index = get_edge_index(G)
    x = get_x(obs, G.number_of_nodes())
    y = get_x(unobs, G.number_of_nodes())
    return Data(x=x, edge_index=edge_index, y=y)

def normalized_cut_2d(edge_index: torch.LongTensor, num_nodes: int) -> torch.FloatTensor:
    edge_attr = torch.ones(edge_index.shape[1], device=edge_index.device)
    return normalized_cut(edge_index, edge_attr, num_nodes=num_nodes)

class ChebNet(torch.nn.Module):
    def __init__(self, num_features, num_nodes):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(num_features, 32, 2)
        self.conv2 = ChebConv(32, 64, 2)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_nodes)
    
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        cluster = graclus(data.edge_index, num_nodes=x.shape[0])
        data = max_pool(cluster, Data(x=x, batch=data.batch, edge_index=data.edge_index))
        
        x = F.relu(self.conv2(data.x, data.edge_index))
        cluster = graclus(data.edge_index, num_nodes=x.shape[0])
        x, batch = max_pool_x(cluster, x, data.batch)

        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x