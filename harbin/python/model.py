
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gat.models import GAT, SpGAT
from gat.model_dgl import GAT_DGL
from gcn.model import GCN, Net, GCN_RES
# from chebnet.chebnet import ChebNet, get_edge_index
import db_osmnx_utils as db_utils
from gin.gin import GIN
# from torch_geometric.data import Data, DataLoader

from dgl.nn.pytorch.glob import AvgPooling

#use_cuda = False
#device = torch.device("cuda" if use_cuda else "cpu")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 dropout, use_selu=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.nonlinear_f = F.selu if use_selu else F.leaky_relu
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h1 = self.dropout(self.nonlinear_f(self.fc1(x)))
        return self.fc2(h1)

class MLP2(nn.Module):
    """
    MLP with two output layers
    """
    def __init__(self, input_size, hidden_size, output_size,
                 dropout, use_selu=False):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, output_size)
        self.fc22 = nn.Linear(hidden_size, output_size)
        self.nonlinear_f = F.selu if use_selu else F.relu
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h1 = self.dropout(self.nonlinear_f(self.fc1(x)))
        return self.fc21(h1), self.fc22(h1)

def logwsumexp(x, w):
    """
    log weighted sum (along dim 1) exp, i.e., log(sum(w * exp(x), 1)).

    Input:
      x (n, m): exponents
      w (n, m): weights
    Output:
      y (n,)
    """
    maxv, _ = torch.max(x, dim=1, keepdim=True)
    y = torch.log(torch.sum(torch.exp(x - maxv) * w, dim=1, keepdim=True)) + maxv
    return y.squeeze(1)

class ProbRho(nn.Module):
    """
    s1 (14): road type
    s2 (7): number of lanes
    s3 (2): one way or not
    """
    def __init__(self, num_u, dim_u, dict_u, lengths,
                       num_s1, dim_s1, dict_s1,
                       num_s2, dim_s2, dict_s2,
                       num_s3, dim_s3, dict_s3,
                       hidden_size, dim_rho,
                       dropout, use_selu, device):
        super(ProbRho, self).__init__()
        self.lengths = torch.tensor(lengths, dtype=torch.float32, device=device)
        self.dict_u = dict_u
        self.dict_s1 = dict_s1
        self.dict_s2 = dict_s2
        self.dict_s3 = dict_s3
        # using padding_idx 0 to make sure the padded path segments doesn't affect our model
        self.embedding_u = nn.Embedding(num_u, dim_u, padding_idx=0)
        self.embedding_s1 = nn.Embedding(num_s1, dim_s1, padding_idx=0)
        self.embedding_s2 = nn.Embedding(num_s2, dim_s2, padding_idx=0)
        self.embedding_s3 = nn.Embedding(num_s3, dim_s3, padding_idx=0)
        self.device = device
        self.f = MLP2(dim_u+dim_s1+dim_s2+dim_s3,
                      hidden_size, dim_rho, dropout, use_selu)

    def roads2u(self, roads):
        """
        road id to word id (u)
        """
        return self.roads_s_i(roads, self.dict_u)

    def roads_s_i(self, roads, dict_s):
        """
        road id to feature id

        This function should be called in cpu
        ---
        Input:
        roads (batch_size * seq_len): road ids
        dict_s (dict): the mapping from road id to feature id
        Output:
        A tensor like roads
        """
        return roads.clone().apply_(lambda k: dict_s[k])

    def roads_length(self, roads, ratios=None):
        """
        roads (batch_size, seq_len): road id to road length
        ratios (batch_size, seq_len): The ratio of each road segment
        """
        if ratios is not None:
            return self.lengths[roads] * ratios.to(self.device)
        else:
            return self.lengths[roads]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, roads):
        """
        roads (batch_size * seq_len)
        """
        u  = self.embedding_u(self.roads2u(roads).to(self.device))
        s1 = self.embedding_s1(self.roads_s_i(roads, self.dict_s1).to(self.device))
        s2 = self.embedding_s2(self.roads_s_i(roads, self.dict_s2).to(self.device))
        s3 = self.embedding_s3(self.roads_s_i(roads, self.dict_s3).to(self.device))
        x  = torch.cat([u, s1, s2, s3], dim=2)
        mu, logvar = self.f(x)
        return self.reparameterize(mu, logvar)



class ProbTraffic(nn.Module):
    """
    Modelling the probability of the traffic state `c`
    """
    def __init__(self, n_in, hidden_size, dim_c, dropout, use_selu):
        super(ProbTraffic, self).__init__()
        conv_layers = [
            nn.Conv2d(n_in, 32, (5, 5), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AvgPool2d(7)
        ]
        self.f1 = nn.Sequential(*conv_layers)
        self.f2 = MLP2(128*2*2, hidden_size, dim_c, dropout, use_selu)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, T):
        """
        Input:
          T (batch_size, nchannel, height, width)
        Output:
          c, mu, logvar (batch_size, dim_c)
        """
        x = self.f1(T)
        mu, logvar = self.f2(x.view(x.size(0), -1))
        return self.reparameterize(mu, logvar), mu, logvar


class ProbTravelTime(nn.Module):
    def __init__(self, dim_rho, dim_c,
                       hidden_size, dropout, use_selu):
        super(ProbTravelTime, self).__init__()
        self.f = MLP2(dim_rho+dim_c, hidden_size, 1, dropout, use_selu)

    def forward(self, rho, c, w, l):
        """
        rho (batch_size, seq_len, dim_rho)
        c (1, dim_c): the traffic state vector sampling from ProbTraffic
        w (batch_size, seq_len): the normalized road lengths
        l (batch_size, ): route or path lengths
        """
        ## (batch_size, seq_len, dim_rho+dim_c)
        x = torch.cat([rho, c.expand(*rho.shape[:-1], -1)], 2)
        ## (batch_size, seq_len, 1)
        logm, logv = self.f(x)
        ## (batch_size, seq_len)
        logm, logv = logm.squeeze(2), logv.squeeze(2)
        #m, v = torch.exp(logm), torch.exp(logv)
        ## (batch_size, )
        #m_agg, v_agg = torch.sum(m * w, 1), torch.sum(v * w.pow(2), 1)
        ## parameters of IG distribution
        ## (batch_size, )
        #logμ = torch.log(l) - torch.log(m_agg)
        #logλ = 3*logμ - torch.log(v_agg) - 2*torch.log(l)
        ## (batch_size, )
#         logm_agg = logwsumexp(logm, w)
#         logv_agg = logwsumexp(logv, w.pow(2))
        logm_agg = torch.logsumexp(logm + torch.log(w), dim=1)
        logv_agg = torch.logsumexp(logv + 2*torch.log(w), dim=1)
        logl = torch.log(l)
        ## parameters of IG distribution
        ## (batch_size, )
        logμ = logl - logm_agg
        logλ = logl - 3*logm_agg - logv_agg
        return logμ, logλ

class TTime_gnn(nn.Module):
    # note that it should satisfy n_head * d_v = dim_u + dim_s1 + dim_s2 + dim_s3
    def __init__(self, num_u, dim_u, dict_u,
                       num_s1, dim_s1, dict_s1,
                       num_s2, dim_s2, dict_s2,
                       num_s3, dim_s3, dict_s3, 
                       dim_rho, dim_c, lengths,
                       hidden_size1, hidden_size2, hidden_size3,
                       dropout, use_selu, device):
        super(TTime_gnn, self).__init__()
        self.probrho = ProbRho(num_u, dim_u, dict_u, lengths,
                               num_s1, dim_s1, dict_s1,
                               num_s2, dim_s2, dict_s2,
                               num_s3, dim_s3, dict_s3,
                               hidden_size1, dim_rho,
                               dropout, use_selu, device).to(device)
#         self.probtraffic = ProbTraffic_gat_dgl(db_utils.get_graph_dgl(device), 3, 3, 8, 128, [4, 4, 4, 1], F.elu, 0.6, 0.6, 0.2, False).to(device)
#         self.probtraffic = ProbTraffic_gat_dgl_pool(db_utils.get_graph_dgl(device), 3, 3, 8, 128, [8, 8, 8, 1], F.elu, 0.2, 0.2, 0.2, False).to(device)
#         self.probtraffic = ProbTraffic_gcn(db_utils.get_graph_dgl(device)).to(device)
#         self.probtraffic = ProbTraffic_gcn_pool(db_utils.get_graph_dgl(device)).to(device)
#         self.probtraffic = ProbTraffic_gcn_res_pool(db_utils.get_graph_dgl(device)).to(device)
#         self.probtraffic = ProbTraffic_gin(db_utils.get_graph_dgl(device)).to(device)
        self.probtraffic = ProbTraffic_chebnet(db_utils.get_graph_dgl(device)).to(device)
#         self.probttime = ProbTravelTime_gnn(dim_rho, dim_c, hidden_size3, dropout, use_selu, device).to(device)
        self.probttime = ProbTravelTime(dim_rho, dim_c, hidden_size3, dropout, use_selu).to(device)

    def forward(self, roads, ratios, T, adj):
        road_lens = self.probrho.roads_length(roads, ratios)
        l = road_lens.sum(dim=1) # the whole trip lengths
        w = road_lens / road_lens.sum(dim=1, keepdim=True) # road weights
        rho = self.probrho(roads)
        c = self.probtraffic(T[:, 2:3] / 0.23398585617542267)
        # normalized average speed
#         c, mu_c, logvar_c = self.probtraffic(T[:, 2:3] / 0.23398585617542267)
#         logμ, logλ = self.probttime(rho, c, w, l, roads)
        logμ, logλ = self.probttime(rho, c, w, l)
#         return logμ, logλ, mu_c, logvar_c
        return logμ, logλ

class ProbTraffic_gat(nn.Module):
    """
    Modelling the probability of the traffic state `c`
    """
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(ProbTraffic_gat, self).__init__()
        self.gat = SpGAT(nfeat, nhid, nclass, dropout, alpha, nheads)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, T, adj):
        """
        Input:
          T (batch_size, nchannel, height, width)
        Output:
          c, mu, logvar (batch_size, dim_c)
        """
        c = self.gat(T, adj)
        return c
    
class ProbTraffic_gat_dgl(nn.Module):
    """
    Modelling the probability of the traffic state `c`
    """
    def __init__(self, g, num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, negative_slope, residual):
        super(ProbTraffic_gat_dgl, self).__init__()
        self.gat = GAT_DGL(g, num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, negative_slope, residual)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, T):
        """
        Input:
          T (batch_size, nchannel, height, width)
        Output:
          c, mu, logvar (batch_size, dim_c)
        """
        c = self.gat(T)
        return c

class ProbTraffic_gat_dgl_pool(nn.Module):
    """
    Modelling the probability of the traffic state `c`
    """
    def __init__(self, g, num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, negative_slope, residual):
        super(ProbTraffic_gat_dgl_pool, self).__init__()
        self.g = g
        self.gat = GAT_DGL(g, num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, negative_slope, residual)
        self.avgpool = AvgPooling()
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, T):
        """
        Input:
          T (batch_size, nchannel, height, width)
        Output:
          c, mu, logvar (batch_size, dim_c)
        """
        c = self.gat(T)
        c = self.avgpool(self.g, c)
        return c


class ProbTraffic_gcn(nn.Module):
    """
    Modelling the probability of the traffic state `c`
    """
    def __init__(self, g):
        super(ProbTraffic_gcn, self).__init__()
        self.g = g
        self.gcn = Net()
#         self.gcn = (g, in_feats, n_hidden, n_classes, n_layers, activation, dropout);
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, T):
        """
        Input:
          T (batch_size, nchannel, height, width)
        Output:
          c, mu, logvar (batch_size, dim_c)
        """
        c = self.gcn(self.g, T)
#         c = self.gcn(T)
        return c

class ProbTraffic_gcn_res(nn.Module):
    """
    Modelling the probability of the traffic state `c`
    """
    def __init__(self, g):
        super(ProbTraffic_gcn_res, self).__init__()
        self.g = g
        self.gcn = GCN_RES()
#         self.gcn = (g, in_feats, n_hidden, n_classes, n_layers, activation, dropout);
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, T):
        """
        Input:
          T (batch_size, nchannel, height, width)
        Output:
          c, mu, logvar (batch_size, dim_c)
        """
        c = self.gcn(self.g, T)
#         c = self.gcn(T)
        return c
    
class ProbTraffic_gcn_pool(nn.Module):
    """
    Modelling the probability of the traffic state `c`
    """
    def __init__(self, g):
        super(ProbTraffic_gcn_pool, self).__init__()
        self.g = g
        self.gcn = Net()
        self.avgpool = AvgPooling()
        self.layernorm = nn.LayerNorm(128)
        self.f1 = MLP(128, 256, 128, 0.2, True)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, T):
        """
        Input:
          T (batch_size, nchannel, height, width)
        Output:
          c, mu, logvar (batch_size, dim_c)
        """
        c = self.gcn(self.g, T)
        c = self.avgpool(self.g, c)
#         c = torch.unsqueeze(c, 0)
#         c = self.layernorm(c)
#         c = torch.squeeze(c)
        c = self.f1(c)
        return c
    
class ProbTraffic_gcn_res_pool(nn.Module):
    """
    Modelling the probability of the traffic state `c`
    """
    def __init__(self, g):
        super(ProbTraffic_gcn_res_pool, self).__init__()
        self.g = g
        self.gcn = GCN_RES()
        self.avgpool = AvgPooling()
        self.layernorm = nn.LayerNorm(128)
        self.f2 = MLP2(128, 256, 128, 0.2, True)
#         self.f1 = MLP(128, 256, 128, 0.2, True)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, T):
        """
        Input:
          T (batch_size, nchannel, height, width)
        Output:
          c, mu, logvar (batch_size, dim_c)
        """
        c = self.gcn(self.g, T)
        c = self.avgpool(self.g, c)
        c = torch.unsqueeze(c, 0)
        c = self.layernorm(c)
        c = torch.squeeze(c)
        mu, logvar = self.f2(c)
        return self.reparameterize(mu, logvar), mu, logvar
#         c = self.f1(c)
#         return c

# class ProbTraffic_chebnet(nn.Module):
#     """
#     Modelling the probability of the traffic state `c`
#     """
#     def __init__(self, g):
#         super(ProbTraffic_chebnet, self).__init__()
#         self.g = g
#         self.edge_idx = get_edge_index(g)
#         self.chebnet = ChebNet(1, 128)
#         self.f2 = MLP2(128, 256, 128, 0.2, True)
# #         self.f1 = MLP(128, 256, 128, 0.2, True)
        
#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5*logvar)
#             eps = torch.randn_like(std)
#             return eps.mul(std).add(mu)
#         else:
#             return mu

#     def forward(self, T):
#         """
#         Input:
#           T (batch_size, nchannel, height, width)
#         Output:
#           c, mu, logvar (batch_size, dim_c)
#         """
#         data = Data(x=T, batch=1, edge_index=self.edge_idx)
#         c = self.chebnet(self.edge_idx, data)
# #         c = self.avgpool(self.g, c)
# #         c = torch.unsqueeze(c, 0)
# #         c = self.layernorm(c)
# #         c = torch.squeeze(c)
# #         mu, logvar = self.f2(c)
# #         return self.reparameterize(mu, logvar), mu, logvar
#         return c
    
    
class ProbTraffic_gin(nn.Module):
    """
    Modelling the probability of the traffic state `c`
    """
    def __init__(self, g):
        super(ProbTraffic_gin, self).__init__()
        self.g = g
        self.gin = GIN(3, 2, 3, 64, 128, 0.2, False, 'sum', 'mean')
        self.f2 = MLP2(128, 256, 128, 0.2, True)
#         self.f2 = MLP(128, 256, 128, 0.2, True)
        self.layernorm = nn.LayerNorm(128)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, T):
        """
        Input:
          T (batch_size, nchannel, height, width)
        Output:
          c, mu, logvar (batch_size, dim_c)
        """
        c = self.gin(self.g, T)
        c = self.layernorm(c)
#         c = self.f2(c)
        mu, logvar = self.f2(c)
        return self.reparameterize(mu, logvar), mu, logvar
#         return c
    
    
class ProbTravelTime_gnn(nn.Module):
    def __init__(self, dim_rho, dim_c,
                       hidden_size, dropout, use_selu, device):
        super(ProbTravelTime_gnn, self).__init__()
        self.f = MLP2(dim_rho+dim_c, hidden_size, 1, dropout, use_selu)
        self.f2 = MLP(128, 256, 128, 0.2, True)
        self.device = device

    def forward(self, rho, c, w, l, roads):
        """
        rho (batch_size, seq_len, dim_rho)
        c (n_nodes, dim_c): the traffic state vector sampling from ProbTraffic
        w (batch_size, seq_len): the normalized road lengths
        l (batch_size, ): route or path lengths
        """
        ## (batch_size, seq_len, dim_rho+dim_c)
        roads = roads.to(self.device)
        pad_feature = torch.zeros((1, list(c.size())[1]), device=self.device)
        c_padded  = torch.cat((pad_feature, c), dim=0)
#         embed_list = [torch.index_select(c_padded, 0, roads[i]) for i in range(list(roads.size())[0])]
#         c_stack = torch.stack(embed_list, 0)
        c_padded_expand = torch.unsqueeze(c_padded, 0).expand(roads.size(0), -1, -1)
        dummy = roads.unsqueeze(2).expand(roads.size(0), roads.size(1), c_padded_expand.size(2))
        c_stack = torch.gather(c_padded_expand, 1, dummy)
        # feed the node features through a 2-layer mlp
        c_stack_transformed = self.f2(c_stack)
        
        x = torch.cat((rho, c_stack_transformed), 2)
        ## (batch_size, seq_len, 1)
        logm, logv = self.f(x)
        ## (batch_size, seq_len)
        logm, logv = logm.squeeze(2), logv.squeeze(2)
        #m, v = torch.exp(logm), torch.exp(logv)
        ## (batch_size, )
        #m_agg, v_agg = torch.sum(m * w, 1), torch.sum(v * w.pow(2), 1)
        ## parameters of IG distribution
        ## (batch_size, )
        #logμ = torch.log(l) - torch.log(m_agg)
        #logλ = 3*logμ - torch.log(v_agg) - 2*torch.log(l)
        ## (batch_size, )
#         logm_agg = logwsumexp(logm, w)
#         logv_agg = logwsumexp(logv, w.pow(2))
        logm_agg = torch.logsumexp(logm + torch.log(w), dim=1)
        logv_agg = torch.logsumexp(logv + 2*torch.log(w), dim=1)
        logl = torch.log(l)
        ## parameters of IG distribution
        ## (batch_size, )
        logμ = logl - logm_agg
        logλ = logl - 3*logm_agg - logv_agg
        return logμ, logλ

class TTime_spatial(nn.Module):
    def __init__(self, num_u, dim_u, dict_u,
                       num_s1, dim_s1, dict_s1,
                       num_s2, dim_s2, dict_s2,
                       num_s3, dim_s3, dict_s3,
                       dim_rho, dim_c, lengths,
                       hidden_size1, hidden_size2, hidden_size3,
                       dropout, use_selu, device):
        super(TTime_spatial, self).__init__()
        self.probrho = ProbRho(num_u, dim_u, dict_u, lengths,
                               num_s1, dim_s1, dict_s1,
                               num_s2, dim_s2, dict_s2,
                               num_s3, dim_s3, dict_s3,
                               hidden_size1, dim_rho,
                               dropout, use_selu, device)
        self.probtraffic = ProbTraffic_spatial(1, hidden_size2, dim_c,
                                       dropout, use_selu)
        self.probttime = ProbTravelTime_spatial(dim_rho, dim_c, hidden_size3,
                                        dropout, use_selu, device)
        self.device = device

    def forward(self, roads, ratios, T, lon_idx, lat_idx):
        road_lens = self.probrho.roads_length(roads, ratios)
        l = road_lens.sum(dim=1) # the whole trip lengths
        w = road_lens / road_lens.sum(dim=1, keepdim=True) # road weights
        rho = self.probrho(roads)
#         c_map, c, c_mu, c_logvar = self.probtraffic(T)
#         logμ, logλ = self.probttime(rho, c, c_map, w, l, roads, lon_idx, lat_idx)
        c = self.probtraffic(T)
        logμ, logλ = self.probttime(rho, c, w, l, roads, lon_idx, lat_idx)
        return logμ, logλ
    
class ProbTraffic_spatial(nn.Module):
    """
    Modelling the probability of the traffic state `c`
    """
    def __init__(self, n_in, hidden_size, dim_c, dropout, use_selu):
        super(ProbTraffic_spatial, self).__init__()
        conv_layers = [
            nn.Conv2d(n_in, 32, (5, 5), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
#             nn.AvgPool2d(3, padding=(1, 0),count_include_pad=False)
        ]
        self.f1 = nn.Sequential(*conv_layers)
        self.f2 = MLP2(128*2*2, hidden_size, dim_c, dropout, use_selu)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, T):
        """
        Input:
          T (batch_size, nchannel, height, width)
        Output:
          c, mu, logvar (batch_size, dim_c)
        """
        x = self.f1(T)
        return x
    
class ProbTravelTime_spatial(nn.Module):
    def __init__(self, dim_rho, dim_c,
                       hidden_size, dropout, use_selu, device):
        super(ProbTravelTime_spatial, self).__init__()
        self.f = MLP2(dim_rho+dim_c, hidden_size, 1, dropout, use_selu)
        self.f2 = MLP(128, 256, 128, 0.2, True)
        self.device = device

    def forward(self, rho, c, w, l, roads, lon_idx, lat_idx):
        """
        rho (batch_size, seq_len, dim_rho)
        c (n_nodes, dim_c): the traffic state vector sampling from ProbTraffic
        w (batch_size, seq_len): the normalized road lengths
        l (batch_size, ): route or path lengths
        """
        ## (batch_size, seq_len, dim_rho+dim_c)
        c = torch.squeeze(c, 0)
        roads = roads.to(self.device)
        c = c.permute(1, 2, 0)
        c_flatten = torch.flatten(c, end_dim=1)
        idx = [lat_idx[i] * 17 + lon_idx[i] for i in range(roads.size(0))]
#         embed_list = [torch.index_select(c_flatten, 0, idx[i].to(self.device)[1]) for i in range(list(roads.size())[0])]
#         embed_list_traj = [torch.squeeze(item, dim=0) for item in embed_list]
        embed_list = [torch.index_select(c_flatten, 0, idx[i].to(self.device)) for i in range(list(roads.size())[0])]
        embed_list_traj = [torch.mean(item, dim=0) for item in embed_list]
        c_stack = torch.stack(embed_list_traj, 0)
        # feed the node features through a 2-layer mlp
        c_stack_transformed = self.f2(c_stack)
        c_expand = torch.unsqueeze(c_stack_transformed, 1).expand(-1, rho.size(1), -1)
        
        x = torch.cat((rho, c_expand), 2)
        ## (batch_size, seq_len, 1)
        logm, logv = self.f(x)
        ## (batch_size, seq_len)
        logm, logv = logm.squeeze(2), logv.squeeze(2)
        #m, v = torch.exp(logm), torch.exp(logv)
        ## (batch_size, )
        #m_agg, v_agg = torch.sum(m * w, 1), torch.sum(v * w.pow(2), 1)
        ## parameters of IG distribution
        ## (batch_size, )
        #logμ = torch.log(l) - torch.log(m_agg)
        #logλ = 3*logμ - torch.log(v_agg) - 2*torch.log(l)
        ## (batch_size, )
#         logm_agg = logwsumexp(logm, w)
#         logv_agg = logwsumexp(logv, w.pow(2))
        logm_agg = torch.logsumexp(logm + torch.log(w), dim=1)
        logv_agg = torch.logsumexp(logv + 2*torch.log(w), dim=1)
        logl = torch.log(l)
        ## parameters of IG distribution
        ## (batch_size, )
        logμ = logl - logm_agg
        logλ = logl - 3*logm_agg - logv_agg
        return logμ, logλ
    

    

class ProbTraffic_spatial_hybrid(nn.Module):
    """
    Modelling the probability of the traffic state `c`
    """
    def __init__(self, n_in, hidden_size, dim_c, dropout, use_selu):
        super(ProbTraffic_spatial_hybrid, self).__init__()
        conv_layers = [
            nn.Conv2d(n_in, 32, (5, 5), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        self.avgPool = nn.AvgPool2d(7)
        self.f1 = nn.Sequential(*conv_layers)
        self.f2 = MLP2(128*2*2, hidden_size, dim_c, dropout, use_selu)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, T):
        """
        Input:
          T (batch_size, nchannel, height, width)
        Output:
          c, mu, logvar (batch_size, dim_c)
        """
        x_map = self.f1(T)
        x = self.avgPool(x_map)
        mu, logvar = self.f2(x.view(x.size(0), -1))
        
        return x_map, self.reparameterize(mu, logvar), mu, logvar
        
    
class ProbTravelTime_spatial_hybrid(nn.Module):
    def __init__(self, dim_rho, dim_c,
                       hidden_size, dropout, use_selu, device):
        super(ProbTravelTime_spatial_hybrid, self).__init__()
        self.f = MLP2(dim_rho+dim_c, hidden_size, 1, dropout, use_selu)
        self.f2 = MLP(128, 256, 128, 0.2, True)
        self.device = device

    def forward(self, rho, c, c_map, w, l, roads, lon_idx, lat_idx):
        """
        rho (batch_size, seq_len, dim_rho)
        c (n_nodes, dim_c): the traffic state vector sampling from ProbTraffic
        w (batch_size, seq_len): the normalized road lengths
        l (batch_size, ): route or path lengths
        """
        ## (batch_size, seq_len, dim_rho+dim_c)
        c_map = torch.squeeze(c_map, 0)
        roads = roads.to(self.device)
        print(c_map.shape)
        c_map = c_map.permute(1, 2, 0)
        c_flatten = torch.flatten(c_map, end_dim=1)
        idx = [lat_idx[i] * 17 + lon_idx[i] for i in range(roads.size(0))]
        embed_list = [torch.index_select(c_flatten, 0, idx[i].to(self.device)) for i in range(list(roads.size())[0])]
        embed_list_traj = [torch.mean(item, dim=0) for item in embed_list]
        c_stack = torch.stack(embed_list_traj, 0)
#         pad_feature = torch.zeros((1, list(c.size())[1]), device=self.device)
        
#         c_padded  = torch.cat((pad_feature, c), dim=0)
# #         embed_list = [torch.index_select(c_padded, 0, roads[i]) for i in range(list(roads.size())[0])]
# #         c_stack = torch.stack(embed_list, 0)
#         c_padded_expand = torch.unsqueeze(c_padded, 0).expand(roads.size(0), -1, -1)
#         dummy = roads.unsqueeze(2).expand(roads.size(0), roads.size(1), c_padded_expand.size(2))
#         c_stack = torch.gather(c_padded_expand, 1, dummy)
        # feed the node features through a 2-layer mlp
        c_stack_transformed = self.f2(c_stack)
        c_expand = torch.unsqueeze(c_stack_transformed, 1).expand(-1, rho.size(1), -1)
        x = torch.cat([rho, c.expand(*rho.shape[:-1], -1)], 2)
        x = torch.cat((x, c_expand), 2)
        ## (batch_size, seq_len, 1)
        logm, logv = self.f(x)
        ## (batch_size, seq_len)
        logm, logv = logm.squeeze(2), logv.squeeze(2)
        #m, v = torch.exp(logm), torch.exp(logv)
        ## (batch_size, )
        #m_agg, v_agg = torch.sum(m * w, 1), torch.sum(v * w.pow(2), 1)
        ## parameters of IG distribution
        ## (batch_size, )
        #logμ = torch.log(l) - torch.log(m_agg)
        #logλ = 3*logμ - torch.log(v_agg) - 2*torch.log(l)
        ## (batch_size, )
#         logm_agg = logwsumexp(logm, w)
#         logv_agg = logwsumexp(logv, w.pow(2))
        logm_agg = torch.logsumexp(logm + torch.log(w), dim=1)
        logv_agg = torch.logsumexp(logv + 2*torch.log(w), dim=1)
        logl = torch.log(l)
        ## parameters of IG distribution
        ## (batch_size, )
        logμ = logl - logm_agg
        logλ = logl - 3*logm_agg - logv_agg
        return logμ, logλ