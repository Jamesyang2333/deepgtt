
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformer.Models import Encoder, PositionalEncoding
from transformer.Layers import EncoderLayer

#use_cuda = False
#device = torch.device("cuda" if use_cuda else "cpu")

# generate the padding mask for transformer encoder
def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


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
    eps = 1e-9
    maxv, _ = torch.max(x, dim=1, keepdim=True)
    y = torch.log(torch.sum(torch.exp(x - maxv) * w, dim=1, keepdim=True) + eps) + maxv
    return y.squeeze(1)

class ProbRho(nn.Module):
    """
    s1 (14): road type
    s2 (7): number of lanes
    s3 (2): one way or not
    num_sx: the number of categories for attribute x.
    u: id of the road (path segment)
    """
    def __init__(self, num_u, dim_u, dict_u, lengths,
                       num_s1, dim_s1, dict_s1,
                       num_s2, dim_s2, dict_s2,
                       num_s3, dim_s3, dict_s3,
                       hidden_size, n_layers, 
                       d_inner, n_head, d_k, d_v, dim_rho,
                       dropout, use_selu, device, n_position=200):
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
        
        # dimensional of the path segment embedding
        dim_embedding = dim_u+dim_s1+dim_s2+dim_s3
        # transformer
        self.position_enc = PositionalEncoding(dim_embedding, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(dim_embedding, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(dim_embedding, eps=1e-6)
        
        self.f = MLP2(dim_embedding,
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

    def forward(self, roads, src_mask=None):
        """
        roads (batch_size * seq_len)
        """
        u  = self.embedding_u(self.roads2u(roads).to(self.device))
        s1 = self.embedding_s1(self.roads_s_i(roads, self.dict_s1).to(self.device))
        s2 = self.embedding_s2(self.roads_s_i(roads, self.dict_s2).to(self.device))
        s3 = self.embedding_s3(self.roads_s_i(roads, self.dict_s3).to(self.device))
        # x (batch * seq_len * (dim_u+dim_s1+dim_s2+dim_s3)
        x  = torch.cat([u, s1, s2, s3], dim=2)
        
        # transformer
        enc_output = self.dropout(self.position_enc(x))
        enc_output = self.layer_norm(enc_output)
        
        # generate padding mask for the encoder
        src_mask = get_pad_mask(self.roads2u(roads).to(self.device), 0)
        
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
        mu, logvar = self.f(enc_output)
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
        # for chengdu dataset
#         conv_layers = [
#             nn.Conv2d(n_in, 32, (3, 3), padding=1),
#             nn.MaxPool2d((2, 2)),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(32, 64, (3, 3), padding=1),
#             nn.MaxPool2d((2, 2)),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(64, 128, (3, 3), padding=1),
#             nn.MaxPool2d((2, 2)),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1, inplace=True),
#         ]
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
        ## logm: log of speed; log v: log of variance
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
#         logv_agg = logwsumexp(logv, w)
#         logv_agg = logwsumexp(logv, w.pow(2))
        logm_agg = torch.logsumexp(logm + torch.log(w), dim=1)
        logv_agg = torch.logsumexp(logv + 2*torch.log(w), dim=1)
    
        logl = torch.log(l)
        ## parameters of IG distribution
        ## (batch_size, )
        logμ = logl - logm_agg
        logλ = logl - 3*logm_agg - logv_agg
        return logμ, logλ, [logm, logv, logm_agg, logv_agg, l, w]



class TTime_combine(nn.Module):
    # note that it should satisfy n_head * d_v = dim_u + dim_s1 + dim_s2 + dim_s3
    def __init__(self, num_u, dim_u, dict_u,
                       num_s1, dim_s1, dict_s1,
                       num_s2, dim_s2, dict_s2,
                       num_s3, dim_s3, dict_s3, 
                       n_layers, d_inner, n_head, d_k, d_v,
                       dim_rho, dim_c, lengths,
                       hidden_size1, hidden_size2, hidden_size3,
                       dropout, use_selu, device):
        super(TTime_combine, self).__init__()
        self.probrho = ProbRho_transformer(num_u, dim_u, dict_u, lengths,
                               num_s1, dim_s1, dict_s1,
                               num_s2, dim_s2, dict_s2,
                               num_s3, dim_s3, dict_s3,
                               hidden_size1, n_layers, d_inner, 
                               n_head, d_k, d_v, dim_rho,
                               dropout, use_selu, device)
        self.probtraffic = ProbTraffic(1, hidden_size2, dim_c,
                                       dropout, use_selu)
        self.probttime = ProbTravelTime(dim_rho, dim_c, hidden_size3,
                                        dropout, use_selu)

    def forward(self, roads, ratios,T):
        road_lens = self.probrho.roads_length(roads, ratios)
        l = road_lens.sum(dim=1) # the whole trip lengths
        w = road_lens / road_lens.sum(dim=1, keepdim=True) # road weights
        rho = self.probrho(roads)
        c = self.probtraffic(T.to(device))
        logμ, logλ = self.probttime(rho, c, w, l)
        return logμ, logλ, mu_c, logvar_c

  


