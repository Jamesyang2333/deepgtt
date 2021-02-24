import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import db_osmnx_utils as db_utils
import torch
from model import *

with h5py.File("/Project0551/jingyi/deepgtt/data/trainpath-fmm-spatial/150103.h5") as f:
    S = np.transpose(f["/1/S"]).copy()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(fname):
    model = torch.load(fname)
    params = model["params"]
    probrho = ProbRho(params["num_u"], params["dim_u"], params["dict_u"], params["lengths"],
                      params["num_s1"], params["dim_s1"], params["dict_s1"],
                      params["num_s2"], params["dim_s2"], params["dict_s2"],
                      params["num_s3"], params["dim_s3"], params["dict_s3"],
                      params["hidden_size1"], params["dim_rho"],
                      params["dropout"], params["use_selu"], device).to(device)
#     probtraffic = ProbTraffic_reconstruct(1, params["hidden_size2"], params["dim_c"],
#                               params["dropout"], params["use_selu"]).to(device)
    probtraffic = ProbTraffic_reconstruct_mask(1, params["hidden_size2"], params["dim_c"],
                              params["dropout"], params["use_selu"], db_utils.get_map_mask()).to(device)
    probttime = ProbTravelTime(params["dim_rho"], params["dim_c"], params["hidden_size3"],
                               params["dropout"], params["use_selu"]).to(device)
    probrho.load_state_dict(model["probrho"])
    probtraffic.load_state_dict(model["probtraffic"])
    probttime.load_state_dict(model["probttime"])
    return probrho, probtraffic, probttime

probrho, probtraffic, probttime = load_model('/Project0551/jingyi/deepgtt/model/mlp-spatial-recon-sup-mask-0.01-new-1.pt')
probrho.eval()
probtraffic.eval()
probttime.eval()
s = torch.tensor(S, dtype=torch.float32)
print(s.shape)
s.unsqueeze_(0).unsqueeze_(0)
c, mu_c, logvar_c, s_comp = probtraffic(s.to(device))
s_numpy = s_comp.detach().cpu().numpy()

with open('./complete_map.npy', 'wb') as f:
    np.save(f, s_numpy)
