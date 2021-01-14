
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from data_osmnx_utils import DataLoader
from model import *
import db_osmnx_utils as db_utils, argparse, os, time
from collections import namedtuple
from Optim import ScheduledOptim
import itertools  

##  python ~/travel-model/travel-modelling/harbin/python/train.py -trainpath /home/xiucheng/travel-model/data-7min/traindata -validpath /home/xiucheng/travel-model/data-7min/validdata -kl_decay 0.0 -use_selu -random_emit

parser = argparse.ArgumentParser(description="train.py")

parser.add_argument("-trainpath", help="Path to train data",
    default="/home/xiucheng/data-backup/bigtable/2015-taxi/data/traindata")

parser.add_argument("-validpath", help="Path to validate data",
    default="/home/xiucheng/data-backup/bigtable/2015-taxi/data/validdata")

parser.add_argument("-dim_u", type=int, default=200,
    help="The dimension of embedding u")

parser.add_argument("-dim_rho", type=int, default=256,
    help="The dimension of rho (road representation)")

parser.add_argument("-dim_c", type=int, default=128,
    help="The dimension of c (traffic state representation)")

parser.add_argument("-dropout", type=float, default=0.2,
    help="The dropout probability")

parser.add_argument("-batch_size", type=int, default=150,
    help="The batch size")

parser.add_argument("-num_epoch", type=int, default=10,
    help="The number of epoch")

parser.add_argument("-max_grad_norm", type=float, default=0.1,
    help="The maximum gradient norm")

parser.add_argument("-lr", type=float, default=0.001,
    help="Learning rate")

parser.add_argument("-lr_decay", type=float, default=0.2,
    help="Learning rate decay")

parser.add_argument("-kl_decay", type=float, default=0,
    help="KL Divergence decay")

parser.add_argument("-print_freq", type=int, default=1000,
    help="Print frequency")

parser.add_argument("-use_cuda", type=bool, default=True)

parser.add_argument("-use_selu", action="store_true")

parser.add_argument("-random_emit", action="store_true")

parser.add_argument("-model_path", type=str, default="../models/best-model",
    help="path to save model")

parser.add_argument("-use_gnn", type=bool, default=False)


args = parser.parse_args()
print(args)

device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed_all(12)
##############################################################################

def log_pdf(logμ, logλ, t):
    """
    log pdf of IG distribution.
    Input:
      logμ, logλ (batch_size, ): the parameters of IG
      t (batch_size, ): the travel time
    ------
    Output:
      logpdf (batch_size, )
    """
    eps = 1e-9
#     print("max / min value of log mu in batch " + str(torch.max(logμ)) + " " + str(torch.min(logμ)))
#     print("max / min value of log lambda in batch " + str(torch.max(logλ)) + " " + str(torch.min(logλ)))
    μ = torch.exp(logμ)
#     print("max / min value of mu in batch " + str(torch.max(μ)) + " " + str(torch.min(μ)))
    if ((μ.pow(2)*t+eps)==0).any():
        idx = ((μ.pow(2)*t+eps)==0).nonzero()
        print(μ[idx])
        print(t[idx])
        print(eps[idx])
#     print("max / min value of numerator in batch " + str(torch.max(-0.5 * torch.exp(logλ)*torch.pow(t-μ,2))) + " " + str(torch.min(-0.5 * torch.exp(logλ)*torch.pow(t-μ,2))))
#     print("max / min value of denominator in batch " + str(torch.max(μ.pow(2)*t+eps)) + " " + str(torch.min(μ.pow(2)*t+eps)))
    expt = -0.5 * torch.exp(logλ)*torch.pow(t-μ,2) / (μ.pow(2)*t+eps)
    logz = 0.5*logλ - 1.5*torch.log(t)
    return expt+logz

def log_prob_loss(logμ, logλ, t):
    """
    logμ, logλ (batch_size, ): the parameters of IG
    t (batch_size, ): the travel time
    ---
    Return the average loss of the log probability of a batch of data
    """
    logpdf = log_pdf(logμ, logλ, t)
    return torch.mean(-logpdf)

def KLD(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]

def adjust_lr(optimizer, epoch):
    lr = args.lr * (args.lr_decay ** (epoch//3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#############################################################################

## Parameters
dim_s1, dim_s2, dim_s3 = 64, 32, 16
hidden_size1, hidden_size2, hidden_size3 = 300, 500, 600
## Fetching metadata from database
lengths = db_utils.get_lengths()
dict_u, num_u = db_utils.get_dict_u()
dict_s1, num_s1 = db_utils.get_dict_s1()
dict_s2, num_s2 = db_utils.get_dict_s2()
dict_s3, num_s3 = db_utils.get_dict_s3()

## Model
TTime_spatial = TTime_spatial(num_u, args.dim_u, dict_u,
                       num_s1, dim_s1, dict_s1,
                       num_s2, dim_s2, dict_s2,
                       num_s3, dim_s3, dict_s3, 
                       args.dim_rho, args.dim_c, lengths,
                       hidden_size1, hidden_size2, hidden_size3,
                       args.dropout, args.use_selu, device).to(device)

Params = namedtuple("params", ["num_u", "dim_u", "dict_u", "lengths", "num_s1",
                               "dim_s1", "dict_s1", "num_s2", "dim_s2", "dict_s2",
                               "num_s3", "dim_s3", "dict_s3", "dim_rho","dim_c",
                               "hidden_size2", "hidden_size3", "hidden_size1",
                               "dropout", "use_selu", "model_path", "use_gnn"])
params = Params(num_u=num_u, dim_u=args.dim_u, dict_u=dict_u, lengths=lengths,
                num_s1=num_s1, dim_s1=dim_s1, dict_s1=dict_s1, num_s2=num_s2,
                dim_s2=dim_s2, dict_s2=dict_s2, num_s3=num_s3, dim_s3=dim_s3,
                dict_s3=dict_s3, dim_rho=args.dim_rho, dim_c=args.dim_c,
                hidden_size1=hidden_size1, hidden_size2=hidden_size2, hidden_size3=hidden_size3,
                dropout=args.dropout, use_selu=args.use_selu, model_path=args.model_path, use_gnn=args.use_gnn)


## Optimizer
optimizer_ttime = torch.optim.Adam(TTime_spatial.parameters(), lr=args.lr, amsgrad=True)
# optimizer_ttime = ScheduledOptim(torch.optim.Adam(TTime_gnn.parameters(), betas=(0.9, 0.98), eps=1e-9, amsgrad=False), 1, 128, 8000)


## Preparing the data
trainfiles = list(filter(lambda x:x.endswith(".h5"),
                         sorted(os.listdir(args.trainpath))))
validfiles = list(filter(lambda x:x.endswith(".h5"),
                         sorted(os.listdir(args.validpath))))
train_dataloader = DataLoader(args.trainpath)
print("Loading the training data...")
train_dataloader.read_files(trainfiles, args.use_gnn)
valid_dataloader = DataLoader(args.validpath)
print("Loading the validation data...")
valid_dataloader.read_files(validfiles, args.use_gnn)
train_slot_size = np.array(list(map(lambda s:s.ntrips, train_dataloader.slotdata_pool)))
train_num_iterations = int(np.ceil(train_slot_size/args.batch_size).sum())
print("There are {} trips in the training dataset".format(train_slot_size.sum()))
print("Number of iterations for an epoch: {}".format(train_num_iterations))
valid_slot_size = np.array(list(map(lambda s:s.ntrips, valid_dataloader.slotdata_pool)))
valid_num_iterations = int(np.ceil(valid_slot_size/args.batch_size).sum())

def validate(num_iterations):
    TTime_spatial.eval()

    with torch.no_grad():
        total_loss, total_mse = 0.0, 0.0
        for _ in range(num_iterations):
            data = valid_dataloader.order_emit(args.batch_size)
            logμ, logλ = TTime_spatial(data.trips, data.ratios, data.S.to(device),  data.lon_idx, data.lat_idx)
#             logμ, logλ, mu_c, logvar_c = TTime_gnn(data.trips, data.ratios, data.links.to(device), train_dataloader.get_adj().to(device))
            times = data.times.to(device)
            loss = log_prob_loss(logμ, logλ, times)
#             loss = log_prob_loss(logμ, logλ, times) + args.kl_decay*KLD(mu_c, logvar_c)
            total_loss += loss.item() * data.trips.shape[0]
            total_mse += F.mse_loss(torch.exp(logμ), times).item() * data.trips.shape[0]
        mean_loss, mean_mse = total_loss/np.sum(valid_slot_size), total_mse/np.sum(valid_slot_size)
        print("Validation Loss {0:.4f} MSE {1:.4f}".format(mean_loss, mean_mse))
    TTime_spatial.train()
    return mean_loss, mean_mse


def train(num_iterations=1000):
    epoch_loss, stage_loss, epoch_mse, stage_mse = 0., 0., 0., 0.
    for it in range(1, num_iterations+1):
        ## Loading the data
        if args.random_emit == True:
            data = train_dataloader.random_emit(args.batch_size)
        else:
            data = train_dataloader.order_emit(args.batch_size)
        with torch.autograd.detect_anomaly():
            ## forward computation
            if it==1:
                lon_idx = [x.tolist() for x in data.lon_idx]
                print(set(list(itertools.chain.from_iterable(lon_idx))))
                lat_idx = [x.tolist() for x in data.lat_idx]
                print(set(list(itertools.chain.from_iterable(lat_idx))))
            logμ, logλ = TTime_spatial(data.trips, data.ratios, data.S.to(device), data.lon_idx, data.lat_idx)
#             logμ, logλ, mu_c, logvar_c = TTime_gnn(data.trips, data.ratios, data.links.to(device), train_dataloader.get_adj().to(device))
#             logμ = torch.div(logμ, 50.0)
#             logλ = torch.div(logλ, 50.0)
            ## move to gpu
            times = data.times.to(device)
            loss = log_prob_loss(logμ, logλ, times)
#             loss = log_prob_loss(logμ, logλ, times) + args.kl_decay*KLD(mu_c, logvar_c)
            epoch_loss += loss.item()
            stage_loss += loss.item()
            ## Measuring the mean square error
            mse = F.mse_loss(torch.exp(logμ), times)
            epoch_mse += mse.item()
            stage_mse += mse.item()
            if it % args.print_freq == 0:
                print("Stage MSE: {0:.4f} Loss {1:.4f} at epoch {2:} iteration {3:}".format\
                      (stage_mse/args.print_freq, stage_loss/args.print_freq, epoch, it))
                stage_mse = 0
                stage_loss = 0
            ## backward optimization
            optimizer_ttime.zero_grad()
            loss.backward()
            ## optimizing
            clip_grad_norm_(TTime_spatial.parameters(), args.max_grad_norm)
            optimizer_ttime.step()
#             optimizer_ttime.step_and_update_lr()
    print("\nEpoch Loss: {0:.4f}".format(epoch_loss / num_iterations))
    print("Epoch MSE: {0:.4f}".format(epoch_mse / num_iterations))

tic = time.time()
min_mse = 1e9
for epoch in range(1, args.num_epoch+1):
    print("epoch {} =====================================>".format(epoch))
    train(train_num_iterations)
    mean_loss, mean_mse = validate(valid_num_iterations)
    if mean_mse < min_mse:
        print("Saving model...")
        torch.save({
            "TTime_gnn": TTime_spatial.state_dict(),
            "params": params._asdict()
        }, args.model_path + ".pt")
        min_mse = mean_mse
    adjust_lr(optimizer_ttime, epoch)
cost = time.time() - tic
print("Time passed: {} hours".format(cost/3600))