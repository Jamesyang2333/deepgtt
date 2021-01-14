
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import numpy as np
from data_utils import DataLoader
from model import *
import db_osmnx_utils as db_utils, argparse, os, time
from collections import namedtuple
from transformer.Optim import ScheduledOptim
import math

##  python ~/travel-model/travel-modelling/harbin/python/train.py -trainpath /home/xiucheng/travel-model/data-7min/traindata -validpath /home/xiucheng/travel-model/data-7min/validdata -kl_decay 0.0 -use_selu -random_emit

parser = argparse.ArgumentParser(description="train.py")

parser.add_argument("-trainpath", help="Path to train data",
    default="/home/xiucheng/data-backup/bigtable/2015-taxi/data/traindata")

parser.add_argument("-validpath", help="Path to validate data",
    default="/home/xiucheng/data-backup/bigtable/2015-taxi/data/validdata")

# parser.add_argument("-dim_u", type=int, default=200,
#     help="The dimension of embedding u")
parser.add_argument("-dim_u", type=int, default=256,
    help="The dimension of embedding u")

parser.add_argument("-dim_s1", type=int, default=128,
    help="The dimension of embedding s1")

parser.add_argument("-dim_s2", type=int, default=64,
    help="The dimension of embedding s2")

parser.add_argument("-dim_s3", type=int, default=64,
    help="The dimension of embedding s3")

parser.add_argument("-n_layers", type=int, default=6,
    help="Number of self-attention layers")

parser.add_argument("-d_inner", type=int, default=2048,
    help="Number of hidden units in transformer feedforward layer")

parser.add_argument("-n_head", type=int, default=8,
    help="Number of head")

parser.add_argument("-d_k", type=int, default=64,
    help="The dimension of self-attention key")

parser.add_argument("-d_v", type=int, default=64,
    help="The dimension of self-attention value")

parser.add_argument("-dim_rho", type=int, default=256,
    help="The dimension of rho (road representation)")

parser.add_argument("-dim_c", type=int, default=400,
    help="The dimension of c (traffic state representation)")

parser.add_argument("-dropout", type=float, default=0.2,
    help="The dropout probability")

parser.add_argument("-batch_size", type=int, default=150,
    help="The batch size")

parser.add_argument("-num_epoch", type=int, default=30,
    help="The number of epoch")

parser.add_argument("-max_grad_norm", type=float, default=0.1,
    help="The maximum gradient norm")

parser.add_argument("-lr", type=float, default=0.5,
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

parser.add_argument("-n_warmup_steps", type=int, default=8000,
    help="number of warmup steps")

parser.add_argument("-use_subtraj", type=bool, default=False)

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
    μ = torch.exp(logμ)
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
    return torch.mean(-logpdf), logpdf

def KLD(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]

def adjust_lr(optimizer, epoch):
    lr = args.lr * (args.lr_decay ** (epoch//3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#############################################################################

## Parameters
# dim_s1, dim_s2, dim_s3 = 64, 32, 16
hidden_size1, hidden_size2, hidden_size3 = 300, 500, 600
## Fetching metadata from database
lengths = db_utils.get_lengths()
dict_u, num_u = db_utils.get_dict_u()
dict_s1, num_s1 = db_utils.get_dict_s1()
dict_s2, num_s2 = db_utils.get_dict_s2()
dict_s3, num_s3 = db_utils.get_dict_s3()

## Model
probrho = ProbRho(num_u, args.dim_u, dict_u, lengths,
                  num_s1, args.dim_s1, dict_s1,
                  num_s2, args.dim_s2, dict_s2,
                  num_s3, args.dim_s3, dict_s3,
                  hidden_size1, args.n_layers, args.d_inner, 
                  args.n_head, args.d_k, args.d_v,args.dim_rho,
                  args.dropout, args.use_selu, device).to(device)
probtraffic = ProbTraffic(1, hidden_size2, args.dim_c,
                          args.dropout, args.use_selu).to(device)
probttime = ProbTravelTime(args.dim_rho, args.dim_c, hidden_size3,
                           args.dropout, args.use_selu).to(device)

Params = namedtuple("params", ["num_u", "dim_u", "dict_u", "lengths", "num_s1",
                               "dim_s1", "dict_s1", "num_s2", "dim_s2", "dict_s2",
                               "num_s3", "dim_s3", "dict_s3", "n_layers",          "d_inner", 
                "n_head", "d_k", "d_v","dim_rho","dim_c",
                               "hidden_size2", "hidden_size3", "hidden_size1",
                               "dropout", "use_selu", "model_path", "n_warmup_steps", "use_subtraj"])
params = Params(num_u=num_u, dim_u=args.dim_u, dict_u=dict_u, lengths=lengths,
                num_s1=num_s1, dim_s1=args.dim_s1, dict_s1=dict_s1, num_s2=num_s2,
                dim_s2=args.dim_s2, dict_s2=dict_s2, num_s3=num_s3, dim_s3=args.dim_s3, dict_s3=dict_s3, n_layers=args.n_layers, d_inner=args.d_inner, 
                n_head=args.n_head, d_k=args.d_k, d_v=args.d_v, 
                dim_rho=args.dim_rho, dim_c=args.dim_c,
                hidden_size1=hidden_size1, hidden_size2=hidden_size2, hidden_size3=hidden_size3,
                dropout=args.dropout, use_selu=args.use_selu, model_path=args.model_path, n_warmup_steps=args.n_warmup_steps, use_subtraj=args.use_subtraj)

dim_embedding = args.dim_u+args.dim_s1+args.dim_s2+args.dim_s3

## Optimizer
# optimizer_rho = torch.optim.Adam(probrho.parameters(), lr=args.lr, amsgrad=True)
# optimizer_traffic = torch.optim.Adam(probtraffic.parameters(), lr=args.lr, amsgrad=True)
# optimizer_ttime = torch.optim.Adam(probttime.parameters(), lr=args.lr, amsgrad=True)
optimizer_rho = ScheduledOptim(optim.Adam(probrho.parameters(), betas=(0.9, 0.98), eps=1e-9, amsgrad=False), args.lr, dim_embedding, args.n_warmup_steps)
optimizer_traffic = ScheduledOptim(optim.Adam(probtraffic.parameters(), betas=(0.9, 0.98), eps=1e-9, amsgrad=False), args.lr, dim_embedding, args.n_warmup_steps)
optimizer_ttime = ScheduledOptim(optim.Adam(probttime.parameters(), betas=(0.9, 0.98), eps=1e-9, amsgrad=False), args.lr, dim_embedding, args.n_warmup_steps)



## Preparing the data
trainfiles = list(filter(lambda x:x.endswith(".h5"),
                         sorted(os.listdir(args.trainpath))))
validfiles = list(filter(lambda x:x.endswith(".h5"),
                         sorted(os.listdir(args.validpath))))
train_dataloader = DataLoader(args.trainpath)
print("Loading the training data...")
# set enable_subtraj to true to train with subtrajectories
train_dataloader.read_files(trainfiles, args.use_subtraj)
valid_dataloader = DataLoader(args.validpath)
print("Loading the validation data...")
valid_dataloader.read_files(validfiles)
train_slot_size = np.array(list(map(lambda s:s.ntrips, train_dataloader.slotdata_pool)))
print("train_slot_size")
print(train_slot_size)
print("batch_size")
print(args.batch_size)
train_num_iterations = int(np.ceil(train_slot_size/args.batch_size).sum())
if args.use_subtraj:
    train_num_iterations = int(np.ceil(train_slot_size/args.batch_size*2).sum())
print("There are {} trips in the training dataset".format(train_slot_size.sum()))
print("Number of iterations for an epoch: {}".format(train_num_iterations))
valid_slot_size = np.array(list(map(lambda s:s.ntrips, valid_dataloader.slotdata_pool)))
valid_num_iterations = int(np.ceil(valid_slot_size/args.batch_size).sum())

def validate(num_iterations):
    probrho.eval()
    probtraffic.eval()
    probttime.eval()

    with torch.no_grad():
        total_loss, total_mse = 0.0, 0.0
        for _ in range(num_iterations):
            data = valid_dataloader.order_emit(args.batch_size)
            road_lens = probrho.roads_length(data.trips, data.ratios)
            l = road_lens.sum(dim=1) # the whole trip lengths
#             if _ == 1:
#                 print(data.trips[0])
#                 print(data.ratios[0])
#                 print(data.times[0])
#                 print(road_lens[0])
#                 print(data.trips[1])
#                 print(data.ratios[1])
#                 print(data.times[1])
#                 print(road_lens[1])
            w = road_lens / road_lens.sum(dim=1, keepdim=True) # road weights
            rho = probrho(data.trips)
            c, mu_c, logvar_c = probtraffic(data.S.to(device))
            logμ, logλ, _ = probttime(rho, c, w, l)
            times = data.times.to(device)
            prob_loss, prob_loss_all = log_prob_loss(logμ, logλ, times)
            loss = prob_loss + args.kl_decay*KLD(mu_c, logvar_c)
            total_loss += loss.item() * data.trips.shape[0]
            total_mse += F.mse_loss(torch.exp(logμ), times).item() * data.trips.shape[0]
        mean_loss, mean_mse = total_loss/np.sum(valid_slot_size), total_mse/np.sum(valid_slot_size)
        print("Validation Loss {0:.4f} MSE {1:.4f}".format(mean_loss, mean_mse))
    probrho.train()
    probtraffic.train()
    probttime.train()
    return mean_loss, mean_mse


def train(num_iterations=1000, epoch=30):
    epoch_loss, epoch_mse, stage_mse = 0., 0., 0.
    prev_record=[]
    found_nan = False
    for it in range(1, num_iterations+1):
        ## Loading the data
        if args.random_emit == True:
            if args.use_subtraj:
                data = train_dataloader.random_emit_sub(args.batch_size)
            else:
                data = train_dataloader.random_emit(args.batch_size)
        else:
            data = train_dataloader.order_emit(args.batch_size)
        with torch.autograd.detect_anomaly():
            ## forward computation
            road_lens = probrho.roads_length(data.trips, data.ratios)
            l = road_lens.sum(dim=1) # the whole trip lengths
            w = road_lens / road_lens.sum(dim=1, keepdim=True) # road weights
#             if it == 1:
#                 print(data.trips[0])
#                 print(data.ratios[0])
#                 print(data.times[0])
#                 print(data.offset[0])
#                 print(road_lens[0])
#                 print(data.trips[75])
#                 print(data.ratios[75])
#                 print(data.times[75])
#                 print(road_lens[75])
#                 print(data.info[0])
            rho = probrho(data.trips)
            c, mu_c, logvar_c = probtraffic(data.S.to(device))
            logμ, logλ, misc = probttime(rho, c, w, l)
            ## move to gpu
            times = data.times.to(device)
            prob_loss, prob_loss_all = log_prob_loss(logμ, logλ, times)
            loss = prob_loss + args.kl_decay*KLD(mu_c, logvar_c)
    #         loss = prob_loss
            epoch_loss += loss.item()

            ## Measuring the mean square error
            mse = F.mse_loss(torch.exp(logμ), times)
            epoch_mse += mse.item()
            stage_mse += mse.item()
            if it % args.print_freq == 0:
                print("Stage MSE: {0:.4f} at epoch {1:} iteration {2:}".format\
                      (stage_mse/args.print_freq, epoch, it))
                stage_mse = 0

            mses = torch.mul(torch.exp(logμ)-times, torch.exp(logμ)-times)
    #         for i in range(list(rho.size())[0]):
    #             if epoch >= 5 and (mses[i] > 2500 or torch.isnan(torch.exp(logμ)[i])):
    #                 if i < 75:
    #                     print(data.trips[i])
    #                     print(data.times[i])
    #                     print(torch.exp(logμ)[i])
    #                     print(data.info[i])
    #                     print(data.trips[i + 75])
    #                     print(data.times[i + 75])
    #                 else:
    #                     print(data.trips[i])
    #                     print(data.times[i])
    #                     print(torch.exp(logμ)[i])
    #                     print(data.info[i - 75])
    #                     print(data.trips[i - 75])
    #                     print(data.times[i - 75])

#             for i in range(list(rho.size())[0]):
#                 if torch.isnan(prob_loss_all[i]):
#                     print("loss term nan")
#                     print(logμ[i])
#                     print(logλ[i])
#                     print(data.trips[i])
#                     print(data.times[i])
#                     print(torch.exp(logμ)[i])
#                     for item in misc:
#                         print(item[i])
#                     exit()

    #         prev_record = [data.trips[i], data.times[i], torch.exp(logμ)[i], misc]

            ## backward optimization
            optimizer_rho.zero_grad()
            optimizer_traffic.zero_grad()
            optimizer_ttime.zero_grad()
            loss.backward()
            clip_grad_norm_(probrho.parameters(), args.max_grad_norm)
            clip_grad_norm_(probtraffic.parameters(), args.max_grad_norm)
            clip_grad_norm_(probttime.parameters(), args.max_grad_norm)
            optimizer_rho.step_and_update_lr()
            optimizer_traffic.step_and_update_lr()
            optimizer_ttime.step_and_update_lr()
    print("\nEpoch Loss: {0:.4f}".format(epoch_loss / num_iterations))
    print("Epoch MSE: {0:.4f}".format(epoch_mse / num_iterations))
    

tic = time.time()
min_mse = 1e9
for epoch in range(1, args.num_epoch+1):
    print("epoch {} =====================================>".format(epoch))
    train(train_num_iterations, epoch)
    mean_loss, mean_mse = validate(valid_num_iterations)
    if mean_mse < min_mse:
        print("Saving model...")
        torch.save({
            "probrho": probrho.state_dict(),
            "probtraffic": probtraffic.state_dict(),
            "probttime": probttime.state_dict(),
            "params": params._asdict()
        }, args.model_path + ".pt")
        min_mse = mean_mse
#     adjust_lr(optimizer_rho, epoch)
#     adjust_lr(optimizer_traffic, epoch)
#     adjust_lr(optimizer_ttime, epoch)
cost = time.time() - tic
print("Time passed: {} hours".format(cost/3600))
