import torch
import torch.nn as nn
import firedrake as fd

from classes import *

device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device("cpu")
print(device)
M, D, depth = 8, 20, 3

for N in [64, 128, 512, 1024, 4096]:
    for T in ["01", "02", "03", "04", "05"]:
        model = KSModel(N, M, D, depth, T, "fourier", "HER", device=device)
        
        # optimizer = torch.optim.Adam(model.network.parameters(), lr=0.01)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 10E-05, 0.01, 100, 100)

        model.train(f"data/KS/samples/N{N}_HER_nu0029_T{T}_samples1200.pt", 500, device=device)
