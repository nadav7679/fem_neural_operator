import torch
import torch.nn as nn
import firedrake as fd

from classes import *

device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")
print(device)
M, D, depth = 16, 64, 4

for N in [4096, 64, 128, 256, 512, 1024, 2048]:
    for T in ["08", "09", "10", "11", "12", "13", "14", "15"]:
        model = KSModel(N, M, D, depth, T, "fourier", "HER", device=device)
        
        # optimizer = torch.optim.Adam(model.network.parameters(), lr=0.01)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 10E-05, 0.01, 100, 100)
        print(f"Training N={N} T={T}")
        model.train(f"data/KS/samples/N{N}_HER_nu0029_T{T}_samples1200.pt", 500, device=device)
