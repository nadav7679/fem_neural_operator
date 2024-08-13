import torch
import torch.nn as nn
import firedrake as fd

from classes import *

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print(device)
N = 8192
M, D, depth = 16, 64, 4

N_domain = [8192]
for i, N in enumerate(N_domain):
    print(f"Training N={N}")
    model = BurgersModel(N, M, D, depth, 1, "fourier", device=device)
    model.train(f"data/burgers/samples/N{N}_nu001_T1_samples1200.pt", 500, lr=0.001, device=device)
