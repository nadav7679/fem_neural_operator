import torch
import torch.nn as nn
import firedrake as fd

from classes import *

device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
print(device)
N = 8192
M, D, depth = 8, 20, 3

for M in [2 ** j for j in range(3, 10)]:
    for D in [i * 10 for i in range(1, 7)]:
        print(f"Training M={M}, D={D}")
        model = BurgersModel(N, M, D, depth, 1, "fourier", device=device)
        model.train(f"data/burgers/samples/N{N}_nu001_T1_samples1200.pt", 500, device=device)
