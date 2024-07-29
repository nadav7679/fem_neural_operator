import torch
import torch.nn as nn
import firedrake as fd

from classes import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

N = 2048
M, D, depth = 10, 20, 3

model = KSModel(N, M, D, depth, "fourier", "HER", device=device)
model.train(f"data/KS/samples/N{N}_HER_nu0029_T01_samples1200.pt", 10, device=device)
