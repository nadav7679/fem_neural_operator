import torch
import torch.nn as nn
import firedrake as fd

from classes import *

device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
print(device)
M, D, depth = 16, 64, 4

for N in [128, 256]:
    for T in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]:
        model = KSModel(N, M, D, depth, T, "fourier", "CG3", device=device)
        
        # optimizer = torch.optim.Adam(model.network.parameters(), lr=0.01)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 10E-05, 0.01, 100, 100)
        print(f"Training N={N} T={T}")
        model.train(f"data/KS/samples/N{N}_CG3_nu0029_T{T}_samples1200.pt", 500, device=device)


# samples = torch.load(f"data/KS/samples/N{64}_HER_nu0029_T01_samples1200.pt").unsqueeze(2).to(device=device, dtype=torch.float32)
# grid = torch.linspace(0, 10, 2 * 64, device=device)
# print("\n\n\n")
# print(samples.shape)
# print(samples[0])
# trainset = KSDataset(64, torch.tensor(samples[:1000]), torch.tensor(grid))
# print(trainset[0])
# testset = KSDataset(64, torch.tensor(samples[1000:]), torch.tensor(grid))
