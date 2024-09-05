import torch
import matplotlib.pyplot as plt
from firedrake import *

from classes import *


device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")

T_domain = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
N_domain = [64, 128, 256, 512, 1024, 2048]
N_losses = [[], [], [], [], [], []]
for i, N in enumerate(N_domain):
    for T in T_domain:
        model = KSModel.load(f"data/KS/models/CG3/fourier/N{N}/T{T}/D64_M16_samples1000_epoch750.pt", N, T, device)
        
        data_path = f"data/KS/samples/N{N}_CG3_nu0029_T{T}_samples1200.pt"
        samples = torch.load(data_path).unsqueeze(2).to(device=device, dtype=torch.float32)

        cg3 = FunctionSpace(model.mesh, "CG", 3)
        x = assemble(interpolate(SpatialCoordinate(model.mesh)[0], cg3))
        grid = torch.tensor(sorted(x.dat.data[:])).to(device=device, dtype=torch.float32)

        trainset = Dataset(torch.tensor(samples[:model.train_samples]), torch.tensor(grid))
        testset = Dataset(torch.tensor(samples[model.train_samples:]), torch.tensor(grid))
        
        mean_rel_l2_loss = lambda x, y: torch.mean(torch.norm(x - y, 2, dim=-1)/torch.norm(y, 2, dim=-1))
        
        optimizer = torch.optim.Adam(model.network.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5)
        
        network_trainer = NeuralNetworkTrainer(
            model,
            trainset,
            testset,
            mean_rel_l2_loss,
            optimizer,
            scheduler,
            max_epoch=500
        )
        
        N_losses[i].append(network_trainer.test_epoch().detach().cpu())



N_losses = torch.tensor(N_losses)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)

for i, losses in enumerate(N_losses):
    axes[0].plot(T_domain, losses, label=f"N = {N_domain[i]}")
    
axes[0].legend()
axes[0].set_xlabel("T $[10 \, X \, dt]$")
axes[0].set_ylabel("RelL2")
axes[0].grid()
# axes[0].title("KS RelH1 vs T for different meshes")
        
axes[1].plot(T_domain, torch.mean(N_losses, dim=0), label="Average over $N$", color="black")
axes[1].set_xlabel("T $[10 \, X \, dt]$")
axes[1].grid()
axes[1].legend()

plt.savefig("KS RelL2 vs T for different meshes 750e")