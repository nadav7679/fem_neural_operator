import torch
from firedrake import *
import matplotlib.pyplot as plt

from classes import *


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def composition_losses(N, T2_domain):
    T1 = "01"    
    model = KSModel.load(f"data/KS/models/CG3/fourier/N{N}/T{T1}/D{64}_M16_samples1000_epoch750.pt", N, T1, device)
    
    cg3 = FunctionSpace(model.mesh, "CG", 3)
    x = assemble(interpolate(SpatialCoordinate(model.mesh)[0], cg3))
    grid = torch.tensor(sorted(x.dat.data[:])).to(device=device, dtype=torch.float32)
    
    losses = []
    for T2 in T2_domain:
        data_path = f"data/KS/samples/N{N}_CG3_nu0029_T{T2}_samples1200.pt"
        samples = torch.load(data_path).unsqueeze(2).to(device=device, dtype=torch.float32)
        
        testset = Dataset(torch.tensor(samples[1000:]), torch.tensor(grid))
        
        mean_rel_l2_loss = lambda x, y: torch.mean(torch.norm(x - y, 2, dim=-1) / torch.norm(y, 2, dim=-1))
        
        with torch.no_grad():
            for _ in range(int(T2) - 1):
                testset.inputs[:, 0, :] = model.network(testset.inputs)[:, 0, :]
            
            losses.append(mean_rel_l2_loss(model.network(testset.inputs), testset.targets).cpu().numpy())
    
    return losses


N_domain = [64, 128, 256, 512, 1024, 2048]
T2_domain = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]

plt.figure(figsize=(7, 4.5))
for N in N_domain:
    plt.plot(T2_domain, composition_losses(N, T2_domain), label=f"N={N} model")
    
plt.legend()
plt.grid()
plt.yscale("log")
plt.title("$T$ times composed network - All $N$")
plt.ylabel("RelL2")
plt.xlabel("T $[10 \, X \, dt]$")

plt.savefig(f"KS compositions N1024 D{64} All N")