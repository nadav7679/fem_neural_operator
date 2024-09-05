import torch
from firedrake import *
import matplotlib.pyplot as plt

from classes import *


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def super_resolution(N1, N2, D, M):
    T = "10"
    model = KSModel.load(f"data/KS/models/CG3/fourier/N{N1}/T{T}/D{D}_M{M}_samples1000_epoch750.pt", N1, "01", device)

    with CheckpointFile(f"data/KS/meshes/N{N2}.h5", "r") as file:
        mesh = file.load_mesh()    
    
    projection = ProjectionCoefficient.load(mesh, "KS", N2, model.L, M, "CG3", "fourier", device)
    
    for projection_layer in model.network.layers:
        projection_layer.coeff = projection.coeff
        projection_layer.coeff_T = projection.coeff.T
        projection_layer.functions = projection.functions
    
    
    data_path = f"data/KS/samples/N{N2}_CG3_nu0029_T{T}_samples1200.pt"
    samples = torch.load(data_path).unsqueeze(2).to(device=device, dtype=torch.float32)
    
    cg3 = FunctionSpace(mesh, "CG", 3)
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
    
    with torch.no_grad():
        return network_trainer.test_epoch().detach().cpu().numpy()
    
    

N2_range = [64, 128, 256, 512, 1024, 2048, 4096]

D = 64
M = 16

plt.figure(figsize=(7, 4.5))
for i in range(5):
    N1 = N2_range[i]
    
    losses = []
    for N2 in N2_range[i:]:
        losses.append(super_resolution(N1, N2, D, M))

    plt.plot(N2_range[i:], losses, label=f"Model N={N1}")

plt.xscale("log", base=2)
plt.title("KS Super-Resolution T=10")
plt.grid()
plt.ylabel("RelL2")
plt.xlabel("Resolution $N$ of tested dataset")
plt.legend()

plt.savefig("KS Super-Resolution T10 750e")