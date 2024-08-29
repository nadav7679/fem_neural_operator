import torch
import firedrake as fd
import matplotlib.pyplot as plt

from classes import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def super_resolution(N1, N2, D, M):
    with fd.CheckpointFile(f"data/burgers/meshes/N{N2}.h5", "r") as file:
        mesh = file.load_mesh()
    
    
    model = BurgersModel.load(f"data/burgers/models/fourier/N{N1}/T1/D{D}_M{M}_samples1000_epoch500.pt", N1, "01", device)
    projection = ProjectionCoefficient.load(mesh, "burgers", N2, 1, M, "CG1", "fourier", device)
    
    for projection_layer in model.network.layers:
        projection_layer.coeff = projection.coeff
        projection_layer.coeff_T = projection.coeff.T
        projection_layer.functions = projection.functions
    
    
    data_path = f"data/burgers/samples/N{N2}_nu001_T1_samples1200.pt"
    
    samples = torch.load(data_path).unsqueeze(2).to(device=device, dtype=torch.float32)
    grid = torch.linspace(0, model.L, N2, device=device)
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
    
    

N2_range = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

D = 64
M = 16

plt.figure(figsize=(7, 4.5))
for i in range(5):
    N1 = N2_range[i]
    
    losses = []
    for N2 in N2_range[i:]:
        losses.append(super_resolution(N1, N2, D, M))

    plt.plot(N2_range[i:], losses, label=f"N={N1} Model")

plt.xscale("log", base=2)
plt.title("Burgers Super-Resolution")
plt.grid()
plt.ylabel("RelL2")
plt.xlabel("$N$ of test dataset")
plt.legend()

plt.savefig("Burgers Super-Resolution")