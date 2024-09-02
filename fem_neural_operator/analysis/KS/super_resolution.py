import torch
import firedrake as fd
import matplotlib.pyplot as plt

from classes import *


device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def super_resolution(N1, N2, D, M):
    with fd.CheckpointFile(f"data/KS/meshes/N{N2}.h5", "r") as file:
        mesh = file.load_mesh()
    
    
    model = KSModel.load(f"data/KS/models/fourier/N{N1}/T01/D{D}_M{M}_samples1000_epoch500.pt", N1, "01", device)
    projection = ProjectionCoefficient.load(mesh, "KS", N2, model.L, M, "HER", "fourier", device)
    
    for projection_layer in model.network.layers:
        projection_layer.coeff = projection.coeff
        projection_layer.coeff_T = projection.coeff.T
        projection_layer.functions = projection.functions
    
    
    data_path = f"data/KS/samples/N{N2}_HER_nu0029_T01_samples1200.pt"
    
    samples = torch.load(data_path).unsqueeze(2).to(device=device, dtype=torch.float32)
    grid = torch.linspace(0, model.L, 2 * N2, device=device)
    trainset = KSDataset(N2, torch.tensor(samples[:model.train_samples]), torch.tensor(grid))
    testset = KSDataset(N2, torch.tensor(samples[model.train_samples:]), torch.tensor(grid))
    
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

    plt.plot(N2_range[i:], losses, label=f"N={N1} Model")

plt.xscale("log", base=2)
plt.title("KS Super-Resolution")
plt.grid()
plt.ylabel("RelH1")
plt.xlabel("$N$ of test dataset")
plt.legend()

plt.savefig("KS Super-Resolution")