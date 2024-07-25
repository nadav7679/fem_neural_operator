import torch
import torch.nn as nn
import firedrake as fd

from classes import *
from burgers import BurgersDataset


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Network and projection setup
N = 4096
M, D, depth = 10, 20, 3

with fd.CheckpointFile(f"data/KS/meshes/N{N}.h5", "r") as f:
    mesh = f.load_mesh()

projection = ProjectionCoefficient(mesh, "KS", "HER", "fourier", M, device)
projection.calculate()

network = NonlocalNeuralOperator(M, D, depth, projection, device)

# Datasets setup
samples = torch.load(f"data/KS/samples/N{N}_HER_nu0029_T01_samples1200.pt").unsqueeze(2).to(device=device, dtype=torch.float32)
print(samples.shape)
grid = torch.linspace(0, 1, 2 * N, device=device)
dataset = BurgersDataset(samples, grid)
print(dataset.inputs.shape, dataset.targets.shape)

trainset = dataset[:1000]  # Cutting off the train data
testset = dataset[1000:]



# Training setup
lr = 0.01
mse_loss = nn.MSELoss(reduction="sum")
loss = lambda x, y: mse_loss(x, y) / (N * len(x))  # Sum of all differences, times step size, divide by batch size
optimizer = torch.optim.Adam(network.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5)

# Model and trainer creation
model = NeuralOperatorModel(network, "KS", 100, 1000)

network_trainer = NeuralNetworkTrainer(
    model,
    trainset,
    testset,
    loss,
    optimizer,
    scheduler,
    max_epoch=500
)

network_trainer.train_me()

