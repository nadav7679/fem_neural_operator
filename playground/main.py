import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from network import NeuralNetworkTrainer, NonlocalNeuralOperator
from burgers.utils import fourier_coefficients


class BurgersDataset(Dataset):
    def __init__(self, data, grid):
        self.targets = data[:, 1, ...]

        grids = grid.repeat(data.shape[0], 1, 1)
        self.inputs = torch.concat((data[:, 0, ...], grids), 1)  # Adding grid/mesh channel to input

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


nx = 50
filename = f"data/burgers__samples_1000__nx_{nx}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_modes = 10

samples = torch.load(f"{filename}.pt").unsqueeze(2).to(device=device, dtype=torch.float32)

try:
    coeff = torch.load(f"{filename}__coefficients_{max_modes}.pt").to(device=device, dtype=torch.float32)

except FileNotFoundError:
    coeff = fourier_coefficients(filename, max_modes).to(device=device, dtype=torch.float32)

grid = torch.linspace(0, 1, nx, device=device)
samples_len = samples.shape[0]
trainset = BurgersDataset(samples[:int(0.8 * samples_len)], grid)
testset = BurgersDataset(samples[int(0.8 * samples_len):], grid)

dim = coeff.shape[1]
print(dim)
d = 10
depth = 3
net = NonlocalNeuralOperator(device, dim, d, depth, coeff).to(dtype=torch.float32)
param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Number of parameters: {param_num}")
print(f"Used device: {device}")


mesh_size = 0.1
lr = 0.01
l1loss = nn.L1Loss(reduction="mean")  # Note that this loss sums over the batch as well
loss = lambda x, y: mesh_size * l1loss(x, y)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
network_trainer = NeuralNetworkTrainer(
    net,
    trainset,
    testset,
    loss,
    optimizer,
    scheduler,
    max_epoch=200
)

losses = network_trainer.train_me()
torch.save(losses, "losses_200.pt")

plt.plot(losses[0], label="Train loss")
plt.plot(losses[1], label="Test loss")
plt.title("Burgers equation L1 avg loss")
plt.xlabel("Epoch")
plt.yscale("log")
plt.grid()
plt.legend()
plt.show()

torch.save(net, "model_trained_on_nx_50.pt")
