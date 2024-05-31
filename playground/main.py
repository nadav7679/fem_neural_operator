import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from network import NeuralNetworkTrainer, NonlocalNeuralOperator
from burgers.utils import fourier_coefficients


class BurgersDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, 0, ...], self.data[index, 1, ...]


filename = "data/burgers__samples_100__nx_100"

samples = torch.load(f"{filename}.pt").unsqueeze(2).to(dtype=torch.float32)

try:
    coeff = torch.load(f"{filename}__coefficients.pt").to(dtype=torch.float32)

except FileNotFoundError:
    max_modes = 8
    coeff = fourier_coefficients(filename, max_modes).to(dtype=torch.float32)

samples_len = samples.shape[0]
trainset = BurgersDataset(samples[:int(0.8 * samples_len)])
testset = BurgersDataset(samples[int(0.8 * samples_len):])


dim = coeff.shape[1]
d = 10
depth = 3
net = NonlocalNeuralOperator(dim, d, depth, coeff)

mesh_size = 0.1
l1loss = nn.L1Loss(reduction="sum")  # Note that this loss sums over the batch as well
loss = lambda x, y: mesh_size * l1loss(x, y)
optimizer = torch.optim.Adam
network_trainer = NeuralNetworkTrainer(
    net,
    trainset,
    testset,
    loss,
    optimizer,
    max_epoch=10
)

network_trainer.train_me()



