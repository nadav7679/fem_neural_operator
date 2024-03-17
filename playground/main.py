import matplotlib.pyplot as plt

from firedrake import *
from firedrake.pyplot import plot

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


from Network import Net, NeuralNetworkTrainer


def plot_coefficients(coefs: torch.Tensor):
    nx, length = 100, 1
    mesh = PeriodicIntervalMesh(nx, length=length)
    cg_space = FunctionSpace(mesh, "CG", degree=1)

    f = Function(cg_space, val=coefs.detach())
    plot(f)
    plt.show()


class BurgersTrainDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, 0, :], self.data[index, 1, :]


full_data = torch.load("burgers.pt")
train_dataset = BurgersTrainDataset(full_data[:int(0.8*len(full_data))])  # 80% train data
test_dataset = BurgersTrainDataset(full_data[int(0.8*len(full_data)):])   # 20% test data

plot_coefficients(test_dataset[1][0])
# net = NeuralNetworkTrainer(train_dataset,
#                            test_dataset,
#                            20,
#                            3,
#                            nn.CrossEntropyLoss(),
#                            torch.optim.Adam,
#                            max_epoch=5
#                            )