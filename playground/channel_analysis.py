from typing import List

import firedrake as fd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from network import NeuralNetworkTrainer, NonlocalNeuralOperator
from burgers import BurgersDataset, fourier_coefficients


def average_loss(models: List[NonlocalNeuralOperator], dataset: BurgersDataset, h: float):
    target = dataset[:][1]

    losses = torch.zeros(len(models), dtype=torch.float32)
    parameters = []
    for i, model in enumerate(models):
        prediction = model(dataset[:][0])
        param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

        losses[i] += h / len(target) * nn.L1Loss(reduction='sum')(target, prediction).detach().cpu()
        parameters.append(param_num)

    return losses, parameters


def average_firedrake_loss(models, dataset, nx):
    targets = dataset[:][1].squeeze(1).detach().cpu().numpy()

    filename = f"data/burgers__samples_1000__nx_{nx}"
    with fd.CheckpointFile(f"{filename}__mesh.h5", "r") as file:
        function_space = fd.FunctionSpace(file.load_mesh(), "CG", 1)

    losses = []
    for model in models:
        predictions = model(dataset[:][0]).squeeze(1).detach().cpu().numpy()

        loss = 0
        for target, predict in zip(targets, predictions):
            loss += fd.errornorm(
                fd.Function(function_space, val=target),
                fd.Function(function_space, val=predict),
                "L1"
            )

        losses.append(loss / len(targets))

    return losses


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nx = 256
    h = 1 / nx
    max_modes = 10
    d_arr = torch.arange(5, 105, 5).to(dtype=torch.int)

    filename = f"data/burgers__samples_1000__nx_{nx}"
    samples = torch.load(f"{filename}.pt").unsqueeze(2).to(device=device, dtype=torch.float32)
    grid = torch.linspace(0, 1, nx, device=device)
    dataset = BurgersDataset(samples, grid)[int(0.8 * len(samples)):]

    models = [torch.load(f"models/nx_{nx}__d_{d}__max_modes_{max_modes}.pt") for d in d_arr]
    losses, parameters = average_loss(models, dataset, h)
    losses_fd = average_firedrake_loss(models, dataset, nx)

    for d, loss, loss_fd, param in zip(d_arr, losses, losses_fd, parameters):
        # print(rf"${d}$ & ${loss:.05}$ & {param}\\")
        print(f"d: {d:03} | Parameters: {param:06} | Average loss: {loss:.04} | Firedrake loss: {loss_fd:.04}")

    plt.plot(d_arr, losses_fd)
    plt.title(f"L1 average loss for Burgers with varying d nx={nx}")
    plt.xlabel("d - channels")
    # plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()
