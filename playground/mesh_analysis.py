from typing import List

import firedrake as fd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import numpy as np

from network import NeuralNetworkTrainer, NonlocalNeuralOperator
from burgers import BurgersDataset, fourier_coefficients


def average_coefficient_loss(models: List[NonlocalNeuralOperator], datasets: List[BurgersDataset], h_arr: List[float]):
    losses = []
    for model, dataset, h in zip(models, datasets, h_arr):
        target = dataset[:][1]
        prediction = model(dataset[:][0])
        losses.append(h / len(target) * nn.L1Loss(reduction='sum')(target, prediction))

    return losses


def average_firedrake_loss(models, datasets, nx_arr, ):
    losses = []
    for model, dataset, nx, in zip(models, datasets, nx_arr):
        filename = f"data/burgers__samples_1000__nx_{nx}"
        with fd.CheckpointFile(f"{filename}__mesh.h5", "r") as file:
            function_space = fd.FunctionSpace(file.load_mesh(), "CG", 1)


        targets = dataset[:][1].squeeze(1).detach().cpu().numpy()
        predictions = model(dataset[:][0]).squeeze(1).detach().cpu().numpy()
        # samples = dataset[:][0][0].detach().cpu().numpy()

        loss = 0
        for target, predict in zip(targets, predictions):
            loss += fd.errornorm(
                fd.Function(function_space, val=target),
                fd.Function(function_space, val=predict),
                "L1"
            )

        losses.append(loss / len(targets) )

    return losses


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = 10
    max_modes = 10
    nx_arr = torch.Tensor([50, 100, 256, 512]).to(dtype=torch.int)
    h_arr = 1 / nx_arr

    models = []
    datasets = []
    for nx in nx_arr:
        filename = f"data/burgers__samples_1000__nx_{nx}"

        samples = torch.load(f"{filename}.pt").unsqueeze(2).to(device=device, dtype=torch.float32)
        grid = torch.linspace(0, 1, nx, device=device)
        dataset = BurgersDataset(samples, grid)

        models.append(torch.load(f"models/nx_{nx}__d_{d}__max_modes_{max_modes}.pt"))
        datasets.append(dataset[int(0.8 * len(dataset)):])  # Cutting off the train data

    losses_coeff = average_coefficient_loss(models, datasets, h_arr)
    losses_fd = average_firedrake_loss(models, datasets, nx_arr)

    for model, nx, loss_fd, loss_coeff in zip(models, nx_arr, losses_fd, losses_coeff):
        print(f"nx: {nx:03} | Average Firedrake loss: {loss_fd:.04} | Average coeff loss: {loss_coeff:.04}")
