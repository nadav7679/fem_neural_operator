from typing import List

import firedrake as fd
import torch
import torch.nn as nn

from burgers import *
from classes import *


def average_coefficient_loss(
        models: List[NeuralOperatorModel],
        datasets: List[BurgersDataset],
        h_arr: List[float],
        loss_type: str = "MSE"
) -> List[torch.Tensor]:
    """
    Calculate the average loss using coefficient approximation (i.e. only using pytorch)
    for a list of models and corresponding datasets.

    Args:
        models (List[NonlocalNeuralOperator]): List of models to evaluate.
        datasets (List[BurgersDataset]): List of datasets to evaluate on.
        h_arr (List[float]): List of the mesh sizes, h=1/nx, corresponding to each dataset.
        loss_type (str, optional): Type of loss to use ("L2" or "L1"). Defaults to "L2". Note that "L2" loss is summing
                                    the square of L2 norms rather than the norm (i.e. ||x||^2_{L2}).

    Returns:
        List[torch.Tensor]: List of average losses for each model-dataset pair.
    """
    loss_func = nn.MSELoss(reduction='sum') if loss_type == "MSE" else nn.L1Loss(reduction='sum')

    losses = []
    for model, dataset, h in zip(models, datasets, h_arr):
        target = dataset[:][1]
        prediction = model.network(dataset[:][0])
        losses.append(h / len(target) * loss_func(target, prediction))

    return losses


def average_firedrake_loss(
        models: List[NeuralOperatorModel],
        datasets: List[BurgersDataset],
        N_arr: List[int],
        loss_type: str = "MSE"
) -> List[float]:
    """
    Calculate the average loss using Firedrake's errornorm for a list of models and datasets.

    Args:
        models (List[NonlocalNeuralOperator]): List of models to evaluate.
        datasets (List[BurgersDataset]): List of datasets to evaluate on.
        nx_arr (List[int]): List of grid resolutions corresponding to each dataset.
        loss_type (str, optional): Type of loss to use ("L2" or "L1"). Defaults to "L2". Note that "L2" loss is summing
                                    the square of L2 norms rather than the norm (i.e. ||x||^2_{L2}).

    Returns:
        List[float]: List of average losses for each model-dataset pair.
    """
    power = 2 if loss_type == "MSE" else 1
    loss_type = "L2" if loss_type == "MSE" else loss_type

    losses = []
    for model, dataset, N, in zip(models, datasets, N_arr):
        with fd.CheckpointFile(f"data/meshes/N{N}.h5", "r") as file:
            function_space = fd.FunctionSpace(file.load_mesh(), "CG", 1)

        targets = dataset[:][1].squeeze(1).detach().cpu().numpy()
        predictions = model.network(dataset[:][0]).squeeze(1).detach().cpu().numpy()

        loss = 0
        for target, predict in zip(targets, predictions):
            loss += fd.errornorm(
                fd.Function(function_space, val=target),
                fd.Function(function_space, val=predict),
                loss_type
            ) ** power

        losses.append(loss / len(targets))

    return losses


def train_models(config, N_arr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for N in N_arr:
        config["N"] = N

        samples = torch.load(f"data/samples/N{N}_samples1000.pt").unsqueeze(2).to(device=device, dtype=torch.float32)
        with fd.CheckpointFile(f"data/meshes/N{N}.h5", "r") as file:
            mesh = file.load_mesh()

        projection = ProjectionCoefficient(mesh, config["projection_type"], config["M"], device)
        projection.calculate()

        network = NonlocalNeuralOperator(
            config["M"],
            config["D"],
            config["depth"],
            projection,
            device
        )
        model = NeuralOperatorModel(network, train_samples=int(0.8 * samples.shape[0]))

        grid = torch.linspace(0, 1, N, device=device)
        trainset = BurgersDataset(samples[:int(0.8 * samples.shape[0])], grid)
        testset = BurgersDataset(samples[int(0.8 * samples.shape[0]):], grid)

        lr = 0.01
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5)
        network_trainer = NeuralNetworkTrainer(
            model,
            trainset,
            testset,
            optimizer,
            scheduler,
            max_epoch=500
        )

        network_trainer.train_me()


def load_models(config, N_arr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    datasets = []
    for N in N_arr:
        filename = f"data/models/{config['projection_type']}/N{N}" \
                   f"/{config['loss_type']}/D{config['D']}_M{config['M']}" \
                   f"_samples{config['train_samples']}_epoch{config['epoch']}.pt"

        samples = torch.load(f"data/samples/N{N}_samples1000.pt").unsqueeze(2).to(device=device, dtype=torch.float32)
        grid = torch.linspace(0, 1, N, device=device)
        dataset = BurgersDataset(samples, grid)

        datasets.append(dataset[int(0.8 * len(dataset)):])  # Cutting off the train data
        models.append(NeuralOperatorModel.load(filename, device))

    return models, datasets


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N_arr = torch.Tensor([64, 128, 256, 512]).to(dtype=torch.int)
    config = {
        "M": 8,
        "D": 10,
        "depth": 3,
        "projection_type": "fourier",
        "loss_type": "MSE",
        "train_samples": 800,
        "epoch": 500,
    }

    train_models(config, N_arr)
    models, datasets = load_models(config, N_arr)

    losses_coeff = average_coefficient_loss(models, datasets, 1/N_arr, config["loss_type"])
    losses_fd = average_firedrake_loss(models, datasets, N_arr, config["loss_type"])

    for model, nx, loss_fd, loss_coeff in zip(models, N_arr, losses_fd, losses_coeff):
        print(
            f"nx: {nx:03} | Average Firedrake loss: {loss_fd:.04} | Average coeff loss: {loss_coeff:.04} | Diff {loss_coeff - loss_fd:.04}")
