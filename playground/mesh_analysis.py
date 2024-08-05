from typing import List

import firedrake as fd
import torch
import torch.nn as nn

from burgers import *
from classes import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_models(config, N_arr):
    """
    Train models based on the given configuration and list of mode values.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        N_arr (List[int]): List of N values for training.
    """
    global device

    for N in N_arr:
        config["N"] = N
        model = BurgersModel(config["N"],
                             config["M"],
                             config["D"],
                             config["depth"],
                             config["T"],
                             config["projection_type"],
                             device=device)
        print(f"Training N={N}, D={config['D']} with param={model.param_num}")
        model.train(f"data/burgers/samples/N{config['N']}_nu001_T{config['T']}_samples1200.pt", config['epoch'],
                    lr=0.001, device=device)


def load_models(config, N_arr):
    """
    Load trained models based on the given configuration and grid resolutions.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        N_arr (List[int]): List of grid resolutions for loading models.

    Returns:
        Tuple[List[NeuralOperatorModel], List[BurgersDataset]]: List of loaded models and corresponding datasets.
    """
    global device

    models = []
    datasets = []
    for N in N_arr:
        config["N"] = N
        filename = f"data/burgers/models/{config['projection_type']}/N{config['N']}/T{config['T']}" \
                   f"/D{config['D']}_M{config['M']}_samples{config['train_samples']}_epoch{config['epoch']}.pt"

        samples = (torch.load(f"data/burgers/samples/N{config['N']}_nu001_T{config['T']}_samples1200.pt")
                   .unsqueeze(2).to(device=device, dtype=torch.float32))
        grid = torch.linspace(0, 1, config['N'], device=device)
        dataset = BurgersDataset(samples, grid)

        models.append(BurgersModel.load(filename, config["N"], config["T"], device))
        datasets.append(dataset[config["train_samples"]:])

    return models, datasets


def average_coefficient_loss(
        models: List[BurgersModel],
        datasets: List[BurgersDataset],
) -> List[torch.Tensor]:
    """
    Calculate the average loss using coefficient approximation (i.e. only using PyTorch)
    for a list of models and corresponding datasets.

    Args:
        models (List[NeuralOperatorModel]): List of models to evaluate.
        datasets (List[BurgersDataset]): List of datasets to evaluate on.

    Returns:
        List[torch.Tensor]: List of average losses for each model-dataset pair.
    """
    mean_rel_l2_loss = lambda x, y: torch.mean(torch.norm(x - y, 2, dim=-1) / torch.norm(y, 2, dim=-1))

    losses = []
    for model, dataset in zip(models, datasets):
        target = dataset[:][1]
        prediction = model.network(dataset[:][0])
        losses.append(mean_rel_l2_loss(target, prediction))

    return losses


def average_firedrake_loss(
        models: List[BurgersModel],
        datasets: List[BurgersDataset],
) -> List[float]:
    """
    Calculate the average loss using Firedrake's errornorm for a list of models and datasets.

    Args:
        models (List[NeuralOperatorModel]): List of models to evaluate.
        datasets (List[BurgersDataset]): List of datasets to evaluate on.

    Returns:
        List[float]: List of average losses for each model-dataset pair.
    """
    losses = []
    for model, dataset, N, in zip(models, datasets):
        with fd.CheckpointFile(f"data/burgers/meshes/N{model.N}.h5", "r") as file:
            function_space = fd.FunctionSpace(file.load_mesh(), "CG", 1)

        targets = dataset[:][1].squeeze(1).detach().cpu().numpy()
        predictions = model.network(dataset[:][0]).squeeze(1).detach().cpu().numpy()

        loss = 0
        for target, predict in zip(targets, predictions):
            target_func = fd.Function(function_space, val=target)
            loss += fd.errornorm(
                target_func,
                fd.Function(function_space, val=predict),
                "L2"
            ) / fd.norm(target_func)

        losses.append(loss / len(targets))

    return losses


if __name__ == "__main__":
    N_arr = torch.Tensor([2 ** i for i in range(6, 9)]).to(dtype=torch.int)
    config = {
        "M": 16,
        "D": 64,
        "depth": 4,
        "T": 1,
        "projection_type": "fourier",
        "train_samples": 1000,
        "epoch": 500,
    }

    # train_models(config, N_arr)
    models, datasets = load_models(config, N_arr)

    losses_coeff = average_coefficient_loss(models, datasets)
    losses_fd = average_firedrake_loss(models, datasets)

    for model, N, loss_fd, loss_coeff in zip(models, N_arr, losses_fd, losses_coeff):
        print(
            f"nx: {N:03} | Average Firedrake loss: {loss_fd:.04} | Average coeff loss: {loss_coeff:.04} | Diff {loss_coeff - loss_fd:.04}")
