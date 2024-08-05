from typing import List

import firedrake as fd
import matplotlib.pyplot as plt
import torch

from burgers import BurgersDataset
from classes import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def average_coefficient_loss(
        models: List[BurgersModel],
        dataset: BurgersDataset,
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
    for model in models:
        prediction = model.network(dataset[:][0])
        losses.append(mean_rel_l2_loss(dataset[:][1], prediction))

    return losses


def train_models(config, D_arr):
    """
    Train models based on the given configuration and list of D values.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        D_arr (List[int]): List of D values for training.
    """
    global device

    for D in D_arr:
        config["D"] = D
        model = BurgersModel(config["N"],
                             config["M"],
                             config["D"],
                             config["depth"],
                             config["T"],
                             config["projection_type"],
                             device=device)
        print(f"Training D={config['D']} with param={model.param_num}")
        model.train(f"data/burgers/samples/N{config['N']}_nu001_T{config['T']}_samples1200.pt", config['epoch'],
                    lr=0.001, device=device)


def load_models(config, D_arr):
    """
    Load trained models based on the given configuration and grid resolutions.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        D_arr (List[int]): List of channels D for loading models.

    Returns:
        Tuple[List[NeuralOperatorModel], List[BurgersDataset]]: List of loaded models and corresponding datasets.
    """
    global device

    models = []
    datasets = []
    for D in D_arr:
        config["D"] = D
        filename = f"data/burgers/models/{config['projection_type']}/N{config['N']}/T{config['T']}" \
                   f"/D{config['D']}_M{config['M']}_samples{config['train_samples']}_epoch{config['epoch']}.pt"

        samples = (torch.load(f"data/burgers/samples/N{config['N']}_nu001_T{config['T']}_samples1200.pt")
                   .unsqueeze(2).to(device=device, dtype=torch.float32))
        grid = torch.linspace(0, 1, config['N'], device=device)
        dataset = BurgersDataset(samples, grid)

        models.append(BurgersModel.load(filename, config["N"], config["T"], device))
        datasets.append(dataset[config["train_samples"]:])

    return models, datasets


if __name__ == "__main__":
    D_arr = torch.arange(10, 31, 10).to(dtype=torch.int)
    print(D_arr)

    config = {
        "N": 4096,
        "M": 16,
        "depth": 4,
        "T": 1,
        "projection_type": "fourier",
        "train_samples": 1000,
        "epoch": 5,
    }

    for M in [2]:
        config["M"] = M
        train_models(config, D_arr)
        losses = average_coefficient_loss(*load_models(config, D_arr))
        plt.plot(D_arr, losses, label=f"M={config['M']}")

    # for d, loss, loss_fd, param in zip(D_arr, losses, losses_fd, parameters):
    #     print(f"d: {d:03} | Parameters: {param:06} | Average loss: {loss:.04} | Firedrake loss: {loss_fd:.04}")

    plt.title(f"MSE average loss vs D N={config['N']}")
    plt.xlabel("D - channels")
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()
