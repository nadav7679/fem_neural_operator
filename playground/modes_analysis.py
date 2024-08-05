from typing import List

import torch
import matplotlib.pyplot as plt
import firedrake as fd

from burgers import BurgersDataset
from classes import NeuralOperatorNetwork, BurgersModel, ProjectionCoefficient, NeuralNetworkTrainer

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def train_models(config, mode_arr):
    """
    Train models based on the given configuration and list of mode values.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        mode_arr (List[int]): List of mode values for training.
    """

    for M in mode_arr:
        config["M"] = M
        model = BurgersModel(config["N"],
                             config["M"],
                             config["D"],
                             config["depth"],
                             config["T"],
                             config["projection_type"],
                             device=device)
        print(f"Training M={M}, D={config['D']} with param={model.param_num}")
        model.train(f"data/burgers/samples/N{config['N']}_nu001_T{config['T']}_samples1200.pt", config['epoch'],
                    lr=0.001, device=device)


def load_models(config, mode_arr):
    """
    Load trained models based on the given configuration and list of mode values.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        mode_arr (List[int]): List of mode values for loading models.

    Returns:
        Tuple[List[NeuralOperatorModel], BurgersDataset]: List of loaded models and the dataset.
    """
    global device

    models = []
    for M in mode_arr:
        filename = f"data/burgers/models/{config['projection_type']}/N{config['N']}/T{config['T']}" \
                   f"/D{config['D']}_M{M}_samples{config['train_samples']}_epoch{config['epoch']}.pt"

        samples = (torch.load(f"data/burgers/samples/N{config['N']}_nu001_T{config['T']}_samples1200.pt")
                   .unsqueeze(2).to(device=device, dtype=torch.float32))
        grid = torch.linspace(0, 1, config['N'], device=device)
        dataset = BurgersDataset(samples, grid)

        models.append(BurgersModel.load(filename, config["N"], config["T"], device))

    return models, dataset[config["train_samples"]:]  # Cutting off the train data


def average_firedrake_loss(
        models: List[BurgersModel],
        dataset: BurgersDataset
) -> List[float]:
    """
    Calculate the average loss using Firedrake's errornorm for a list of models on a given dataset.

    Args:
        models (List[NeuralOperatorModel]): List of models to evaluate.
        dataset (BurgersDataset): Dataset to evaluate on.

    Returns:
        List[float]: List of average losses for each model.
    """
    targets = dataset[:][1].squeeze(1).detach().cpu().numpy()

    with fd.CheckpointFile(f"data/burgers/meshes/N{models[0].N}.h5", "r") as file:
        function_space = fd.FunctionSpace(file.load_mesh(), "CG", 1)

    losses = []
    for model in models:
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
    config = {
        "D": 64,
        "N": 4096,
        "depth": 4,
        "T": 1,
        "projection_type": "fourier",
        "loss_type": "MSE",
        "train_samples": 1000,
        "epoch": 500,
    }

    mode_arr = [i for i in range(0, 27, 2)]
    train_models(config, mode_arr)

    losses = average_firedrake_loss(*load_models(config, mode_arr))
    plt.plot(mode_arr, losses)
    plt.yscale("log")
    plt.title(f"RelL2 loss vs Fourier modes for N={config['N']}")
    plt.xlabel("Fourier Modes")
    plt.ylabel("RelL2")
    plt.grid()
    plt.legend()
    plt.show()
