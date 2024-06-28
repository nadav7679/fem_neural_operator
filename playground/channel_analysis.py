from typing import List

import firedrake as fd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from burgers import BurgersDataset
from classes import *


def average_coefficient_loss(
        models: List[NeuralOperatorModel],
        dataset: BurgersDataset,
        h: float,
        loss_type: str = "MSE"
) -> (torch.Tensor, List[int]):
    """
    Calculate the average loss using coefficient approximation (i.e., only using PyTorch)
    for a list of models on a given dataset. We assume that the models are on the same Firedrake mesh, but the
    networks' architectures differ.

    Args:
        models (List[NeuralOperatorModel]): List of models to evaluate.
        dataset (BurgersDataset): Dataset to evaluate on.
        h (float): Mesh size, h=1/nx, corresponding to the dataset and mesh.
        loss_type (str, optional): Type of loss to use ("MSE" or "L1"). Defaults to "MSE".

    Returns:
        (torch.Tensor, List[int]): Tuple of average losses and the number of parameters for each model.
    """
    target = dataset[:][1]

    losses = torch.zeros(len(models), dtype=torch.float32)
    loss_func = nn.MSELoss(reduction='sum') if loss_type == "MSE" else nn.L1Loss(reduction='sum')

    parameters = []
    for i, model in enumerate(models):
        prediction = model.network(dataset[:][0])
        param_num = sum(p.numel() for p in model.network.parameters() if p.requires_grad)

        losses[i] += h / len(target) * loss_func(target, prediction).detach().cpu()
        parameters.append(param_num)

    return losses, parameters


def average_firedrake_loss(
        models: List[NeuralOperatorModel],
        dataset: BurgersDataset,
        N: int,
        loss_type: str = "MSE"
) -> List[float]:
    """
    Calculate the average loss using Firedrake's errornorm for a list of models on a given dataset.

    Args:
        models (List[NeuralOperatorModel]): List of models to evaluate.
        dataset (BurgersDataset): Dataset to evaluate on.
        N (int): Mesh resolution corresponding to the dataset.
        loss_type (str, optional): Type of loss to use ("MSE" or "L1"). Defaults to "MSE".

    Returns:
        List[float]: List of average losses for each model.
    """
    targets = dataset[:][1].squeeze(1).detach().cpu().numpy()

    with fd.CheckpointFile(f"data/meshes/N{N}.h5", "r") as file:
        function_space = fd.FunctionSpace(file.load_mesh(), "CG", 1)

    power = 2 if loss_type == "MSE" else 1
    loss_type = "L2" if loss_type == "MSE" else loss_type

    losses = []
    for model in models:
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


def train_models(config, D_arr):
    """
    Train models based on the given configuration and list of D values.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        D_arr (List[int]): List of D values for training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for D in D_arr:
        config["D"] = D

        samples = (torch.load(f"data/samples/N{config['N']}_samples1000.pt")
                   .unsqueeze(2).to(device=device, dtype=torch.float32))
        with fd.CheckpointFile(f"data/meshes/N{config['N']}.h5", "r") as file:
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

        grid = torch.linspace(0, 1, config["N"], device=device)
        trainset = BurgersDataset(samples[:int(0.8 * samples.shape[0])], grid)
        testset = BurgersDataset(samples[int(0.8 * samples.shape[0]):], grid)

        lr = 0.01
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 0.5*config["epoch"], gamma=0.1)
        network_trainer = NeuralNetworkTrainer(
            model,
            trainset,
            testset,
            optimizer,
            scheduler,
            max_epoch=config["epoch"]
        )

        network_trainer.train_me(logs=False)


def load_models(config, D_arr):
    """
    Load trained models based on the given configuration and list of D values.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        D_arr (List[int]): List of D values for loading models.

    Returns:
        Tuple[List[NeuralOperatorModel], BurgersDataset]: List of loaded models and the dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for D in D_arr:
        filename = f"data/models/{config['projection_type']}/N{config['N']}/{config['loss_type']}" \
                   f"/D{D}_M{config['M']}_samples{config['train_samples']}_epoch{config['epoch']}.pt"

        samples = (torch.load(f"data/samples/N{config['N']}_samples1000.pt")
                   .unsqueeze(2).to(device=device, dtype=torch.float32))
        grid = torch.linspace(0, 1, config['N'], device=device)
        dataset = BurgersDataset(samples, grid)

        models.append(NeuralOperatorModel.load(filename, device))

    return models, dataset[int(0.8 * len(dataset)):]  # Cutting off the train data


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D_arr = torch.arange(5, 160, 5).to(dtype=torch.int)

    config = {
        "M": 8,
        "N": 64,
        "depth": 3,
        "projection_type": "fourier",
        "loss_type": "MSE",
        "train_samples": 800,
        "epoch": 500,
    }


    for epoch in [50, 100, 200, 300, 400, 500]:
        config["epoch"] = epoch
        # train_models(config, D_arr)
        losses = average_firedrake_loss(*load_models(config, D_arr), config["N"], "MSE")
        plt.plot(D_arr, losses, label=f"{config['epoch']} epochs")

    # for d, loss, loss_fd, param in zip(D_arr, losses, losses_fd, parameters):
    #     print(f"d: {d:03} | Parameters: {param:06} | Average loss: {loss:.04} | Firedrake loss: {loss_fd:.04}")

    plt.title(f"MSE average loss vs D N={config['N']}")
    plt.xlabel("D - channels")
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()
