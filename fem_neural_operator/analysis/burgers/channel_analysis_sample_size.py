from typing import List

import firedrake as fd
import matplotlib.pyplot as plt
import torch

from classes import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def average_firedrake_loss(
        models: List[BurgersModel],
        dataset: Dataset
) -> List[float]:
    """
    Calculate the average loss using Firedrake's errornorm for a list of models on a given dataset.

    Args:
        models (List[NeuralOperatorModel]): List of models to evaluate.
        dataset (Dataset): Dataset to evaluate on.

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
                fd.Function(function_space, val=predict)
            ) / fd.norm(target_func)

        losses.append(loss / len(targets))

    return losses


def average_coefficient_loss(
        models: List[BurgersModel],
        dataset: Dataset,
) -> List[torch.Tensor]:
    """
    Calculate the average loss using coefficient approximation (i.e. only using PyTorch)
    for a list of models and corresponding datasets.

    Args:
        models (List[NeuralOperatorModel]): List of models to evaluate.
        datasets (List[Dataset]): List of datasets to evaluate on.

    Returns:
        List[torch.Tensor]: List of average losses for each model-dataset pair.
    """
    mean_rel_l2_loss = lambda x, y: torch.mean(torch.norm(x - y, 2, dim=-1) / torch.norm(y, 2, dim=-1))

    losses = []
    with torch.no_grad():
        for model in models:
            prediction = model.network(dataset[:][0])
            losses.append(mean_rel_l2_loss(dataset[:][1], prediction).cpu().numpy())

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
                             config["train_samples"],
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
        Tuple[List[NeuralOperatorModel], List[Dataset]]: List of loaded models and corresponding datasets.
    """
    global device

    samples = (torch.load(f"data/burgers/samples/N{config['N']}_nu001_T{config['T']}_samples1200.pt")
                .unsqueeze(2).to(device=device, dtype=torch.float32))
    grid = torch.linspace(0, 1, config['N'], device=device)
    dataset = Dataset(samples, grid)

    models = []
    for D in D_arr:
        config["D"] = D
        filename = f"data/burgers/models/CG1/{config['projection_type']}/N{config['N']}/T{config['T']}" \
                   f"/D{config['D']}_M{config['M']}_samples{config['train_samples']}_epoch{config['epoch']}.pt"


        models.append(BurgersModel.load(filename, config["N"], config["T"], device))

    return models, dataset[1000:]


if __name__ == "__main__":
    D_arr = torch.arange(10, 125, 10).to(dtype=torch.int)
    config = {
        "N": 4096,
        "M": 8,
        "depth": 4,
        "T": 1,
        "projection_type": "fourier",
        "epoch": 500,
    }

    plt.figure(figsize=(8, 6))

    for train_samples in [300, 700, 1000]:
        config["train_samples"] = train_samples
        # train_models(config, D_arr)
        losses = average_coefficient_loss(*load_models(config, D_arr))
        
        
        # losses = average_coefficient_loss(*load_models(config, D_arr))
        plt.plot(D_arr, losses, label=f"Train samples={train_samples}")

    # # for d, loss, loss_fd, param in zip(D_arr, losses, losses_fd, parameters):
    # #     print(f"d: {d:03} | Parameters: {param:06} | Average loss: {loss:.04} | Firedrake loss: {loss_fd:.04}")

    # plt.title(f"RelL2 loss $D$ - $N={config['N']}$")
    plt.xlabel("D - channels")
    plt.ylabel("RelL2")
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.savefig("channel_analysis_sample_size")
