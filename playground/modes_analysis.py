from typing import List

import torch
import matplotlib.pyplot as plt
import firedrake as fd

from burgers import BurgersDataset
from classes import NonlocalNeuralOperator, NeuralOperatorModel, ProjectionCoefficient, NeuralNetworkTrainer


def train_models(config, mode_arr):
    """
    Train models based on the given configuration and list of mode values.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        mode_arr (List[int]): List of mode values for training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for M in mode_arr:
        config["M"] = M

        samples = torch.load(f"data/samples/N{config['N']}_samples1000.pt").unsqueeze(2).to(device=device,
                                                                                            dtype=torch.float32)
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)
        network_trainer = NeuralNetworkTrainer(
            model,
            trainset,
            testset,
            optimizer,
            scheduler,
            max_epoch=config["epoch"]
        )

        print(f"Training M={M}, D={config['D']} with param={sum(p.numel() for p in network.parameters() if p.requires_grad)}")
        network_trainer.train_me(logs=False)


def load_models(config, mode_arr):
    """
    Load trained models based on the given configuration and list of mode values.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        mode_arr (List[int]): List of mode values for loading models.

    Returns:
        Tuple[List[NeuralOperatorModel], BurgersDataset]: List of loaded models and the dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for M in mode_arr:
        filename = f"data/models/{config['projection_type']}/N{config['N']}/{config['loss_type']}" \
                   f"/D{config['D']}_M{M}_samples{config['train_samples']}_epoch{config['epoch']}.pt"

        samples = (torch.load(f"data/samples/N{config['N']}_samples1000.pt")
                   .unsqueeze(2).to(device=device, dtype=torch.float32))
        grid = torch.linspace(0, 1, config['N'], device=device)
        dataset = BurgersDataset(samples, grid)

        models.append(NeuralOperatorModel.load(filename, device))

    return models, dataset[int(0.8 * len(dataset)):]  # Cutting off the train data


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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "D": 10,
        "N": 64,
        "depth": 3,
        "projection_type": "fourier",
        "loss_type": "MSE",
        "train_samples": 800,
        "epoch": 200,
        "C": 64
    }

    mode_arr = [i for i in range(1, 27, 2)]
    # train_models(config, mode_arr)

    losses = average_firedrake_loss(*load_models(config, mode_arr), config["N"], config["loss_type"])

    plt.plot(mode_arr, losses)
    plt.yscale("log")
    plt.title(f"MSE average loss for N={config['N']}")
    plt.xlabel("Fourier Modes")
    plt.ylabel("MSE")
    plt.grid()
    plt.legend()
    plt.show()
