import firedrake as fd
import torch

from .NonlocalNeuralOperator import NonlocalNeuralOperator
from .ProjectionCoefficient import ProjectionCoefficient


class NeuralOperatorModel:
    def __init__(
            self,
            network: NonlocalNeuralOperator,
            config: dict,
            save: bool = True
    ):
        """
        Class to store pretrained NonLocalNeuralOperator networks and their metadata, used for inference.
        """
        self.network = network
        self.config = config

        self.config["N"] = network.projection.N
        self.config["D"] = network.D
        self.config["M"] = network.M
        self.config["depth"] = network.depth

        if save:
            self.filename = f"data/models/{network.projection.projection_type}/N_{network.projection.N}" \
                            f"/{config['loss_type']}/D{network.D}_M{network.M}" \
                            f"_samples{config['train_samples']}_epoch{config['epoch']}.pt"

            torch.save({
                "state_dict": network.state_dict(),
                "config": config
            }, self.filename)

    @staticmethod
    def load(filename, device):
        state_dict, config = torch.load(filename).values()

        mesh = fd.PeriodicIntervalMesh(config["N"], 1)

        projection = ProjectionCoefficient.load(
            f"../data/projection_coefficients"
            f"/{config['projection_type']}/N{config['N']}_M{config['M']}.pt",
            mesh)
        network = NonlocalNeuralOperator(config["M"], config["D"], config["depth"], projection, device)
        network.load_state_dict(state_dict)

        model = NeuralOperatorModel(network, config, save=False)
        model.filename = filename

        return model

        # network = NonlocalNeuralOperator()


if __name__ == "__main__":
    N, M, D, depth = 100, 8, 10, 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh = fd.PeriodicIntervalMesh(N, 1)
    projection = ProjectionCoefficient(mesh, "fourier", M, device)
    projection.calculate()
    network = NonlocalNeuralOperator(M, D, depth, projection, device)

    config = {
        "N": N,
        "M": M,
        "D": D,
        "depth": depth,
        "projection_type": "fourier",
        "epoch": 0,
        "loss_type": "MSE",
        "train_samples": 5000
    }

    model1 = NeuralOperatorModel(network, config)
    model2 = NeuralOperatorModel.load("../data/models/fourier/N_100/MSE/D10_M8_samples5000_epoch0.pt", device)

    print(model1.network.state_dict()['lifting.weight'] == model2.network.state_dict()['lifting.weight'])
