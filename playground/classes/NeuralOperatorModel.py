import firedrake as fd
import torch

from .NonlocalNeuralOperator import NonlocalNeuralOperator
from .ProjectionCoefficient import ProjectionCoefficient


class NeuralOperatorModel:
    def __init__(
            self,
            network: NonlocalNeuralOperator,
            epoch=0,
            loss_type="MSE",
            train_samples=1000,
            save: bool = False
    ):
        """
        Class to store pretrained NonLocalNeuralOperator networks and their metadata, used for inference.
        """
        self.network = network
        self.config = {
            "M": network.M,
            "D": network.D,
            "depth": network.depth,
            "N": network.projection.N,
            "projection_type": network.projection.projection_type,
            "epoch": epoch,
            "loss_type": loss_type,
            "train_samples": train_samples
        }

        self.filename = f"data/models/{network.projection.projection_type}/N_{network.projection.N}" \
                        f"/{loss_type}/D{network.D}_M{network.M}" \
                        f"_samples{train_samples}_epoch{epoch}.pt"

        if save:
            self.save()

    def save(self):
        self.filename = f"data/models/{self.network.projection.projection_type}/N{self.network.projection.N}" \
                        f"/{self.config['loss_type']}/D{self.network.D}_M{self.network.M}" \
                        f"_samples{self.config['train_samples']}_epoch{self.config['epoch']}.pt"

        torch.save({
            "state_dict": self.network.state_dict(),
            "config": self.config
        }, self.filename)

    @staticmethod
    def load(filename, device):
        state_dict, config = torch.load(filename).values()

        mesh = fd.PeriodicIntervalMesh(config["N"], 1)

        projection = ProjectionCoefficient.load(
            f"data/projection_coefficients"
            f"/{config['projection_type']}/N{config['N']}_M{config['M']}.pt",
            mesh)
        network = NonlocalNeuralOperator(config["M"], config["D"], config["depth"], projection, device)
        network.load_state_dict(state_dict)

        model = NeuralOperatorModel(network, config["epoch"], config["loss_type"], config["train_samples"], save=False)
        model.filename = filename

        return model


if __name__ == "__main__":
    N, M, D, depth = 100, 8, 10, 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh = fd.PeriodicIntervalMesh(N, 1)
    projection = ProjectionCoefficient(mesh, "fourier", M, device)
    projection.calculate()
    network = NonlocalNeuralOperator(M, D, depth, projection, device)

    model1 = NeuralOperatorModel(network, save=True)
    model2 = NeuralOperatorModel.load(f"../data/models/fourier/N_100/MSE/D{D}_M{M}_samples1000_epoch0.pt", device)

    print(model1.network.state_dict()['lifting.weight'] == model2.network.state_dict()['lifting.weight'])
