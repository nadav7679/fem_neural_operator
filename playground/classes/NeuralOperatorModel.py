import firedrake as fd
import torch

from .NonlocalNeuralOperator import NonlocalNeuralOperator
from .ProjectionCoefficient import ProjectionCoefficient


class NeuralOperatorModel:
    def __init__(
            self,
            network: NonlocalNeuralOperator,
            equation_type,
            finite_element_space,
            epoch=0,
            train_samples=1000,
            save: bool = False
    ):
        """
        Class to store pretrained NonLocalNeuralOperator networks and their metadata, used for inference.

        Args:
            network (NonlocalNeuralOperator): The neural network model to be saved and managed.
            equation_type (str): The equation the Operator should learn. Either 'Burgers' or 'KS'
            epoch (int): The amount of epochs the model was trained on.
            train_samples (int): The number of training samples used during training.
            save (bool): Whether to save the model immediately after initialization.
        """
        self.network = network
        self.equation_type = equation_type
        self.finite_element_space = finite_element_space
        self.config = {
            "M": network.M,
            "D": network.D,
            "depth": network.depth,
            "N": network.projection.N,
            "projection_type": network.projection.projection_type,
            "finite_element_space": finite_element_space,
            "epoch": epoch,
            "train_samples": train_samples
        }

        self.filename = f"data/{equation_type}/models/{network.projection.projection_type}/N{network.projection.N}" \
                        f"/D{network.D}_M{network.M}_samples{train_samples}_epoch{epoch}.pt"
        
        self.param_num = sum(p.numel() for p in network.parameters() if p.requires_grad)

        if save:
            self.save()

    def save(self):
        self.filename = f"data/{self.equation_type}/models/{self.network.projection.projection_type}/N{self.network.projection.N}" \
                        f"/D{self.network.D}_M{self.network.M}_samples{self.config['train_samples']}_epoch{self.config['epoch']}.pt"

        torch.save({
            "state_dict": self.network.state_dict(),
            "config": self.config
        }, self.filename)

    @staticmethod
    def load(filename, equation_type, device):
        state_dict, config = torch.load(filename).values()

        mesh = fd.PeriodicIntervalMesh(config["N"], 1)

        projection = ProjectionCoefficient.load(
            f"data/{equation_type}/projection_coefficients/{finite_element_space}"
            f"/{config['projection_type']}/N{config['N']}_M{config['M']}.pt",
            mesh, device)
        network = NonlocalNeuralOperator(config["M"], config["D"], config["depth"], projection, device)
        network.load_state_dict(state_dict)

        model = NeuralOperatorModel(network, equation_type, config["finite_element_space"], config["epoch"], config["train_samples"], save=False)
        model.filename = filename

        return model

# Usage Example
if __name__ == "__main__":
    N, M, D, depth, equation_type = 100, 8, 10, 3, "KS"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    mesh = fd.PeriodicIntervalMesh(N, 1)
    projection = ProjectionCoefficient(mesh, equation_type, "fourier", M, device)
    projection.calculate()
    network = NonlocalNeuralOperator(M, D, depth, projection, device)

    # Create and save the model
    model1 = NeuralOperatorModel(network, equation_type, save=True)

    # Load the model
    model2 = NeuralOperatorModel.load(f"data/{equation_type}/models/fourier/N100/D{D}_M{M}_samples1000_epoch0.pt", equation_type, device)

    print(model1.network.state_dict()['lifting.weight'] == model2.network.state_dict()['lifting.weight'])
