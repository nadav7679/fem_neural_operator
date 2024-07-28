from abc import ABCMeta, abstractmethod

import firedrake as fd
import torch

from .NeuralOperatorNetwork import NeuralOperatorNetwork
from .ProjectionCoefficient import ProjectionCoefficient


class NeuralOperatorModel(metaclass=ABCMeta):
    def __init__(
            self,
            N,
            L,
            projection_type,
            equation_name,
            finite_element_space,
            train_samples=1000,
    ):
        """
        Class to store pretrained NonLocalNeuralOperator networks and their metadata, used for inference.

        Args:
            network (NonlocalNeuralOperator): The neural network model to be saved and managed.
            equation_name (str): The equation the Operator should learn. Either 'Burgers' or 'KS'
            epoch (int): The amount of epochs the model was trained on.
            train_samples (int): The number of training samples used during training.
            save (bool): Whether to save the model immediately after initialization.
        """
        self.N = N
        self.L = L
        self.projection_type = projection_type
        self.equation_name = equation_name
        self.finite_element_space = finite_element_space
        self.train_samples = train_samples
        self.epoch = 0

        if finite_element_space == "HER":
            self.dof_count = 2 * N

        elif finite_element_space == "CG3":
            self.dof_count = 3 * N

        else:
            self.dof_count = N

        with fd.CheckpointFile(f"data/{equation_name}/meshes/N{N}.h5", "r") as f:
            self.mesh = f.load_mesh()

    @property
    @abstractmethod
    def network(self):
        pass

    @property
    @abstractmethod
    def param_num(self):
        pass

    def save(self):
        self.filename = f"data/{self.equation_name}/models/{self.projection_type}/N{self.N}" \
                        f"/D{self.network.D}_M{self.network.M}_samples{self.train_samples}_epoch{self.epoch}.pt"

        config = {
            "M": self.network.M,
            "D": self.network.D,
            "depth": self.network.depth,
            "N": self.N,
            "projection_type": self.projection_type,
            "finite_element_space": self.finite_element_space,
            "epoch": self.epoch,
            "train_samples": self.train_samples
        }

        torch.save({
            "state_dict": self.network.state_dict(),
            "config": config
        }, self.filename)

    @staticmethod
    def load(filename, equation_name, device):
        state_dict, config = torch.load(filename).values()

        mesh = fd.PeriodicIntervalMesh(config["N"], 1)

        projection = ProjectionCoefficient.load(
            f"data/{equation_name}/projection_coefficients/{finite_element_space}"
            f"/{config['projection_type']}/N{config['N']}_M{config['M']}.pt",
            mesh, device)
        network = NonlocalNeuralOperator(config["M"], config["D"], config["depth"], projection, device)
        network.load_state_dict(state_dict)

        model = NeuralOperatorModel(network, equation_name, config["finite_element_space"], config["epoch"],
                                    config["train_samples"], save=False)
        model.filename = filename

        return model


class BurgersModel(NeuralOperatorModel):
    network, param_num = None, None

    def __init__(self, N, M, D, depth, projection_type, train_samples=1000, device="cuda", dtype=torch.float64):
        super().__init__(N, 1, projection_type, "burgers", "CG1", train_samples)

        try:
            self.projection = ProjectionCoefficient.load(self.mesh, "burgers", N, self.L, M, "CG1", projection_type,
                                                         device)
        except FileNotFoundError:
            self.projection = ProjectionCoefficient(self.mesh, N, self.L, M, "CG1", projection_type, device)
            self.projection.calculate(
                f"data/burgers/projection_coefficients/CG1/{projection_type}/N{N}_M{M}.pt")

        self.network = NeuralOperatorNetwork(M, D, depth, self.projection, device)
        self.param_num = sum(p.numel() for p in self.network.parameters() if p.requires_grad)


class KSModel(NeuralOperatorModel):
    network, param_num = None, None

    def __init__(self, N, M, D, depth, projection_type, finite_element_family, train_samples=1000, device="cuda",
                 dtype=torch.float64):
        super().__init__(N, 10, "fourier", "KS", finite_element_family, train_samples)

        try:
            self.projection = ProjectionCoefficient.load(self.mesh, "KS", N, self.L, M, finite_element_family,
                                                         projection_type,
                                                         device)
        except FileNotFoundError:
            self.projection = ProjectionCoefficient(self.mesh, N, self.L, M, finite_element_family, projection_type,
                                                    device)
            self.projection.calculate(
                f"data/{equation_name}/projection_coefficients/{finite_element_family}/{projection_type}/N{N}_M{M}.pt")

        self.network = NeuralOperatorNetwork(M, D, depth, self.projection, device)
        self.param_num = sum(p.numel() for p in self.network.parameters() if p.requires_grad)


# Usage Example
if __name__ == "__main__":
    N, M, D, depth, equation_name = 128, 8, 10, 3, "KS"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    BurgersModel(N, M, D, depth, "fourier", device=device)
    KSModel(N, M, D, depth, "fourier", "HER", device=device)

    # mesh = fd.PeriodicIntervalMesh(N, 1)
    # projection = ProjectionCoefficient(mesh, equation_name, "fourier", M, device)
    # projection.calculate()
    # network = NonlocalNeuralOperator(M, D, depth, projection, device)
    #
    # # Create and save the model
    # model1 = NeuralOperatorModel(network, equation_name, save=True)
    #
    # # Load the model
    # model2 = NeuralOperatorModel.load(f"data/{equation_name}/models/fourier/N100/D{D}_M{M}_samples1000_epoch0.pt",
    #                                   equation_name, device)
    #
    # print(model1.network.state_dict()['lifting.weight'] == model2.network.state_dict()['lifting.weight'])
