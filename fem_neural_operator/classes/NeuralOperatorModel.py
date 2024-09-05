from abc import ABC, abstractmethod

import firedrake as fd
import torch
import torch.nn as nn

from .NeuralOperatorNetwork import NeuralOperatorNetwork
from .ProjectionCoefficient import ProjectionCoefficient
from .NetworkTrainer import NeuralNetworkTrainer
from .Dataset import Dataset, KSHerDataset


class NeuralOperatorModel(ABC):
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

        with fd.CheckpointFile(f"/home/clustor/ma/n/np923/fem_neural_operator/fem_neural_operator/data/KS/meshes/N{N}.h5", "r") as f:
            self.mesh = f.load_mesh()

    def train(self, data_path, max_epoch, lr=0.01, optimizer = None, scheduler=None, device="cuda"):
        samples = torch.load(data_path).unsqueeze(2).to(device=device, dtype=torch.float32)
        
        if self.finite_element_space == "CG3":
            cg3 = fd.FunctionSpace(self.mesh, "CG", 3)
            x = fd.assemble(fd.interpolate(fd.SpatialCoordinate(self.mesh)[0], cg3))
            grid = torch.tensor(sorted(x.dat.data[:])).to(device=device, dtype=torch.float32)
        
        else:
            grid = torch.linspace(0, self.L, self.dof_count, device=device)
        
        
        if self.finite_element_space == "HER": 
            trainset = KSHerDataset(self.N, torch.tensor(samples[:self.train_samples]), torch.tensor(grid))
            testset = KSHerDataset(self.N, torch.tensor(samples[self.train_samples:]), torch.tensor(grid))
                
        else:
            trainset = Dataset(torch.tensor(samples[:self.train_samples]), torch.tensor(grid))
            testset = Dataset(torch.tensor(samples[1000:]), torch.tensor(grid))
        
        
        mean_rel_l2_loss = lambda x, y: torch.mean(torch.norm(x - y, 2, dim=-1)/torch.norm(y, 2, dim=-1))
        # mse_loss = nn.MSELoss(reduction="sum")
        # loss = lambda x, y: mse_loss(x, y) / (self.N * x.shape[0])  # Sum of differences, times step size, divide by batch size

        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr) if optimizer is None else optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5) if scheduler is None else scheduler

        network_trainer = NeuralNetworkTrainer(
            self,
            trainset,
            testset,
            mean_rel_l2_loss,
            optimizer,
            scheduler,
            max_epoch=max_epoch
        )

        return network_trainer.train_me()

    @property
    @abstractmethod
    def network(self):
        pass

    @property
    @abstractmethod
    def param_num(self):
        pass

    def save(self):
        self.filename = f"../../data/{self.equation_name}/models/{self.finite_element_space}/{self.projection_type}/N{self.N}" \
                        f"/T{self.T}/D{self.network.D}_M{self.network.M}_samples{self.train_samples}_epoch{self.epoch}.pt"

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
    def load(filename, N, L, M, equation_name, finite_element_space, device):
        state_dict, config = torch.load(filename).values()

        with fd.CheckpointFile(f"data/{equation_name}/meshes/N{N}.h5", "r") as f:
            mesh = f.load_mesh()
            
            
        projection = ProjectionCoefficient.load(mesh, equation_name, N, L, M, finite_element_space, "fourier", device=device)
        
        network = NeuralOperatorNetwork(config["M"], config["D"], config["depth"], projection, device)
        network.load_state_dict(state_dict)

        model = self.__init__(network, equation_name, config["finite_element_space"], config["epoch"],
                                    config["train_samples"], save=False)

        return model


class BurgersModel(NeuralOperatorModel):
    network, param_num = None, None

    def __init__(self, N, M, D, depth, T, projection_type, train_samples=1000, device="cuda", dtype=torch.float32):
        super().__init__(N, 1, projection_type, "burgers", "CG1", train_samples)
        
        self.T = T

        try:
            self.projection = ProjectionCoefficient.load(self.mesh, "burgers", N, self.L, M, "CG1", projection_type,
                                                         device)
        except FileNotFoundError:
            self.projection = ProjectionCoefficient(self.mesh, N, self.L, M, "CG1", projection_type, device)
            self.projection.calculate(
                f"data/burgers/projection_coefficients/CG1/{projection_type}/N{N}_M{M}.pt")

        self.network = NeuralOperatorNetwork(M, D, depth, self.projection, device)
        self.param_num = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        
    @staticmethod
    def load(filename, N, T, device):
        state_dict, config = torch.load(filename, map_location=device).values()

        with fd.CheckpointFile(f"/home/clustor/ma/n/np923/fem_neural_operator/fem_neural_operator/data/burgers/meshes/N{N}.h5", "r") as f:
            mesh = f.load_mesh()
                    
        
        model = BurgersModel(config["N"], config["M"], config["D"], config["depth"], T, config["projection_type"], config["train_samples"], device=device)
        model.network.load_state_dict(state_dict)

        return model




class KSModel(NeuralOperatorModel):
    network, param_num = None, None

    def __init__(self, N, M, D, depth, T, projection_type, finite_element_family, train_samples=1000, device="cuda",
                 dtype=torch.float32):
        super().__init__(N, 10, "fourier", "KS", finite_element_family, train_samples)
        
        self.T = T

        try:
            self.projection = ProjectionCoefficient.load(self.mesh, "KS", N, self.L, M, finite_element_family,
                                                         projection_type,
                                                         device)
        except FileNotFoundError:
            self.projection = ProjectionCoefficient(self.mesh, N, self.L, M, finite_element_family, projection_type,
                                                    device)
            self.projection.calculate(
                f"/home/clustor/ma/n/np923/fem_neural_operator/fem_neural_operator/data/KS/projection_coefficients/{finite_element_family}/{projection_type}/N{N}_M{M}.pt")

        self.network = NeuralOperatorNetwork(M, D, depth, self.projection, device)
        self.param_num = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
    
    @staticmethod
    def load(filename, N, T, device):
        state_dict, config = torch.load(filename, map_location=device).values()

        # with fd.CheckpointFile(f"/home/clustor/ma/n/np923/fem_neural_operator/fem_neural_operator/data/KS/meshes/N{N}.h5", "r") as f:
        #     mesh = f.load_mesh()
            
            
        # projection = ProjectionCoefficient.load(mesh, equation_name, N, L, M, finite_element_space, "fourier", device=device)
        
        
        model = KSModel(config["N"], config["M"], config["D"], config["depth"], T, config["projection_type"], config["finite_element_space"], device=device)
        model.network.load_state_dict(state_dict)

        return model



# Usage Example
if __name__ == "__main__":
    N, M, D, depth, equation_name = 2048, 8, 10, 3, "KS"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    BurgersModel(N, M, D, depth, "fourier", device=device)
    ks_model = KSModel(N, M, D, depth, "fourier", "HER", device=device)

    ks_model.train(f"data/KS/samples/N{N}_HER_nu0029_T01_samples1200.pt", 10, device=device)

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
