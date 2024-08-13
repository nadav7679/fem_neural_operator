import torch
import torch.nn as nn
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt

from burgers import BurgersDataset
from classes import ProjectionCoefficient, NeuralOperatorLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DummyInnerLayer(NeuralOperatorLayer):
    def __init__(
            self,
            M: int,
            D: int,
            projection: ProjectionCoefficient,
            device: str
    ):
        super().__init__(M, D, projection, device)
        del self.linear, self.weights  # Torch is angry if you have multiple tracked parameters with the same name

        self.identity = nn.Identity(device=device)
        self.linear = lambda x: 0 * self.identity(x)  # Ignore linear part
        self.activation = lambda x: x  # Identity
        self.weights = torch.eye(D, D).unsqueeze(0).repeat(2 * M + 1, 1, 1).to(device=device)  # Identity matrices T_m=I


N = 1024
D = 64
M_arr = [2, 8, 32]


# ------------ Creating firedrake functions and storing in torch ----------------------#

with fd.CheckpointFile(f"data/burgers/meshes/N{N}.h5", "r") as file:
    mesh = file.load_mesh()

fs = fd.FunctionSpace(mesh, "CG", 1)
x = fd.SpatialCoordinate(mesh)[0]

func1 = fd.Function(fs)
func1.interpolate(-10 * (x-0.5)**2)
func1.interpolate(fd.sign(x-0.5) * (x-0.5))

func2 = fd.Function(fs)
func2.interpolate(fd.tan(x))

input_u_coeff = torch.tensor(func1.dat.data).unsqueeze(0).repeat(D, 1).unsqueeze(0).to(device=device, dtype=torch.float32)  # Duplicating the function D times
input_u_coeff[0, 10, :] = torch.tensor(func2.dat.data).to(device=device, dtype=torch.float32)  # Inserting a different function at D=1- just for fun

# ------------ Creating dummy layers and plotting ----------------------#
plt.figure(figsize=(7, 4))

domain = np.linspace(0, 1, N)
plt.plot(domain, input_u_coeff[0, 10, :].cpu().detach(), label="Input", color="black")

for M in M_arr:
    projection = ProjectionCoefficient(mesh, N, 1, M, "CG1", "fourier", device=device)
    projection.calculate()
    layer = DummyInnerLayer(M, D, projection, device)

    plt.plot(domain, (layer(input_u_coeff))[0, 10, :].cpu().detach(), label=f"M={M}", linestyle="--", linewidth=2)

plt.legend()
plt.xlabel("x")
plt.savefig("media/DummyNeuralOperator")
plt.show()
