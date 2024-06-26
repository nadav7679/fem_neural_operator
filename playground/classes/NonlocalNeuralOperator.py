import torch
import torch.nn as nn
import torch.nn.functional as functional
import firedrake as fd

from ProjectionCoefficient import ProjectionCoefficient


class NeuralOperatorLayer(nn.Module):
    def __init__(
            self,
            M: int,
            D: int,
            projection: ProjectionCoefficient,
            device: str
    ):
        """
        A network module that performs the interior layers operation in the Nonlocal Neural Operator architecture.
        This includes: Linear transformation with bias, preprojection linear transformation, projection to 'psi'
        functions (multiplication by coefficient), and nonlinearity.

        Args:
            M (int): Number of Fourier modes.
            D (int): Number of channels for fan-out.
            projection (ProjectionCoefficient): An instance of ProjectionCoefficient.
            device (str): Torch device to use ('cpu' or 'cuda').
        """
        super().__init__()

        self.coeff = projection.coeff
        self.coeff_T = projection.coeff.T

        # Linear matrix multiplication that mixes up the channels (W operator), also called MLP. It includes the bias.
        self.linear = nn.Conv1d(D, D, kernel_size=1, device=device)

        self.weights = nn.Parameter(torch.rand(2*M + 1, D, D, requires_grad=True, device=device))  # M x D x D parameters

    def forward(self, u):
        wu = self.linear(u)
        s = torch.einsum("mji, bim, mn -> bjn", self.weights, u @ self.coeff_T, self.coeff)

        return functional.gelu(wu + s)


class NonlocalNeuralOperator(nn.Module):
    def __init__(
            self,
            M: int,
            D: int,
            depth: int,
            projection: ProjectionCoefficient,
            device: str = 'cpu'
    ):
        """
        A 1D nonlocal neural operator on finite dimensions (intended for FEM). As the function spaces are finite,
        the network's data is expansion coefficients functions in a given basis.

        Args:
            M (int): Number of Fourier modes.
            D (int): Number of channels for fan-out.
            depth (int): Number of inner layers.
            projection (ProjectionCoefficient): An instance of ProjectionCoefficient.
            device (str): Torch device to use ('cpu' or 'cuda').

        """
        super().__init__()
        self.M = M
        self.D = D
        self.depth = depth
        self.projection = projection

        self.lifting = nn.Conv1d(2, D, 1, device=device)
        self.lowering = nn.Conv1d(D, 1, 1, device=device)

        layers = []
        for _ in range(depth):
            layers.append(NeuralOperatorLayer(M, D, projection, device))

        self.layers = nn.ModuleList(layers)

    def forward(self, u):
        u = self.lifting(u)

        for i in range(self.depth):
            u = self.layers[i](u)

        return self.lowering(u)


if __name__ == "__main__":
    N = 100
    M = 8
    D = 10
    batchsize = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh = fd.PeriodicIntervalMesh(N, 1)
    projection = ProjectionCoefficient(mesh, 'fourier', M, device)
    projection.calculate(save=False)

    u = torch.randn((batchsize, 2, N), device=device)  # Unsqueeze to add channel dimension

    model = NonlocalNeuralOperator(M, D, 3, projection, device)
    u = model(u)
    print(u, u.shape)
