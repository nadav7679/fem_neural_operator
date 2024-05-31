import torch
import torch.nn as nn
import torch.nn.functional as functional


class NeuralOperatorLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            coeff: torch.Tensor,
    ):
        """
        A network module that performs the interior layers operation in the Nonlocal Neural Operator architecture.
        This includes: Linear transformation with bias, preprojection linear transformation, projection to 'psi'
        functions (multiplication by coefficient), and nonlinearity.
        Args:
            dim: Dimension of the FE space, i.e. N
            coeff: An M x N matrix of projection inner products, Cmn = <psi_m, phi_n>. M is the number of psi functions.
        """
        super().__init__()

        #: Linear matrix multiplication that mixes up the channels (W operator), called also MLP. It includes the bias.
        self.linear = nn.Linear(dim, dim)
        #: The matrix multiplication before the inner product (the T_m, assuming T_m=T forall m).
        self.preprojection_linear = nn.Linear(dim, dim, bias=False)
        #: The matrix containing the inner product of phi and psi.
        self.coeff_squared = coeff.T @ coeff

    def forward(self, u):
        wu = self.linear(u)
        spectral_u = self.preprojection_linear(u) @ self.coeff_squared

        return functional.gelu(wu + spectral_u)


class NonlocalNeuralOperator(nn.Module):
    def __init__(
            self,
            dim: int,
            channels: int,
            depth: int,
            coeff: torch.Tensor
    ):
        """
        A 1D nonlocal neural operator on finite dimensions (intended for FEM). As the function spaces are finite,
        the network's data is expansion coefficients functions in a given basis.

        Args:
            dim: Dimension of FE space, i.e. N
            channels: Number of channels for fan-out, i.e. d
            depth: Number of inner layers
            coeff: Coefficient matrix of inner products of FE basis {phi}^N with projection functions {psi}^M

        Note: Currently the input is simply the `dim` coefficients of the function in the finite function-space basis.
        In general, the  lifting operator should get the function u(x)=x as well (I think). A possible change is to
        include this input as well: another `dim` coefficients that represent u(x) in the basis.
        """
        super().__init__()
        self.dim = dim
        self.channels = channels
        self.depth = depth

        self.lifting = nn.Conv1d(1, channels, 1)

        layers = []
        for _ in range(depth):
            layers.append(NeuralOperatorLayer(dim, coeff))

        self.layers = nn.ModuleList(layers)

        self.projection = nn.Conv1d(channels, 1, 1)

    def forward(self, u):
        u = self.lifting(u)

        for i in range(self.depth):
            u = self.layers[i](u)

        return self.projection(u)


if __name__ == "__main__":
    N = 100
    M = 17
    d = 10
    batchsize = 10

    coeff = torch.randn((M, N))

    u = torch.randn((batchsize, N)).unsqueeze(1)  # Unsqueeze to add channel dimension

    # projection_block = NeuralOperatorLayer(N, coeff)
    # print(projection_block(u).shape)

    model = NonlocalNeuralOperator(N, d, 4, coeff)
    u = model(u)
    print(u, u.shape)
