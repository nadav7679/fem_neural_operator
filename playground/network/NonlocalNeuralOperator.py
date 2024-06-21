import torch
import torch.nn as nn
import torch.nn.functional as functional


class NeuralOperatorLayer(nn.Module):
    def __init__(
            self,
            device,
            modes: int,
            channels: int,
            coeff: torch.Tensor,
    ):
        """
        A network module that performs the interior layers operation in the Nonlocal Neural Operator architecture.
        This includes: Linear transformation with bias, preprojection linear transformation, projection to 'psi'
        functions (multiplication by coefficient), and nonlinearity.
        Args:
            device: Torch device to use (gpu or cpu)
            dim: Dimension of the FE space, i.e. N
            coeff: An M x N matrix of projection inner products, Cmn = <psi_m, phi_n>. M is the number of psi functions.
        """
        super().__init__()

        self.coeff = coeff
        self.coeff_T = coeff.T
        self.modes = modes

        #: Linear matrix multiplication that mixes up the channels (W operator), called also MLP. It includes the bias.
        self.linear = nn.Conv1d(channels, channels, kernel_size=1, device=device)

        self.weights = nn.Parameter(torch.rand(modes, channels, channels, requires_grad=True, device=device))  # M x D x D parameters

    def forward(self, u):
        wu = self.linear(u)
        # print(u.shape, self.weights.shape, (u @ self.coeff_T).shape, self.coeff.shape)
        s = torch.einsum("mji, bim, mn -> bjn", self.weights, u @ self.coeff_T, self.coeff)

        return functional.gelu(wu + s)


class NonlocalNeuralOperator(nn.Module):
    def __init__(
            self,
            device,
            modes: int,
            dim: int,
            channels: int,
            depth: int,
            coeff: torch.Tensor
    ):
        """
        A 1D nonlocal neural operator on finite dimensions (intended for FEM). As the function spaces are finite,
        the network's data is expansion coefficients functions in a given basis.

        Args:
            device: Torch device to use
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

        self.lifting = nn.Conv1d(2, channels, 1, device=device)

        layers = []
        for _ in range(depth):
            layers.append(NeuralOperatorLayer(device, modes, channels, coeff))

        self.layers = nn.ModuleList(layers)

        self.projection = nn.Conv1d(channels, 1, 1, device=device)

    def forward(self, u):
        u = self.lifting(u)

        for i in range(self.depth):
            u = self.layers[i](u)

        return self.projection(u)


if __name__ == "__main__":
    N = 100
    M = 8
    d = 10
    batchsize = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coeff = torch.randn((M, N), device=device)

    u = torch.randn((batchsize, 2, N), device=device)  # Unsqueeze to add channel dimension

    # projection_block = NeuralOperatorLayer(N, coeff)
    # print(projection_block(u).shape)

    model = NonlocalNeuralOperator(device, M, N, d, 3, coeff)
    u = model(u)
    print(u, u.shape)
