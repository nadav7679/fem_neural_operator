import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset


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
              nclass (int): Number of classes in the output.
              width (int): Width of the hidden layers.
              depth (int): Number of hidden layers.

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
        """
          Forward pass of the neural network.

          Args:
              x (torch.Tensor): Input tensor. A list of coefficients.

          Returns:
              torch.Tensor: the output of the network for given input.
        """

        u = self.lifting(u)

        for i in range(self.depth):
            u = self.layers[i](u)

        return self.projection(u)


# def test_net(net=None):
#     """
#       Test the Net class by creating an instance and making a forward pass with a sample.
#
#       Args:
#           net (Net, optional): A pre-trained Net instance. If None, create a new instance.
#     """
#
#     mnist_net = Net((28, 28), 10, 16, 2) if net is None else net
#     sample_index = np.random.randint(10000)
#
#     x = train_set_mnist.data[sample_index, :, :]
#     x = torch.unsqueeze(x, 0)
#     print(mnist_net(x), train_set_mnist.targets[sample_index])
#
#


class NeuralNetworkTrainer():
    def __init__(
            self,
            trainset,
            testset,
            width,
            depth,
            criterion,  # Notice we assume that reduction="mean"
            optimizer,
            batch_size=64,
            lr=0.001,
            max_epoch=1,
            normalize=True
    ):
        """
          A high-level class for creating, training, and evaluating neural networks on MNIST or CIFAR datasets.

          Args:
              width (int): Width of the hidden layers in the neural network. Hyperparamater!
              depth (int): Number of hidden layers in the neural network. Hyperparamater!
              criterion (torch loss function): Loss function for training. Assume the loss uses reduction="mean" !
              optimizer (torch.optim.Optimizer): Optimization algorithm for training.
              batch_size (int, optional): Batch size for training and testing. Default is 64. Hyperparamater!
              lr (float, optional): Learning rate for the optimizer. Default is 0.001. Hyperparamater!
              max_epoch (int, optional): Maximum number of training epochs. Default is 1. Hyperparamater!
              normalize (bool, optional): If True, normalize the data. Default is True.
        """

        self.trainset = trainset
        self.testset = testset
        self.batch_size = batch_size
        self.trainloader, self.testloader = self.loading_data()

        self.net = Net(dim, nclass, width, depth)

        self.lr = lr
        self.max_epoch = max_epoch
        self.optimizer = optimizer(self.net.parameters(), lr=self.lr)
        self.criterion = criterion

    def loading_data(self):  # Notice that all of the required arguments are now attributes!
        """
          Create DataLoader instances for the training and testing datasets.

          Returns:
              tuple: Tuple containing DataLoader instances for training and testing.
        """

        trainloader = DataLoader(self.trainset, self.batch_size, shuffle=True)
        testloader = DataLoader(self.testset, self.batch_size, shuffle=False)

        return trainloader, testloader

    def train_epoch(self):  # Notice that all of the required arguments are now attributes!
        """
          Perform one training epoch.

          Returns:
              torch.Tensor: Training loss for the epoch.
        """

        self.net.train()

        for X, y in self.trainloader:
            y_hat = self.net(X)
            local_loss = self.criterion(y_hat, y)

            local_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return self.criterion(
            self.net(self.trainloader.dataset.data),
            self.trainloader.dataset.targets
        )

    def test_epoch(self):
        """
          Perform one testing epoch.

          The error is computed as follows:
          torch.max(..., 1)[1] is doing argmax over each row and returns a list if ints, s.t.
          each int corresponds to the index that has maximal value in this row.
          argmax on a prediction returns the most likely class, argmax on the targets give the target.
          So, the number of zeros in the expression argmax(predict) - argmax(target) will give back the
          number of succsesful predictions, and the number of nonzero elements will give the errors!
          Example (batch=2, nclass=3):

          predictions = [[0.6, 0.4, 0.,],
                        [0.7, 0.2, 0.1,]]

          targets =     [[1, 0., 0.,],
                        [0., 1, 0.,]]

          argmax(predictions) - argmax(targets) = [0 - 0, 0 - 1] = [0, -1]
          => number of errors = numbers of nonzero elements = 1



          Returns:
              tuple: Tuple containing testing loss and number of classification errors.
        """
        self.net.eval()
        y = self.testloader.dataset.data
        targets = self.testloader.dataset.targets

        y_hat = self.net(y)
        mean_loss = self.criterion(y_hat, targets)  # Asuuming reduction="mean"

        target_class = torch.max(targets, 1)[1]
        predicted_class = torch.max(y_hat, 1)[1]  # Argmax gives predicted_class

        num_errors = len(torch.nonzero(predicted_class - target_class))

        return mean_loss, num_errors

    def train_me(self, logs=True):
        """
          Train the neural network for  max_epoch and print training and testing statistics.
        """

        samples_len = self.testset.data.shape[0]
        for i in range(self.max_epoch):
            epoch = i + 1
            train_loss = self.train_epoch()
            test_loss, test_err = self.test_epoch()

            if logs:
                print(f"Epoch: {epoch} | Train Loss: {train_loss:.04} |"
                      f"Test Loss: {test_loss:.04} | Test Error: {test_err / samples_len:.04}")

        return np.array([float(train_loss), float(test_loss)])


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

