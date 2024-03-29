import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset


class SpectralConv1d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            n_layers,
            n_modes,
            max_modes=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes  # Total number of modes kept along each dimension
        self.n_layers = n_layers
        self.max_modes = n_modes if max_modes is None else max_modes


class NonlocalOperatorNet(nn.Module):
    def __init__(
            self,
            in_length,
            lift_channels,
            nclass,
            width,
            depth
    ):
        """
          A 1D nonlocal neural operator.

          Args:
              dim (tuple): Input dimensions (e.g., (28, 28) for MNIST).
              nclass (int): Number of classes in the output.
              width (int): Width of the hidden layers.
              depth (int): Number of hidden layers.
        """
        super().__init__()
        self.in_length = in_length
        self.lift_channels = lift_channels
        self.nclass = nclass
        self.width = width
        self.depth = depth

        self.lifting = nn.Conv1d(in_length, lift_channels, 1)

        self.linear_in = nn.Linear(self.input_length, width, device=device)
        self.linear_hidden = nn.Linear(width, width, device=device)
        self.relu = nn.ReLU()
        self.linear_out = nn.Linear(width, nclass, device=device)

    def forward(self, x):
        """
          Forward pass of the neural network.

          Args:
              x (torch.Tensor): Input tensor. A list of coefficients.

          Returns:
              torch.Tensor: the output of the network for given input.
        """

        x = self.lifting(x)

        processed_x = lifted_x
        for _ in range(self.depth):
            processed_x = self.relu(self.linear_hidden(processed_x))

        return self.linear_out(processed_x)


def test_net(net=None):
    """
      Test the Net class by creating an instance and making a forward pass with a sample.

      Args:
          net (Net, optional): A pre-trained Net instance. If None, create a new instance.
    """

    mnist_net = Net((28, 28), 10, 16, 2) if net is None else net
    sample_index = np.random.randint(10000)

    x = train_set_mnist.data[sample_index, :, :]
    x = torch.unsqueeze(x, 0)
    print(mnist_net(x), train_set_mnist.targets[sample_index])


# test_net()


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
