import torch
from torch.utils.data import DataLoader


class NeuralNetworkTrainer():
    def __init__(
            self,
            net,
            trainset,
            testset,
            criterion,
            optimizer,
            batch_size=32,
            lr=0.001,
            max_epoch=1,
    ):
        """
          A high-level class for creating, training, and evaluating neural network.

          Args:
              criterion (torch loss function): Loss function for training. Assume the loss uses reduction="sum" !
              optimizer (torch.optim.Optimizer): Optimization algorithm for training.
              batch_size (int, optional): Batch size for training and testing. Default is 32. Hyperparamater!
              lr (float, optional): Learning rate for the optimizer. Default is 0.001. Hyperparamater!
              max_epoch (int, optional): Maximum number of training epochs. Default is 1. Hyperparamater!
        """

        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer(net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch

        self.trainloader = DataLoader(trainset, batch_size, shuffle=True)
        self.testloader = DataLoader(testset, batch_size, shuffle=False)

    def train_epoch(self):
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

          Returns:
              tuple: Testing loss.
        """
        self.net.eval()
        y = self.testloader.dataset.data
        targets = self.testloader.dataset.targets

        y_hat = self.net(y)
        tot_loss = self.criterion(y_hat, targets)

        return tot_loss

    def train_me(self, logs=True):
        """
          Train the neural network for max_epoch and print training and testing statistics.
        """

        for i in range(self.max_epoch):
            epoch = i + 1
            train_loss = self.train_epoch()
            test_loss = self.test_epoch()

            if logs:
                print(f"Epoch: {epoch} | Train Loss: {train_loss:.04} | Test Loss: {test_loss:.04}")

        return float(train_loss)
