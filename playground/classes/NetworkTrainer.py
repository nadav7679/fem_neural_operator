import torch
from torch.utils.data import DataLoader


class NeuralNetworkTrainer():
    def __init__(
            self,
            model,
            trainset,
            testset,
            loss,
            optimizer,
            scheduler,
            batch_size=32,
            max_epoch=1,
    ):
        """
        A high-level class for creating, training, and evaluating neural network models.

        Args:
            model (NeuralOperatorModel): The neural network model to be trained and evaluated.
            trainset (torch.utils.data.Dataset): The training dataset.
            testset (torch.utils.data.Dataset): The testing dataset.
            optimizer (torch.optim.Optimizer): Optimization algorithm for training.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            batch_size (int, optional): Batch size for training and testing. Default is 32.
            lr (float, optional): Learning rate for the optimizer. Default is 0.001.
            max_epoch (int, optional): Maximum number of training epochs. Default is 1.
        """

        self.model = model
        self.trainloader = DataLoader(trainset, batch_size, shuffle=True)
        self.testloader = DataLoader(testset, batch_size, shuffle=False)
        self.criterion = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.max_epoch = max_epoch

    def train_epoch(self):
        """
          Perform one training epoch.

          Returns:
              torch.Tensor: Training loss for the epoch.
        """

        self.model.network.train()

        for X, y in self.trainloader:
            y_hat = self.model.network(X)
            local_loss = self.criterion(y_hat, y)

            local_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.model.epoch += 1
        return self.criterion(
            self.model.network(self.trainloader.dataset.inputs),
            self.trainloader.dataset.targets
        )

    def test_epoch(self):
        """
          Perform one testing epoch.

          Returns:
              torch.Tensor: Testing loss.
        """
        self.model.network.eval()
        y = self.testloader.dataset.inputs
        targets = self.testloader.dataset.targets

        y_hat = self.model.network(y)
        tot_loss = self.criterion(y_hat, targets)

        return tot_loss

    def train_me(self, logs=True, save=True):
        """
        Train the neural network for max_epoch and print training and testing statistics.

        Args:
            logs (bool, optional): Whether to print logs during training. Default is True.
            save (bool, optional): Whether to save the model after training. Default is True.

        Returns:
            torch.Tensor: Tensor containing training and testing losses for each epoch.
        """
        losses = torch.zeros((2, self.max_epoch), dtype=torch.float32, device="cpu")

        for i in range(self.max_epoch):
            epoch = i + 1
            train_loss = self.train_epoch().detach().cpu()
            test_loss = self.test_epoch().detach().cpu()

            if logs:
                print(f"Epoch: {epoch} | Train Loss: {train_loss:.04} | Test Loss: {test_loss:.04} "
                      f"| lr: {self.scheduler.get_last_lr()[0]}")

            losses[0, i], losses[1, i] = train_loss, test_loss
            self.scheduler.step()

        if save:
            self.model.save()

        return losses
