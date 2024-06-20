import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from burgers import BurgersDataset, fourier_coefficients
from network import NeuralNetworkTrainer, NonlocalNeuralOperator

nx = 256
h = 1 / nx
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
filename = f"data/burgers__samples_1000__nx_{nx}"
depth = 3

samples = torch.load(f"{filename}.pt").unsqueeze(2).to(device=device, dtype=torch.float32)
grid = torch.linspace(0, 1, nx, device=device)
samples_len = samples.shape[0]
trainset = BurgersDataset(samples[:int(0.8 * samples_len)], grid)
testset = BurgersDataset(samples[int(0.8 * samples_len):], grid)


lr = 0.01
my_loss = nn.MSELoss(reduction="sum")
loss = lambda x, y: h / len(x) * my_loss(x, y)  # Sum of all differences, times step size, divide by batch size

def main():
    d = 20
    modes = [2 ** i for i in range(7)]

    test_loss = []
    for m in modes:
        coeff = fourier_coefficients(filename, m, nx, d).to(device=device, dtype=torch.float32)
        dim = coeff.shape[1]

        net = NonlocalNeuralOperator(device, dim, d, depth, coeff).to(dtype=torch.float32)
        param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
        network_trainer = NeuralNetworkTrainer(
            net,
            trainset,
            testset,
            loss,
            optimizer,
            scheduler,
            max_epoch=200
        )

        print(f"Training m={m}, d={d} with param={param_num}")
        losses = network_trainer.train_me(logs=False)
        test_loss.append(losses[1, -1])
        torch.save(net, f"models_modes/nx_{nx}__d_{d}__max_modes_{m}.pt")

    return modes, test_loss


if __name__ == "__main__":
    modes, losses = main()

    plt.plot(modes, losses)
    plt.yscale("log")
    plt.show()

