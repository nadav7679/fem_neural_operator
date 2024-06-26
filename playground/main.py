import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from network import NeuralNetworkTrainer, NonlocalNeuralOperator
from burgers import BurgersDataset, fourier_coefficients


def train_network(nx, d, max_modes):
    h = 1 / nx
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filename = f"data/burgers__samples_1000__nx_{nx}"

    samples = torch.load(f"{filename}.pt").unsqueeze(2).to(device=device, dtype=torch.float32)

    try:
        coeff = torch.load(f"models_L2/nx_{nx}__d_{d}__max_modes_{max_modes}__coeff.pt").to(device=device,
                                                                                            dtype=torch.float32)

    except FileNotFoundError:
        coeff = fourier_coefficients(filename, max_modes, nx, d).to(device=device, dtype=torch.float32)

    grid = torch.linspace(0, 1, nx, device=device)
    samples_len = samples.shape[0]
    trainset = BurgersDataset(samples[:int(0.8 * samples_len)], grid)
    testset = BurgersDataset(samples[int(0.8 * samples_len):], grid)

    dim = coeff.shape[1]
    depth = 3
    net = NonlocalNeuralOperator(device, 2 * max_modes + 1, dim, d, depth, coeff).to(dtype=torch.float32)
    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of parameters: {param_num}")
    print(f"Used device: {device}")

    lr = 0.01
    mse_loss = nn.MSELoss(reduction="sum")
    loss = lambda x, y: h / len(x) * mse_loss(x, y)  # Sum of all differences, times step size, divide by batch size
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5)
    network_trainer = NeuralNetworkTrainer(
        net,
        trainset,
        testset,
        loss,
        optimizer,
        scheduler,
        max_epoch=500
    )

    losses = network_trainer.train_me()
    torch.save(net, f"models_L2/nx_{nx}__d_{d}__max_modes_{max_modes}.pt")

    # plt.plot(losses[0], label="Train loss")
    # plt.plot(losses[1], label="Test loss")
    # plt.title(f"L1 loss for Burgers with nx={nx}")
    # plt.xlabel("Epoch")
    # plt.yscale("log")
    # plt.grid()
    # plt.legend()
    # plt.show()
    return net, network_trainer


if __name__ == "__main__":
    for d in range(40, 45, 5):
        train_network(512, d, 8)
