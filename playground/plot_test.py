import torch
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt

from burgers import BurgersDataset


if __name__ == "__main__":
    nx = 100
    max_modes = 10
    d = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filename = f"data/burgers__samples_1000__nx_{nx}"

    net = torch.load(f"models/nx_{nx}__d_{d}__max_modes_{max_modes}.pt").to(device=device)
    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of parameters: {param_num}")

    data = torch.load(f"{filename}.pt").unsqueeze(2).to(device=device, dtype=torch.float32)
    grid = torch.linspace(0, 1, nx, device=device)
    testset = BurgersDataset(data[int(0.8 * len(data)):], grid)

    prediction_coeffs = net(testset[:][0]).detach().cpu()

    with fd.CheckpointFile(f"{filename}__mesh.h5", "r") as file:
        function_space = fd.FunctionSpace(file.load_mesh(), "CG", degree=1)

    initial_functions = []
    solutions = []
    predictions = []
    for predict, datapoint in zip(prediction_coeffs, testset):
        initial_functions.append(fd.Function(function_space, val=datapoint[0][0].detach().cpu()))
        solutions.append(fd.Function(function_space, val=datapoint[1][0].detach().cpu()))
        predictions.append(fd.Function(function_space, val=predict[0]))

    # Number of points for plotting
    num_plot_points = 1000
    plot_points = np.linspace(0, 1, num_plot_points)

    # Create a figure for plotting
    fig, ax = plt.subplots()

    # Plot each function
    index = 40
    for func, label in zip([solutions[index], predictions[index], initial_functions[index]], ["Firedrake solution", "Network solution", "Initial condition"]):
        # Evaluate the function at the plot points
        plot_values = np.array([func.at(x) for x in plot_points])

        # Plot the function
        ax.plot(plot_points, plot_values, label=label)

    # Set plot labels and title
    ax.set_xlabel("$x$")
    ax.set_title(f"Burgers Firedrake and Net solutions, $N_x={nx}$")
    ax.legend()

    # Show the plot
    plt.show()
