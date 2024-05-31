import torch
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    filename = "data/burgers__samples_1000__nx_100"

    net = torch.load("model.pt")
    data = torch.load(f"{filename}.pt")

    with fd.CheckpointFile(f"{filename}__mesh.h5", "r") as file:
        mesh = file.load_mesh()

    function_space = fd.FunctionSpace(mesh, "CG", degree=1)

    simulation_coeff = net(data[:, 0, None, :].to(dtype=torch.float32))

    initial_functions = []
    solutions = []
    simulations = []
    for i, arr in enumerate(data):
        initial_functions.append(fd.Function(function_space, val=arr[0]))
        solutions.append(fd.Function(function_space, val=arr[1]))
        simulations.append(fd.Function(function_space, val=simulation_coeff[i, 0, :].detach().numpy()))


    # Number of points for plotting
    num_plot_points = 1000
    plot_points = np.linspace(0, 1, num_plot_points)

    # Create a figure for plotting
    fig, ax = plt.subplots()

    # Plot each function
    for func in [solutions[-1], simulations[-1]]:
        # Evaluate the function at the plot points
        plot_values = np.array([func.at(x) for x in plot_points])

        # Plot the function
        ax.plot(plot_points, plot_values)

    # Set plot labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('Value')
    ax.set_title('Multiple Firedrake Functions')
    ax.legend()

    # Show the plot
    plt.show()

