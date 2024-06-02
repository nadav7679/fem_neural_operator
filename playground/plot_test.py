import torch
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filename = "data/burgers__samples_1000__nx_100"

    net = torch.load("model.pt").to(device=device)
    data = torch.load(f"{filename}.pt").to(device=device, dtype=torch.float32)

    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of parameters: {param_num}")

    with fd.CheckpointFile(f"{filename}__mesh.h5", "r") as file:
        mesh = file.load_mesh()

    function_space = fd.FunctionSpace(mesh, "CG", degree=1)

    simulation_coeff = net(data[:, 0, None, :]).cpu().detach().numpy()
    data = data.to(device="cpu").numpy()

    initial_functions = []
    solutions = []
    simulations = []
    for i, arr in enumerate(data):
        initial_functions.append(fd.Function(function_space, val=arr[0]))
        solutions.append(fd.Function(function_space, val=arr[1]))
        simulations.append(fd.Function(function_space, val=simulation_coeff[i, 0, :]))

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
