import torch
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt

from classes import ProjectionCoefficient


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, M, D = 512, 16, 10

    with fd.CheckpointFile(f"data/KS/meshes/N{N}.h5", "r") as file:
        mesh = file.load_mesh()

    x = fd.SpatialCoordinate(mesh)[0]
    fs = fd.FunctionSpace(mesh, "HER", 3)

    fs_out = fd.FunctionSpace(mesh, "CG", 3)

    projection = ProjectionCoefficient(mesh, "KS", "HER", "fourier", M, device)
    projection.calculate(False)

    initial_function = fd.Function(fs, name="initial")
    initial_function.interpolate(-(x-0.5)**2)

    fig, axes = plt.subplots()
    # fd.plot(initial_function, axes=axes)

    psi_arr = []
    for i in range(2*M + 1):
        if i == 0:
            psi_arr.append(fd.Function(fs, name=f"Mode {i}").interpolate(1))
            continue

        psi_arr.append(fd.Function(fs, name=f"Mode {i}").interpolate(2 * fd.sin(i * 2 * fd.pi * x)))
        psi_arr.append(fd.Function(fs, name=f"Mode {i}").interpolate(fd.cos(i * 2 * fd.pi * x)))



        # psi_arr.append(fd.Function(fs, val=130 * projection.coeff[i, :].cpu().detach().numpy(), name=f"Mode {i}"))

    for truncate in range(1, 17, 4):
        fourier = fd.Function(fs, name=f"Fourier {truncate}")
        for m in range(truncate * 2 + 1):
            m_coeff = 0
            for n in range(N):
                m_coeff += initial_function.dat.data[n] * projection.coeff[m, n].cpu().numpy()

            fourier += m_coeff * psi_arr[m]

        fd.plot(fourier, axes=axes)
    # fd.plot(psi_arr[0], axes=axes)
    # fd.plot(psi_arr[1], axes=axes)
    # fd.plot(psi_arr[2], axes=axes)
    # fd.plot(psi_arr[6], axes=axes)
    plt.legend()
    plt.show()



    # fd.plot(initial_function)
    # plt.show()
    # projection.coeff
    #
    # # Number of points for plotting
    # num_plot_points = 1000
    # plot_points = np.linspace(0, 1, num_plot_points)
    #
    # # Create a figure for plotting
    # fig, ax = plt.subplots()
    #
    # # Plot each function
    # index = 25
    # for func, label in zip([solutions[index], predictions[index], initial_functions[index]], ["Firedrake solution", "Network solution", "Initial condition"]):
    #     # Evaluate the function at the plot points
    #     plot_values = np.array([func.at(x) for x in plot_points])
    #
    #     # Plot the function
    #     ax.plot(plot_points, plot_values, label=label)
    #
    # # Set plot labels and title
    # ax.set_xlabel("$x$")
    # ax.set_title(f"Burgers solutions - Firedrake and Network, $N_x={nx}, \, D={d}$")
    # ax.legend()
    #
    # # Show the plot
    # plt.show()
