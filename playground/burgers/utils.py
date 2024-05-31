from firedrake import pi, cos, sin, dx

import firedrake as fd
import torch

if __name__ == "__main__":
    filename = "../data/burgers__samples_100__nx_100"

    data = torch.load(f"{filename}.pt")

    with fd.CheckpointFile(f"{filename}__mesh.h5", "r") as file:
        mesh = file.load_mesh()

    function_space = fd.FunctionSpace(mesh, "CG", degree=1)

    initial_functions = []
    solutions = []
    for arr in data:
        initial_functions.append(fd.Function(function_space, val=arr[0]))
        solutions.append(fd.Function(function_space, val=arr[1]))

    x = fd.SpatialCoordinate(mesh)[0]
    v = fd.TestFunction(function_space)

    # Get coefficient matrix cij=<\psi_i,\phi_j> for j <= N (FE space dimension) and i < max_modes
    max_modes = 8
    dim = len(mesh.cell_sizes.dat.data)
    c = torch.zeros((2 * max_modes + 1, dim), dtype=torch.float64)  # Zero mode (const) and cos, sin funcs for each mode

    c[0] += fd.assemble(v * dx).dat.data
    for i in range(1, max_modes + 1):
        c[2 * i - 1] += fd.assemble(sin(i * 2 * pi * x) * v * dx).dat.data
        c[2 * i] += fd.assemble(cos(i * 2 * pi * x) * v * dx).dat.data

    # Integral test
    for i in range(2 * max_modes - 1):
        if not i:
            assert abs(torch.sum(c[i]) - 1) < 10E-12

        else:
            assert abs(torch.sum(c[i])) < 10E-12

    torch.save(c, f"{filename}__coefficients.pt")
