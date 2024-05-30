import firedrake as fd
import torch


if __name__ == "__main__":
    filename = "data/burgers__samples_100__nx_100"

    data = torch.load(f"{filename}.pt")

    with fd.CheckpointFile(f"{filename}__mesh.h5", "r") as file:
        mesh = file.load_mesh()

    function_space = fd.FunctionSpace(mesh, "CG", degree=1)

    initial_functions = []
    solutions = []
    for arr in data:
        initial_functions.append(fd.Function(function_space, val=arr[0]))
        solutions.append(fd.Function(function_space, val=arr[1]))

