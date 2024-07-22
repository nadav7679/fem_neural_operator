"""
This file generates initial conditions and solutions to Burgers Equation using firedrake.
"""
import torch
from firedrake import *
import argparse

from BurgersSolver import BurgersSolver


def gaussian_field(cg_space, dg_space, cell_area, rg, d=25., s=625.):

    w = rg.normal(dg_space)
    w.assign(w/(cell_area**0.5))
    w -= assemble(w*dx)
    
    v_func = TestFunction(cg_space)
    u_func = TrialFunction(cg_space)

    a = d*u_func * v_func * dx + dot(grad(u_func), grad(v_func)) * dx
    F = s*w * v_func * dx

    uh = Function(cg_space)
    solve(a == F, uh)

    v_func = TestFunction(cg_space)
    u_func = TrialFunction(cg_space)
    
    a = d*u_func * v_func * dx + dot(grad(u_func), grad(v_func)) * dx
    F = v_func * uh * dx
    
    uh_2 = Function(cg_space)
    solve(a == F, uh_2)

    return uh_2

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate Burgers data")
    
    parser.add_argument("project_dir", type=str, help="Directory of the project")
    parser.add_argument("nu", type=float, help="The parameter nu in the equation")
    parser.add_argument("N", type=int, help="Number of cells in the grid")
    parser.add_argument("samples", type=int, help="Number of samples in current batch")
    parser.add_argument("seed", type=int, help="seed to generate data from")
    parser.add_argument("batch", type=int, help="Batch number")
    
    args = parser.parse_args()
    
    # Parameters
    degree, t_end = 1, 1
    length = 1
    d, s = 25, 625

    # with CheckpointFile(f"/tmp/np923/N{nx}.h5", "w") as f:
    #     f.save_mesh(mesh)
    
    with CheckpointFile(f"{args.project_dir}/playground/data/meshes/N{args.N}.h5", "r") as f:
        mesh = f.load_mesh()

    cg_space = FunctionSpace(mesh, "CG", degree=1)
    dg_space = FunctionSpace(mesh, "DG", degree=0)

    burgers = BurgersSolver(mesh, args.nu, degree, t_end)

    pcg = PCG64(seed=args.seed)
    rg = Generator(pcg)

    data = torch.zeros((args.samples, 2, cg_space.dof_count), dtype=torch.float64)
    for i in ProgressBar("Burgers Generator").iter(range(args.samples)):
        gf = gaussian_field(cg_space, dg_space, length/args.N, rg, d=d, s=s)
        sol = burgers.solve(gf)

        data[i, 0] = torch.tensor(gf.dat.data)
        data[i, 1] = torch.tensor(sol.dat.data)

    torch.save(data, f"{args.project_dir}/playground/data/samples/N{args.N}_nu{args.nu}_samples{args.samples}_batch{args.batch}.pt")



