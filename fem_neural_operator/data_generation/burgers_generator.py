"""
This file generates initial conditions and solutions to Burgers Equation using firedrake.
"""
import torch
from firedrake import *
import argparse
import numpy as np
from firedrake.pyplot import plot
import matplotlib.pyplot as plt


class BurgersSolver:
    """
    A class that solves burgers equation for a given initial condition and mesh.
    """
    def __init__(self, mesh: PeriodicIntervalMesh, nu: float, degree: int, t_end: float):
        """

        Args:
            mesh: Periodic mesh
            nu: The constant \nu from Burgers
            degree: Degree of Lagrange function space
            t_end: Final time for simulation
        """
        self.mesh = mesh
        self.t_end = t_end
        self.dt = 1./(10*len(mesh.cell_sizes.dat.data))  # Equispaced so nx=const
        self.nu = Constant(nu)
        self.x = SpatialCoordinate(self.mesh)[0]
        self.space = FunctionSpace(self.mesh, "Lagrange", degree)

        self.u_n1 = Function(self.space, name="u^{n+1}")
        self.u_n = Function(self.space, name="u^{n}")
        self.v = TestFunction(self.space)

        f = (((self.u_n1 - self.u_n)) * self.v +
             self.dt * self.u_n1 * self.u_n1.dx(0) * self.v +
             self.dt * self.nu*self.u_n1.dx(0)*self.v.dx(0)) * dx

        problem = NonlinearVariationalProblem(f, self.u_n1)
        self.solver = NonlinearVariationalSolver(problem)

    def solve(self, u_init, running_plot=False):
        self.u_n.interpolate(u_init)

        self.u_n1.assign(self.u_n)

        for t in np.linspace(0, self.t_end, int(self.t_end/self.dt)):
            if running_plot:
                if t == 0:
                    fig, axes = plt.subplots()

                axes.set_title(f"t={t:.3f}")
                axes.grid()
                plot(self.u_n, axes=axes)
                plt.pause(0.0001)
                axes.clear()

            self.solver.solve()
            self.u_n.assign(self.u_n1)

        return self.u_n


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
    
    
    # -------- Test BurgersSolver ----------#
    # mesh = PeriodicIntervalMesh(100, 1)
    # params = {
    #     "mesh": mesh,
    #     "nu": 1e-2,
    #     "degree": 2,
    #     "t_end": 1
    # }
    
    # burgers = BurgersSolver(**params)
    # u_init = sin(2*pi*burgers.x)
    # u = burgers.solve(u_init, running_plot=True)
        
    # fig, axes = plt.subplots()
    # plt.grid()
    # plot(u, axes=axes)
    # plt.savefig("media/fig")




