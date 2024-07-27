from firedrake import *
import numpy as np
import argparse
import torch


def KS_sample(mesh, nu, L, T, dt, samples, transient=2, metrics=False):
    """_summary_

    Args:
        mesh (_type_): Periodic interval mesh of length L.
        nu (_type_): nu parameter in KS equation
        L (_type_): Length of system
        T (_type_): The period of time we want to solve for, i.e., the period of time the model will learn to imitate.
        dt (_type_): Time intervals to solve with
        samples (_type_): Amount of samples (input-output pairs) to be outputted. 
        transient (int, optional): Time (solver time) until the system forgets the initial condition. Defaults to 2.
        metrics (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # Define the problem
    V = FunctionSpace(mesh, "HER", 3)
    v = TestFunction(V)
    
    un = Function(V)
    unp1 = Function(V)
    uh = (un + unp1)/2

    dT = Constant(dt)
    nu = Constant(nu)

    eqn = (
        v*(unp1 - un)
        - dT*v.dx(0)*uh.dx(0)
        + dT*nu*v.dx(0).dx(0)*uh.dx(0).dx(0)
        - dT*0.5*v.dx(0)*uh*uh
        )*dx

    params = {
        "snes_atol": 1.0e-50,
        "snes_rtol": 1.0e-6,
        "snes_stol": 1.0e-50,
        "ksp_type":"preonly",
        "pc_type":"lu"
    }

    # Make the solver
    KS_solver = NonlinearVariationalSolver(NonlinearVariationalProblem(eqn, unp1), solver_parameters=params)
    
    # Define interpolated space
    VOut = FunctionSpace(mesh, "CG", 3)
    uout = Function(VOut)
    uout.interpolate(un)    

    # Initial condition
    x, = SpatialCoordinate(mesh)
    un.project(-sin(pi*2*x/L))
    
    
    # Sampling in `transient` intervals
    data = torch.zeros((samples, 2, V.dof_count), dtype=torch.float64)
    for i in range(samples):
        un = KS_iterate(KS_solver, unp1, un, transient, dt, metrics) # Iterate system to forget transient
        data[i, 0] = torch.tensor(un.dat.data) # Save input
        
        un = KS_iterate(KS_solver, unp1, un, T, dt) # Solve for T seconds
        data[i, 1] = torch.tensor(un.dat.data) # Save output
    
    return data


def KS_iterate(KS_solver, unp1, un, tmax, dt, metrics=False):
    t_range = np.arange(0, tmax, dt)

    # norms = []
    # integrals = []
    for _ in t_range:
        KS_solver.solve()
        un.assign(unp1)

        # if metrics:
        #     integrals.append(norm(un))
        #     norms.append(assemble(un * dx))
            
    return un
        
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate Burgers data")
    
    parser.add_argument("project_dir", type=str, help="Directory of the project")
    parser.add_argument("N", type=int, help="Number of cells in the grid")
    parser.add_argument("T", type=float, help="The period of time we want to solve for, i.e., the period of time the model will learn to imitate.")
    parser.add_argument("samples", type=int, help="Number of samples in current batch")
    parser.add_argument("batch", type=int, help="Current batch number")

    parser.add_argument("--nu", type=float, nargs="?", help="The parameter nu in the equation", default=0.029)
    parser.add_argument("--dt", type=float, nargs="?", help="Time interval in the solver", default=0.01)
    
    args = parser.parse_args()

    L = 10.
    
    # with CheckpointFile(f"{args.project_dir}/playground/data/KS/meshes/N{args.N}.h5", "r") as f:
    #     mesh = f.load_mesh()
    
    with CheckpointFile(f"{args.project_dir}/playground/data/KS/meshes/N{args.N}.h5", "r") as f:
        mesh = f.load_mesh()

    data = KS_sample(mesh, args.nu, L, args.T, args.dt, args.samples)
    torch.save(data, f"{args.project_dir}/playground/data/KS/samples/N{args.N}_nu{args.nu}_T{args.T}_samples{args.samples}_batch{args.batch}.pt")
