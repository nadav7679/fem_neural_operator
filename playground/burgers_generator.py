"""
This file generates initial conditions and solutions to Burgers Equation using firedrake.
"""

import torch
from firedrake import *

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


# Parameters
SEED = 12345
samples = 1000
degree, t_end = 1, 2
nx, length = 100, 1
d, s, nu = 25, 625, 5e-2
cell_area = length/nx


mesh = PeriodicIntervalMesh(nx, length=length)
cg_space = FunctionSpace(mesh, "CG", degree=1)
dg_space = FunctionSpace(mesh, "DG", degree=0)

burgers = BurgersSolver(nx, length, nu, degree, t_end, mesh)

data = torch.zeros((samples, 2, cg_space.dof_count), dtype=torch.float64)
pcg = PCG64(seed=SEED)
rg = Generator(pcg)


for i in ProgressBar("Burgers Generator").iter(range(samples)):
    gf = gaussian_field(cg_space, dg_space, cell_area, rg, d=d, s=s)
    sol = burgers.solve(gf)

    data[i, 0] = torch.tensor(gf.dat.data)
    data[i, 1] = torch.tensor(sol.dat.data)

torch.save(data, "burgers_1000.pt")

