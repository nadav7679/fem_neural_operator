import numpy as np
from firedrake import *
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


if __name__ == "__main__":
    from firedrake.pyplot import plot
    import matplotlib.pyplot as plt
    
    params = {
        "n": 100,
        "length": 2,
        "nu": 1e-2,
        "degree": 2,
        "t_end": 1
    }
    
    burgers = BurgersSolver(**params)
    u_init = sin(2*pi*burgers.x)
    u = burgers.solve(u_init, running_plot=True)
        
    # fig, axes = plt.subplots()
    # plt.grid()
    # plot(u, axes=axes)
    # plt.savefig("media/fig")