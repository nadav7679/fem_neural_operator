import torch
from firedrake import *


class BasisFunctions():
    def __init__(self, fs: FunctionSpace, basis):
        self.fs = fs

        # Creating firedrake basis functions on function space
        self.basis = []
        for func in basis:
            f = Function(fs)
            f.interpolate(func)

            self.basis.append(f)

        # Storing the integral of the basis with the function space nodal basis
        self.coeffs = np.zeros((len(self.basis)), fs.dof_count)
        for f in self.basis:
            v = TestFunction(self.fs)
            cofunc = assemble(v * f * dx)
            print(cofunc.dat.data)



if __name__ == "__main__":
    mesh = PeriodicIntervalMesh(100, length=1)
    x = SpatialCoordinate(mesh)

    cg_space = FunctionSpace(mesh, "CG", degree=1)
    monomials = [pow(x[0], i) for i in range(5)]

    phis = BasisFunctions(cg_space, monomials)
