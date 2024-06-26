import firedrake as fd
import torch


class ProjectionCoefficient:
    """
    Class to calculate and manage the projection coefficient matrix C_{mn} = <psi_m, phi_n> for the FE nodal basis
    {phi}^N and projection functions {psi}^M.

    Attributes:
        mesh (fd.Mesh): The computational mesh.
        projection_type (str): The type of projection (currently only 'fourier' is supported).
        M (int): Number of projection functions (number of modes in the case of Fourier, then M=2*modes+1).
        N (int): Number of mesh points.
        device (str): Device for tensor computations ('cpu' or 'cuda').
        filename (str): Path to save the coefficient matrix.
        coeff (torch.Tensor): Coefficient matrix.
    """

    def __init__(
            self,
            mesh: fd.Mesh,
            projection_type: str,
            M: int,
            device='cpu'
    ):
        """
        Initialize the ProjectionCoefficient class.

        Args:
            mesh (fd.Mesh): The computational mesh.
            projection_type (str): The type of projection (currently only 'fourier' is supported).
            M (int): Number of Fourier modes.
            device (str): Device for tensor computations ('cpu' or 'cuda').
        """
        self.mesh = mesh
        self.projection_type = projection_type
        self.M = M
        self.device = device

        self.N = int(len(mesh.cell_sizes.dat.data))
        self.filename = f"data/projection_coefficients/{projection_type}/N_{self.N}__M_{self.M}.pt"
        self.coeff = torch.zeros((M, self.N))

    def _calculate_fourier(self):
        """
        Calculate the coefficient matrix C_{mn} = <psi_m, phi_n> for the FE basis {phi}^N and
        trigonometric functions {psi}^(2*M+1), up to mode M. The function also saves
        the coefficient matrix.

        Returns:
            None
        """
        function_space = fd.FunctionSpace(self.mesh, "CG", degree=1)
        x = fd.SpatialCoordinate(self.mesh)[0]
        v = fd.TestFunction(function_space)

        self.coeff = torch.zeros((2 * self.M + 1, self.N),  # Zero mode and cos, sin for each mode
                                 dtype=torch.float64,
                                 device=self.device)
        for i in range(self.M + 1):
            if i == 0:
                self.coeff[i] += fd.assemble(v * fd.dx).dat.data
                continue

            self.coeff[2 * i - 1] += fd.assemble(fd.sin(i * 2 * fd.pi * x) * v * fd.dx).dat.data
            self.coeff[2 * i] += fd.assemble(fd.cos(i * 2 * fd.pi * x) * v * fd.dx).dat.data

    def _test_fourier(self):
        """
        Test the Fourier coefficients to ensure correctness. Testing that the integral vanishes for modes above 0,
        and integral is unity for mode 0.

        Returns:
            None
        """
        for i in range(2 * self.M - 1):
            if i == 0:
                assert abs(torch.sum(self.coeff[i]) - 1) < 10E-12

            else:
                assert abs(torch.sum(self.coeff[i])) < 10E-12

    def calculate(self):
        """
        Calculate the projection coefficients based on the specified projection_type. Note that the resulting matrix's
        size for fourier is 2*M+1.

        Returns:
            None

        Raises:
            ValueError: If the projection type is not supported.
        """
        if self.projection_type == "fourier":
            self._calculate_fourier()
            self._test_fourier()
            torch.save(self.coeff, self.filename)

        else:
            raise ValueError("Only 'fourier' projection_type is supported")

    @staticmethod
    def load(filename: str, mesh: fd.Mesh, device='cpu'):
        """
        Load the projection coefficient matrix from a file.

        Args:
            filename (str): Path to the file containing the coefficient matrix.
            mesh (fd.Mesh): The computational mesh.
            device (str): Device for tensor computations ('cpu' or 'cuda').

        Returns:
            ProjectionCoefficient: An instance of ProjectionCoefficient with loaded coefficients.
        """
        projection_type = filename.split("/")[2]
        M = int(filename[filename.find("N") + 2: filename.find("__M")])
        N = int(len(mesh.cell_sizes.dat.data))
        proj = ProjectionCoefficient(mesh,
                                     projection_type,
                                     M,
                                     N)
        proj.coeff = torch.load(filename).to(device=device)

        return proj


if __name__ == "__main__":
    mesh = fd.PeriodicIntervalMesh(100, 1)
    proj = ProjectionCoefficient(mesh, 'fourier', 10)
    proj.calculate()

    proj2 = ProjectionCoefficient.load("data/projection_coefficients/fourier/N_100__M_10.pt", mesh)
    print(torch.all(proj2.coeff == proj.coeff))
