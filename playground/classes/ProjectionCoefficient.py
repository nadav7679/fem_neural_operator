import firedrake as fd
import torch


class ProjectionCoefficient:
    """
    Class to calculate and manage the projection coefficient matrix C_{mn} = <psi_m, phi_n> for the FE nodal basis
    {phi}^N and projection functions {psi}^M.

    Attributes:
        mesh (fd.Mesh): The computational mesh.
        equation_type (str): The equation the Operator should learn. Either 'Burgers' or 'KS'
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
            equation_type: str,
            finite_element_family: str,
            projection_type: str,
            M: int,
            device='cpu'
    ):
        """
        Initialize the ProjectionCoefficient class.

        Args:
            mesh (fd.Mesh): The computational mesh.
            equation_type (str): The equation the Operator should learn. Either 'Burgers' or 'KS'
            finite_element_family (str): The FE family, either CG1, CG3 or HER
            projection_type (str): The type of projection (currently only 'fourier' is supported).
            M (int): Number of Fourier modes.
            device (str): Device for tensor computations ('cpu' or 'cuda').
        """
        self.mesh = mesh
        self.equation_type = equation_type
        self.finite_element_family = finite_element_family
        self.projection_type = projection_type
        self.M = M
        self.device = device

        self.N = int(len(mesh.cell_sizes.dat.data))


        self.filename = f"data/{equation_type}/projection_coefficients/{finite_element_family}/{projection_type}/N{self.N}_M{self.M}.pt"
        self.coeff = torch.zeros((M, self.N))

    def _calculate_fourier(self):
        """
        Calculate the coefficient matrix C_{mn} = <psi_m, phi_n> for the FE basis {phi}^N and
        trigonometric functions {psi}^(2*M+1), up to mode M. The function also saves
        the coefficient matrix.

        Returns:
            None
        """
        degree = 3 if self.finite_element_family in ["CG3", "HER"] else 1
        family = "CG" if self.finite_element_family in ["CG1", "CG3"] else self.finite_element_family

        function_space = fd.FunctionSpace(self.mesh, family, degree)
        x = fd.SpatialCoordinate(self.mesh)[0]
        v = fd.TestFunction(function_space)

        self.coeff = torch.zeros((2 * self.M + 1, function_space.dof_count), dtype=torch.float64)  # Zero mode and cos, sin for each mode
        for i in range(self.M + 1):
            if i == 0:
                self.coeff[i] += fd.assemble(v * fd.dx).dat.data
                continue

            self.coeff[2 * i - 1] += fd.assemble(2 * fd.sin(i * 2 * fd.pi * x) * v * fd.dx).dat.data
            self.coeff[2 * i] += fd.assemble(2 * fd.cos(i * 2 * fd.pi * x) * v * fd.dx).dat.data

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
                assert abs(torch.sum(self.coeff[i])) < 10E-10

    def calculate(self, save=True):
        """
        Calculate the projection coefficients based on the specified projection_type. Note that the resulting matrix's
        size for fourier is 2*M+1.

        Returns:
            None

        Raises:
            ValueError: If the projection type is not supported.
        """
        try:
            self.coeff = self.load(self.filename, self.mesh, self.device).coeff
            return

        except FileNotFoundError:
            pass

        if self.projection_type == "fourier":
            self._calculate_fourier()
            self._test_fourier()
            self.coeff = self.coeff.to(device=self.device, dtype=torch.float32)

        else:
            raise ValueError("Only 'fourier' projection_type is supported")

        if save:
            print(self.filename)
            torch.save(self.coeff, self.filename)

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
        finite_element_family, projection_type = filename.split("/")[3:5]
        equation_type = filename.split("/")[1]

        degree = 3 if finite_element_family in ["CG3", "HER"] else 1
        M = int(filename[filename.find("M") + 1:filename.find(".pt")])
        N = int(len(mesh.cell_sizes.dat.data))

        proj = ProjectionCoefficient(mesh, equation_type, finite_element_family, projection_type, M, N)
        proj.coeff = torch.load(filename).to(device=device)

        return proj


if __name__ == "__main__":
    mesh = fd.PeriodicIntervalMesh(100, 1)
    proj1 = ProjectionCoefficient(mesh, "burgers", "CG1", 'fourier', 12)
    proj1.calculate()

    proj2 = ProjectionCoefficient.load("data/burgers/projection_coefficients/CG1/fourier/N100_M12.pt", mesh)
    print(proj1.coeff.shape)
    print(torch.all(proj2.coeff == proj1.coeff))
