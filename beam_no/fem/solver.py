"""Static FEM solver: Kw = F, for a fixed boundary condition."""
import numpy as np
from .assembler import FEMAssembler
from .loads import LoadConverter
from .boundary import BoundaryConditions, SupportType


class FEMSolver:
    """Solves the static equilibrium equation for a beam with a *fixed*
    support type (set at construction time), for arbitrary q(x)."""

    def __init__(self, beam, support: SupportType):
        self.beam = beam
        self.support = support
        self.K = FEMAssembler(beam).assemble_stiffness()
        self.load_converter = LoadConverter(beam)
        self.fixed_dofs = BoundaryConditions.constrained_dofs(beam, support)
        self.free_dofs = np.delete(np.arange(beam.num_dofs), self.fixed_dofs)
        # Pre-factor the reduced stiffness matrix once, since the support never changes (and therefore the free/fixed partition) .
        self._Kff = self.K[np.ix_(self.free_dofs, self.free_dofs)]

    def solve(self, q: np.ndarray) -> np.ndarray:
        """Solve K u = F for the given distributed load q(x)."""
        beam = self.beam
        F = self.load_converter.equivalent_nodal_load(q)
        Ff = F[self.free_dofs]

        Uf = np.linalg.solve(self._Kff, Ff)

        U = np.zeros(beam.num_dofs)
        U[self.free_dofs] = Uf
        return U

    @staticmethod
    def extract_deflection(U: np.ndarray) -> np.ndarray:
        """Extract vertical displacement w(x) from the full DOF vector
        (DOF pattern is [w, theta, w, theta, ...])."""
        return U[0::2]
