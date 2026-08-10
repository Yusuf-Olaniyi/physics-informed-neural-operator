"""Global stiffness matrix assembly."""
import numpy as np
from .elements import BeamElement


class FEMAssembler:
    def __init__(self, beam):
        self.beam = beam

    def assemble_stiffness(self) -> np.ndarray:
        beam = self.beam
        K = np.zeros((beam.num_dofs, beam.num_dofs))
        ke = BeamElement.stiffness(beam.E, beam.I, beam.Le)

        for e in range(beam.num_elements):
            node1, node2 = e, e + 1
            dofs = [2 * node1, 2 * node1 + 1, 2 * node2, 2 * node2 + 1]
            for i in range(4):
                for j in range(4):
                    K[dofs[i], dofs[j]] += ke[i, j]

        return K

    def assemble_mass(self) -> np.ndarray:
        """Global consistent mass matrix, for dynamic (time-domain) analysis.
        Same assembly pattern as `assemble_stiffness`, using the element
        mass matrix instead of the element stiffness matrix."""
        beam = self.beam
        M = np.zeros((beam.num_dofs, beam.num_dofs))
        me = BeamElement.mass_matrix(beam.rho, beam.A, beam.Le)

        for e in range(beam.num_elements):
            node1, node2 = e, e + 1
            dofs = [2 * node1, 2 * node1 + 1, 2 * node2, 2 * node2 + 1]
            for i in range(4):
                for j in range(4):
                    M[dofs[i], dofs[j]] += me[i, j]

        return M
