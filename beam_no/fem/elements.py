"""Element-level operators for the Euler-Bernoulli beam finite element."""
import numpy as np


class BeamElement:
    """Local (element) stiffness matrix for a 2-node Euler-Bernoulli beam element.

    DOF order per element: [w1, theta1, w2, theta2]
    """

    @staticmethod
    def stiffness(E: float, I: float, L: float) -> np.ndarray:
        factor = E * I / (L ** 3)
        k = factor * np.array([
            [12,     6 * L,   -12,     6 * L],
            [6 * L,  4 * L**2, -6 * L,  2 * L**2],
            [-12,   -6 * L,    12,    -6 * L],
            [6 * L,  2 * L**2, -6 * L,  4 * L**2],
        ])
        return k

    @staticmethod
    def mass_matrix(rho: float, A: float, L: float) -> np.ndarray:
        """Consistent (not lumped) element mass matrix for a 2-node
        Euler-Bernoulli beam element, same DOF order [w1, theta1, w2, theta2]
        as `stiffness`. Standard closed-form result from the same cubic
        (Hermite) shape functions used to derive the stiffness matrix.
        """
        factor = rho * A * L / 420.0
        m = factor * np.array([
            [156,      22 * L,     54,      -13 * L],
            [22 * L,   4 * L**2,   13 * L,  -3 * L**2],
            [54,       13 * L,     156,     -22 * L],
            [-13 * L, -3 * L**2,  -22 * L,   4 * L**2],
        ])
        return m
