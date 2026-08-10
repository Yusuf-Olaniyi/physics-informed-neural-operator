"""Euler-Bernoulli beam geometry and material definition."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Beam:
    """Euler-Bernoulli beam definition.

    Parameters
    ----------
    length : float
        Beam length [m].
    E : float
        Young's modulus [Pa].
    I : float
        Second moment of area [m^4].
    num_elements : int
        Number of finite elements used to discretize the beam.
    rho : float
        Mass density [kg/m^3]. Only used by the dynamic (mass-matrix /
        time-domain) solver; ignored by the static solver.
    A : float
        Cross-sectional area [m^2]. Only used by the dynamic solver.
    """

    length: float = 10.0
    E: float = 200e9
    I: float = 8e-6
    num_elements: int = 64
    rho: float = 7850.0
    A: float = 0.01

    num_nodes: int = field(init=False)
    num_dofs: int = field(init=False)
    Le: float = field(init=False)
    x: np.ndarray = field(init=False)

    def __post_init__(self):
        self.num_nodes = self.num_elements + 1
        self.num_dofs = self.num_nodes * 2  # displacement + rotation per node
        self.Le = self.length / self.num_elements
        self.x = np.linspace(0.0, self.length, self.num_nodes)

    @property
    def mass_per_length(self) -> float:
        """rho * A, [kg/m]."""
        return self.rho * self.A
