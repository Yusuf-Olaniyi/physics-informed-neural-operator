"""Dynamic FEM solver: time-domain integration of

    rho*A * d^2w/dt^2 + C * dw/dt + K * w = F(t)

via the Newmark-beta method (average-acceleration variant, beta=0.25,
gamma=0.5, which is unconditionally stable -- convenient for generating
many dataset samples without a CFL-type restriction on dt).

Damping uses the standard Rayleigh model C = alpha*M + beta_damp*K, which
reuses the mass/stiffness matrices already assembled for the structure
rather than requiring a separate damping model.
"""
import numpy as np
from .assembler import FEMAssembler
from .loads import LoadConverter
from .boundary import BoundaryConditions, SupportType


class DynamicFEMSolver:
    """Solves the forced-vibration equation for a beam with a *fixed*
    support type, fixed time step `dt`, and fixed Rayleigh damping
    coefficients (all set at construction time, so the effective
    stiffness matrix can be factored once and reused for every sample).
    """

    def __init__(self, beam, support: SupportType, dt: float,
                 rayleigh_alpha: float = 0.5, rayleigh_beta: float = 1e-4,
                 newmark_beta: float = 0.25, newmark_gamma: float = 0.5):
        self.beam = beam
        self.support = support
        self.dt = dt
        self.rayleigh_alpha = rayleigh_alpha
        self.rayleigh_beta = rayleigh_beta
        self.newmark_beta = newmark_beta
        self.newmark_gamma = newmark_gamma

        assembler = FEMAssembler(beam)
        self.K = assembler.assemble_stiffness()
        self.M = assembler.assemble_mass()
        self.C = rayleigh_alpha * self.M + rayleigh_beta * self.K

        self.load_converter = LoadConverter(beam)
        self.fixed_dofs = BoundaryConditions.constrained_dofs(beam, support)
        self.free_dofs = np.delete(np.arange(beam.num_dofs), self.fixed_dofs)

        self._Kff = self.K[np.ix_(self.free_dofs, self.free_dofs)]
        self._Mff = self.M[np.ix_(self.free_dofs, self.free_dofs)]
        self._Cff = self.C[np.ix_(self.free_dofs, self.free_dofs)]

        b = newmark_beta
        g = newmark_gamma
        self._a0 = 1.0 / (b * dt ** 2)
        self._a1 = g / (b * dt)
        self._a2 = 1.0 / (b * dt)
        self._a3 = 1.0 / (2 * b) - 1.0
        self._a4 = g / b - 1.0
        self._a5 = dt * (g / (2 * b) - 1.0)

        # Effective stiffness is constant across time steps (dt, support,
        # and damping are all fixed for this solver instance) -- factor
        # once via a Cholesky-friendly solve using np.linalg.solve per
        # step would still re-factor every call, so precompute explicitly.
        self._Keff = self._Kff + self._a1 * self._Cff + self._a0 * self._Mff

    def solve(self, q_xt: np.ndarray) -> np.ndarray:
        """Time-march the forced response for a given load history.

        Parameters
        ----------
        q_xt : (num_steps, num_nodes) array
            Distributed load q(x, t_n) at every node, for every time step.

        Returns
        -------
        w_history : (num_steps, num_nodes) array
            Transverse displacement w(x, t_n) at every node/time step.
        """
        beam = self.beam
        num_steps = q_xt.shape[0]

        U = np.zeros(beam.num_dofs)
        V = np.zeros(beam.num_dofs)

        F0 = self.load_converter.equivalent_nodal_load(q_xt[0])
        F0f = F0[self.free_dofs]
        # Initial acceleration from the equation of motion at t=0 (U=V=0).
        Af = np.linalg.solve(self._Mff, F0f)
        A = np.zeros(beam.num_dofs)
        A[self.free_dofs] = Af

        w_history = np.zeros((num_steps, beam.num_nodes))
        w_history[0] = U[0::2]

        Uf, Vf, Af = U[self.free_dofs], V[self.free_dofs], A[self.free_dofs]

        for n in range(1, num_steps):
            F = self.load_converter.equivalent_nodal_load(q_xt[n])
            Ff = F[self.free_dofs]

            F_eff = (
                Ff
                + self._Mff @ (self._a0 * Uf + self._a2 * Vf + self._a3 * Af)
                + self._Cff @ (self._a1 * Uf + self._a4 * Vf + self._a5 * Af)
            )

            Uf_new = np.linalg.solve(self._Keff, F_eff)
            Af_new = self._a0 * (Uf_new - Uf) - self._a2 * Vf - self._a3 * Af
            Vf_new = Vf + self.dt * ((1 - self.newmark_gamma) * Af + self.newmark_gamma * Af_new)

            Uf, Vf, Af = Uf_new, Vf_new, Af_new

            U = np.zeros(beam.num_dofs)
            U[self.free_dofs] = Uf
            w_history[n] = U[0::2]

        return w_history

    @staticmethod
    def extract_deflection(U: np.ndarray) -> np.ndarray:
        return U[0::2]
