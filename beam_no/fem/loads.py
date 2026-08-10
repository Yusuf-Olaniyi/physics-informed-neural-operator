"""Distributed load generation and conversion to equivalent nodal forces.

Load families are deterministic parametric functions (uniform, triangular,
reverse-triangular, sinusoidal, gaussian bump, random truncated Fourier
series) with randomized amplitude / shape parameters. 
"""
import numpy as np


class LoadGenerator:
    """Generates randomized transverse load distributions q(x) on a beam."""

    def __init__(self, beam, amplitude_range=(5.0, 20.0), rng: np.random.Generator = None):
        self.beam = beam
        self.amplitude_range = amplitude_range
        self.rng = rng if rng is not None else np.random.default_rng()

    def _xi(self):
        return self.beam.x / self.beam.length

    def uniform(self):
        magnitude = self.rng.uniform(*self.amplitude_range)
        return np.ones(self.beam.num_nodes) * magnitude

    def triangular(self):
        magnitude = self.rng.uniform(*self.amplitude_range)
        xi = self._xi()
        return magnitude * xi

    def reverse_triangular(self):
        magnitude = self.rng.uniform(*self.amplitude_range)
        xi = self._xi()
        return magnitude * (1 - xi)

    def sinusoidal(self):
        magnitude = self.rng.uniform(*self.amplitude_range)
        xi = self._xi()
        return magnitude * np.sin(np.pi * xi)

    def gaussian(self):
        magnitude = self.rng.uniform(*self.amplitude_range)
        center = self.rng.uniform(0.2, 0.8)
        sigma = self.rng.uniform(0.05, 0.15)
        xi = self._xi()
        return magnitude * np.exp(-(xi - center) ** 2 / (2 * sigma ** 2))

    def random_fourier(self, modes: int = 5):
        xi = self._xi()
        q = np.zeros_like(xi)
        for k in range(1, modes + 1):
            a = self.rng.uniform(-10, 10)
            b = self.rng.uniform(-10, 10)
            q += a * np.sin(k * np.pi * xi) + b * np.cos(k * np.pi * xi)
        return q

    def generate(self, load_type: str = None):
        """Randomly choose a load type and generate a sample, or generate
        a specific `load_type` if given (one of: uniform, triangular,
        reverse_triangular, sinusoidal, gaussian, random_fourier)."""
        families = {
            "uniform": self.uniform,
            "triangular": self.triangular,
            "reverse_triangular": self.reverse_triangular,
            "sinusoidal": self.sinusoidal,
            "gaussian": self.gaussian,
            "random_fourier": self.random_fourier,
        }
        if load_type is None:
            load_type = self.rng.choice(list(families.keys()))
        return families[load_type](), load_type


class LoadConverter:
    """Converts a distributed load q(x) into equivalent FEM nodal forces."""

    def __init__(self, beam):
        self.beam = beam

    def equivalent_nodal_load(self, q: np.ndarray) -> np.ndarray:
        beam = self.beam
        F = np.zeros(beam.num_dofs)

        for e in range(beam.num_elements):
            n1, n2 = e, e + 1
            q_avg = (q[n1] + q[n2]) / 2
            L = beam.Le

            fe = np.array([
                q_avg * L / 2,
                q_avg * L ** 2 / 12,
                q_avg * L / 2,
                -q_avg * L ** 2 / 12,
            ])

            dofs = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
            for i in range(4):
                F[dofs[i]] += fe[i]

        return F
