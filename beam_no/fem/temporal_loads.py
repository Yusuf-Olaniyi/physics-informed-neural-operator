"""Temporal load-profile generation for dynamic (forced-vibration) analysis.

Mirrors `fem/loads.py`'s design: a small set of structured, parametrically
randomized families (not a Gaussian Process / GRF sampler over time), so
the dynamic study follows the same "structured randomization, GRF flagged
as future work" convention established for the static case.

The full spatiotemporal load is modeled as separable:

    q(x, t) = q_spatial(x) * f_temporal(t)

which keeps dataset generation simple (reuse the existing spatial
`LoadGenerator` unmodified) while still producing a rich variety of
forced-vibration scenarios via the temporal profile.
"""
import numpy as np


class TemporalLoadGenerator:
    """Generates randomized dimensionless temporal profiles f(t) in [-1, 1]
    (approximately), sampled at `t = np.linspace(0, duration, num_steps)`.
    """

    def __init__(self, duration: float, num_steps: int, rng: np.random.Generator = None):
        self.duration = duration
        self.num_steps = num_steps
        self.t = np.linspace(0.0, duration, num_steps)
        self.rng = rng if rng is not None else np.random.default_rng()

    def step_ramp(self):
        """Load ramps up linearly over a random fraction of the duration,
        then holds constant (models a suddenly-applied, then sustained load)."""
        ramp_frac = self.rng.uniform(0.02, 0.2)
        t_ramp = ramp_frac * self.duration
        f = np.clip(self.t / max(t_ramp, 1e-8), 0.0, 1.0)
        return f

    def sinusoidal(self):
        """Single-frequency sinusoidal forcing at a randomized frequency/phase."""
        freq = self.rng.uniform(0.5, 5.0)  # Hz
        phase = self.rng.uniform(0, 2 * np.pi)
        return np.sin(2 * np.pi * freq * self.t + phase)

    def multi_frequency(self, num_modes: int = 4):
        """Randomized sum of a few sinusoids -- a broadband forcing signal,
        analogous to `LoadGenerator.random_fourier` in the spatial domain."""
        f = np.zeros_like(self.t)
        for _ in range(num_modes):
            freq = self.rng.uniform(0.2, 8.0)
            amp = self.rng.uniform(0.2, 1.0)
            phase = self.rng.uniform(0, 2 * np.pi)
            f += amp * np.sin(2 * np.pi * freq * self.t + phase)
        return f / np.max(np.abs(f) + 1e-12)

    def gaussian_pulse(self):
        """A short-duration pulse centered at a random point in time."""
        center = self.rng.uniform(0.15, 0.85) * self.duration
        width = self.rng.uniform(0.02, 0.08) * self.duration
        return np.exp(-((self.t - center) ** 2) / (2 * width ** 2))

    def generate(self, profile_type: str = None):
        families = {
            "step_ramp": self.step_ramp,
            "sinusoidal": self.sinusoidal,
            "multi_frequency": self.multi_frequency,
            "gaussian_pulse": self.gaussian_pulse,
        }
        if profile_type is None:
            profile_type = self.rng.choice(list(families.keys()))
        return families[profile_type](), profile_type


class DynamicLoadGenerator:
    """Combines a spatial load shape and a temporal profile into a
    separable spatiotemporal load q(x, t) = q_spatial(x) * f_temporal(t).
    """

    def __init__(self, spatial_generator, duration: float, num_steps: int,
                 rng: np.random.Generator = None):
        self.spatial_generator = spatial_generator
        self.temporal_generator = TemporalLoadGenerator(duration, num_steps, rng=rng)

    def generate(self):
        """Returns
        -------
        q_xt : (num_steps, num_nodes) array
        spatial_type : str
        temporal_type : str
        """
        q_spatial, spatial_type = self.spatial_generator.generate()
        f_temporal, temporal_type = self.temporal_generator.generate()

        q_xt = np.outer(f_temporal, q_spatial)  # (num_steps, num_nodes)
        return q_xt, spatial_type, temporal_type
