"""Physics-based dataset generation for the dynamic forward operator
G_f: q(x,t) -> w(x,t).

Same design conventions as the static `generator.py`:
  - a single SupportType is fixed for the entire dataset (see
    `fem/boundary.py` docstring) -- BC is never an operator input.
  - x (and here, t) are appended as coordinate channels purely so the
    FNO has access to spatiotemporal position, not as physics metadata.

    X_f = [q(x,t), x, t]   (3 channels, on an (num_nodes, num_steps) grid)
    Y_f = w(x,t)
"""
from dataclasses import dataclass
import numpy as np

from beam_no.fem import Beam, DynamicFEMSolver, LoadGenerator, DynamicLoadGenerator, SupportType


@dataclass
class DynamicForwardSample:
    q_xt: np.ndarray          # (num_steps, num_nodes)
    w_xt: np.ndarray          # (num_steps, num_nodes)
    spatial_type: str
    temporal_type: str


class DynamicDatasetGenerator:
    def __init__(self, beam: Beam, support: SupportType, dt: float, num_steps: int,
                 num_samples: int = 2000, amplitude_range=(5.0, 20.0),
                 rayleigh_alpha: float = 0.5, rayleigh_beta: float = 1e-4,
                 seed: int = None):
        self.beam = beam
        self.support = support
        self.dt = dt
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.duration = dt * (num_steps - 1)

        self.rng = np.random.default_rng(seed)
        self.spatial_generator = LoadGenerator(beam, amplitude_range=amplitude_range, rng=self.rng)
        self.load_generator = DynamicLoadGenerator(
            self.spatial_generator, self.duration, num_steps, rng=self.rng
        )
        self.solver = DynamicFEMSolver(
            beam, support, dt=dt,
            rayleigh_alpha=rayleigh_alpha, rayleigh_beta=rayleigh_beta,
        )
        self.t = np.linspace(0.0, self.duration, num_steps)

    def generate_sample(self) -> DynamicForwardSample:
        q_xt, spatial_type, temporal_type = self.load_generator.generate()
        w_xt = self.solver.solve(q_xt)
        return DynamicForwardSample(q_xt=q_xt, w_xt=w_xt,
                                     spatial_type=spatial_type, temporal_type=temporal_type)

    def generate_dataset(self, verbose: bool = True):
        """Returns
        -------
        X : (num_samples, num_nodes, num_steps, 3) array -- channels [q, x, t]
        Y : (num_samples, num_nodes, num_steps, 1) array -- channel [w]
        x : (num_nodes,) array
        t : (num_steps,) array
        meta : list[dict]  -- {"spatial_type", "temporal_type"} per sample
        """
        X, Y, meta = [], [], []

        x_grid, t_grid = np.meshgrid(self.beam.x, self.t, indexing="ij")  # (num_nodes, num_steps)

        for i in range(self.num_samples):
            sample = self.generate_sample()

            # solver returns (num_steps, num_nodes) -> transpose to (num_nodes, num_steps)
            q_grid = sample.q_xt.T
            w_grid = sample.w_xt.T

            X.append(np.stack([q_grid, x_grid, t_grid], axis=-1))
            Y.append(w_grid[..., np.newaxis])
            meta.append({"spatial_type": sample.spatial_type, "temporal_type": sample.temporal_type})

            if verbose and (i + 1) % 200 == 0:
                print(f"{i + 1}/{self.num_samples} dynamic samples generated")

        return np.array(X), np.array(Y), self.beam.x, self.t, meta
