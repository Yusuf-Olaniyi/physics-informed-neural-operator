"""Physics-based dataset generation for the forward operator G_f: q(x) -> w(x).

The single SupportType is fixed for the entire dataset (passed into the `FEMSolver` used here), so that the generated
mapping is a genuine function of q(x) alone:

    X_f = q(x)          (plus the coordinate x, appended as a channel
                          purely so the FNO has access to spatial position)
    Y_f = w(x)

If multiple boundary conditions are of interest, instantiate this
generator once per `SupportType` and keep the resulting datasets (and the
operators trained on them) separate.
"""
from dataclasses import dataclass
import numpy as np

from beam_no.fem import Beam, FEMSolver, LoadGenerator, SupportType


@dataclass
class ForwardSample:
    q: np.ndarray
    w: np.ndarray
    load_type: str


class DatasetGenerator:
    def __init__(self, beam: Beam, support: SupportType, num_samples: int = 10000,
                 amplitude_range=(5.0, 20.0), seed: int = None):
        self.beam = beam
        self.support = support
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)
        self.load_generator = LoadGenerator(beam, amplitude_range=amplitude_range, rng=self.rng)
        self.solver = FEMSolver(beam, support)

    def generate_sample(self) -> ForwardSample:
        q, load_type = self.load_generator.generate()
        U = self.solver.solve(q)
        w = self.solver.extract_deflection(U)
        return ForwardSample(q=q, w=w, load_type=load_type)

    def generate_dataset(self, verbose: bool = True):
        """Generate the complete dataset.

        Returns
        -------
        X : (num_samples, num_nodes, 2) array   -- channels: [q, x]
        Y : (num_samples, num_nodes, 1) array   -- channel: [w]
        x : (num_nodes,) array                  -- shared coordinate grid
        load_types : list[str]                  -- load family per sample (for analysis/plots)
        """
        X, Y, load_types = [], [], []

        for i in range(self.num_samples):
            sample = self.generate_sample()

            q_channel = sample.q.reshape(-1, 1)
            x_channel = self.beam.x.reshape(-1, 1)

            X.append(np.concatenate([q_channel, x_channel], axis=1))
            Y.append(sample.w.reshape(-1, 1))
            load_types.append(sample.load_type)

            if verbose and (i + 1) % 1000 == 0:
                print(f"{i + 1}/{self.num_samples} samples generated")

        return np.array(X), np.array(Y), self.beam.x, load_types
