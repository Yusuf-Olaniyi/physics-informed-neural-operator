"""Boundary condition definitions for the Euler-Bernoulli beam.

NOTE (design decision):
A single support type is fixed for an entire
dataset / training run, so that the learned mapping G: q(x) -> w(x) is a
well-defined function (see `configs/default.yaml: support_type`).
If you want to study several boundary conditions, generate a separate
dataset and train a separate operator per support type, rather than mixing
them into one dataset with a hidden/encoded BC.
"""
from enum import Enum
import numpy as np


class SupportType(Enum):
    CANTILEVER = 1
    SIMPLY_SUPPORTED = 2
    FIXED_FIXED = 3
    FIXED_PINNED = 4

    @classmethod
    def from_str(cls, name: str) -> "SupportType":
        return cls[name.strip().upper()]


class BoundaryConditions:
    @staticmethod
    def constrained_dofs(beam, support: SupportType):
        """Return the list of globally constrained DOF indices.

        DOF layout: displacement = even index, rotation = odd index.
        """
        last_node = beam.num_nodes - 1

        if support == SupportType.CANTILEVER:
            return [0, 1]  # [w0, theta0]

        elif support == SupportType.SIMPLY_SUPPORTED:
            return [0, 2 * last_node]  # [left vertical, right vertical]

        elif support == SupportType.FIXED_FIXED:
            return [0, 1, 2 * last_node, 2 * last_node + 1]

        elif support == SupportType.FIXED_PINNED:
            return [0, 1, 2 * last_node]

        else:
            raise ValueError(f"Unknown support type: {support}")
