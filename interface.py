"""
generator/interface.py - Standardized Solver Generator Interface
VeriPDE Specification (October 2026)

Enforces strict isolation between Generator Zone and Blind Verification Vault.
The generator ONLY knows this clean interface contract and receives physical specs.
It has ZERO access to the vault's MMS ansatz, step-halving probes, or hidden test suites.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union
import numpy as np


class PDESolverCallable(Protocol):
    """
    Standardized callable protocol for all autonomous PDE solvers in Fourier/VeriPDE.
    """

    def __call__(
        self,
        N: int,
        dt: Optional[float] = None,
        t_end: float = 0.2,
        source_fn: Optional[Callable[..., Any]] = None,
        exact_bc_fn: Optional[Callable[..., Any]] = None,
        bc_dict: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        ...
