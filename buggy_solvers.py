"""
generator/buggy_solvers.py - Adversarial & Buggy Solver Implementations
VeriPDE Specification (October 2026)

Contains adversarial failure modes designed to verify that the 4-Rung Blind Vault
catches 100% of injected bugs (driving Silent-Failure Rate SFR -> 0.0%):
1. Dropped Factor of 2: In diffusion operator, drops a factor of 2.
   Passes isfinite and mass conservation, but caught by Rung 4 MMS.
2. 1st-Order Upwind Smearing: Introduces artificial numerical viscosity alpha_num = |v|*dx / 2.
   Passes naive energy checks, but caught by Rung 3 Step-Consistency and Rung 4 MMS order degradation.
3. Static Field Cheating: Outputs u(x,t) = u(x,0).
   Preserves invariants trivially, but caught by Rung 3 discrete PDE residual.
4. Goodhart Rescaling Hack: Appends u = u * (M0 / sum(u)).
   Forces mass drift to 0.00%, but violates discrete operator residual.
5. CFL / von Neumann Overflow: Exceeds stability bounds, triggering NaN contagion.
   Caught by Rung 1 IEEE 754 guard.
"""

from __future__ import annotations
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


def buggy_dropped_factor_of_two_heat_1d(
    N: int = 50,
    dt: Optional[float] = None,
    t_end: float = 0.1,
    alpha: float = 0.05,
    source_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    exact_bc_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    domain: Tuple[float, float] = (0.0, 1.0),
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    BUGGY: Drops a factor of 2 in diffusion coefficient (alpha * 0.5).
    Preserves mass conservation and passes isfinite, but fails Blind MMS verification!
    """
    x = np.linspace(domain[0], domain[1], N, endpoint=False)
    dx = (domain[1] - domain[0]) / N
    safe_dt = 0.4 * (dx * dx) / (2.0 * alpha)
    dt = safe_dt if dt is None else min(dt, safe_dt)

    n_steps = max(1, int(math.ceil(t_end / dt)))
    dt = t_end / n_steps

    if exact_bc_fn is not None:
        u = exact_bc_fn(x, 0.0)
    else:
        u = np.sin(np.pi * x)

    trajectory = [u.copy()]
    t = 0.0

    for step in range(n_steps):
        # BUG: missing factor of 2 in coefficient (0.5 * alpha instead of alpha)
        d2u = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)
        rhs = (0.5 * alpha) * d2u  # Injected bug!
        if source_fn is not None:
            rhs += source_fn(x, t)

        u = u + dt * rhs
        t += dt
        trajectory.append(u.copy())

    return u, x, trajectory


def buggy_first_order_upwind_advection_1d(
    N: int = 50,
    dt: Optional[float] = None,
    t_end: float = 0.1,
    velocity: float = 1.0,
    alpha: float = 0.01,
    source_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    exact_bc_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    domain: Tuple[float, float] = (0.0, 1.0),
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    BUGGY: Uses 1st-order Upwind instead of 2nd-order TVD-MUSCL.
    Injects artificial numerical viscosity alpha_num = 0.5 * |v| * dx, smearing fronts
    and causing phase drift. Passes naive energy check, but fails Rung 3 and Rung 4!
    """
    x = np.linspace(domain[0], domain[1], N, endpoint=False)
    dx = (domain[1] - domain[0]) / N
    safe_dt = 0.4 * min(dx / abs(velocity), (dx * dx) / (2.0 * alpha))
    dt = safe_dt if dt is None else min(dt, safe_dt)

    n_steps = max(1, int(math.ceil(t_end / dt)))
    dt = t_end / n_steps

    if exact_bc_fn is not None:
        u = exact_bc_fn(x, 0.0)
    else:
        u = np.exp(-((x - 0.3) ** 2) / 0.02)

    trajectory = [u.copy()]
    t = 0.0

    for step in range(n_steps):
        # 1st-order upwind stencil (diffusive smearing)
        if velocity >= 0:
            adv = -velocity * (u - np.roll(u, 1)) / dx
        else:
            adv = -velocity * (np.roll(u, -1) - u) / dx

        diff = alpha * (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)
        rhs = adv + diff
        if source_fn is not None:
            rhs += source_fn(x, t)

        u = u + dt * rhs
        t += dt
        trajectory.append(u.copy())

    return u, x, trajectory


def buggy_static_field_cheating(
    N: int = 50,
    dt: Optional[float] = None,
    t_end: float = 0.1,
    source_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    exact_bc_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    domain: Tuple[float, float] = (0.0, 1.0),
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    BUGGY: Returns static initial field u(x,t) = u(x,0).
    Trivially passes isfinite and initial extremum / mass checks, but fails discrete PDE residual!
    """
    x = np.linspace(domain[0], domain[1], N, endpoint=False)
    if exact_bc_fn is not None:
        u = exact_bc_fn(x, 0.0)
    else:
        u = np.sin(np.pi * x)

    # Returns static array without time evolution
    trajectory = [u.copy(), u.copy()]
    return u, x, trajectory


def buggy_goodhart_rescaling_hack(
    N: int = 50,
    dt: Optional[float] = None,
    t_end: float = 0.1,
    source_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    exact_bc_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    domain: Tuple[float, float] = (0.0, 1.0),
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    BUGGY: Appends Goodhart post-hoc rescaling u = u * (M0 / sum(u)).
    Fakes zero mass drift while corrupting the physics.
    """
    x = np.linspace(domain[0], domain[1], N, endpoint=False)
    dx = (domain[1] - domain[0]) / N
    dt = 0.001 if dt is None else dt
    n_steps = max(1, int(math.ceil(t_end / dt)))

    if exact_bc_fn is not None:
        u = exact_bc_fn(x, 0.0)
    else:
        u = np.sin(np.pi * x)

    m0 = float(np.sum(u))
    trajectory = [u.copy()]

    for step in range(n_steps):
        # Arbitrary wrong dynamics
        u = u * 1.05 + 0.01 * np.cos(x)
        # Goodhart rescaling cheat:
        u = u * (m0 / (np.sum(u) + 1e-12))
        trajectory.append(u.copy())

    return u, x, trajectory


def buggy_cfl_overflow_blowup(
    N: int = 50,
    dt: Optional[float] = None,
    t_end: float = 0.1,
    alpha: float = 1.0,
    domain: Tuple[float, float] = (0.0, 1.0),
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    BUGGY: Intentionally violates von Neumann stability bound dt >> dx^2 / (2*alpha),
    triggering exponential growth, float64 overflow to +/-Inf and Quiet NaN.
    """
    x = np.linspace(domain[0], domain[1], N, endpoint=False)
    dx = (domain[1] - domain[0]) / N
    # Heavily unstable time step (20x beyond von Neumann limit)
    dt_unstable = 20.0 * (dx * dx) / (2.0 * alpha)
    dt = dt_unstable

    n_steps = 220
    u = np.sin(np.pi * x)
    trajectory = [u.copy()]

    for step in range(n_steps):
        d2u = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)
        u = u + dt * (alpha * d2u)
        trajectory.append(u.copy())

    return u, x, trajectory
