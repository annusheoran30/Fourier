from .interface import PDESolverCallable
from .canonical_solvers import (
    solve_heat_1d,
    solve_heat_2d,
    solve_advection_diffusion_1d,
    solve_burgers_1d,
    solve_wave_2d,
    solve_poisson_2d,
    solve_helmholtz_2d,
    solve_gray_scott_2d,
    solve_allen_cahn_1d,
    solve_fisher_kpp_1d,
    solve_navier_stokes_cavity_2d,
)
from .buggy_solvers import (
    buggy_dropped_factor_of_two_heat_1d,
    buggy_first_order_upwind_advection_1d,
    buggy_static_field_cheating,
    buggy_goodhart_rescaling_hack,
    buggy_cfl_overflow_blowup,
)

__all__ = [
    "PDESolverCallable",
    "solve_heat_1d",
    "solve_heat_2d",
    "solve_advection_diffusion_1d",
    "solve_burgers_1d",
    "solve_wave_2d",
    "solve_poisson_2d",
    "solve_helmholtz_2d",
    "solve_gray_scott_2d",
    "solve_allen_cahn_1d",
    "solve_fisher_kpp_1d",
    "solve_navier_stokes_cavity_2d",
    "buggy_dropped_factor_of_two_heat_1d",
    "buggy_first_order_upwind_advection_1d",
    "buggy_static_field_cheating",
    "buggy_goodhart_rescaling_hack",
    "buggy_cfl_overflow_blowup",
]
