#!/usr/bin/env python3
"""
Initial conditions for **non-spin custom** EOB Hamiltonians (e.g. :class:`Ham_nonspin_custom_C`).

Motivation
----------
The generic aligned-spin routine :func:`initial_conditions_aligned_opt.computeIC_opt`
uses the same *equations* (quasi-circular + dissipative balance, Khalil et al.), but it is
meant for the full aligned-spin Hamiltonian where calibration/spin channels enter the
analytic Jacobian in a specific way.

For a custom conservative potential :math:`A(r)=1-2/r+a_2/r^2+\\cdots` with coefficients
:math:`a_i` that may be **PA constants** or **implicit functions of phase space**, you may
want to:

1. Keep **χ = 0** fixed and avoid any confusion with spin-parameter derivatives; and
2. Replace :func:`Hamiltonian_C.dynamics` / :func:`Hamiltonian_C.hessian` by **explicit**
   formulas for :math:`\\partial H/\\partial r`, :math:`\\partial H/\\partial p_{r^*}`, etc.,
   when you freeze :math:`a_i` or add the chain rule through :math:`h_{\\mathrm{eff}}`.

This module implements the **same** QC + dissipative system as
``initial_conditions_aligned_opt``, but:

* Hard-codes **χ₁ = χ₂ = 0** in all calls to ``H.dynamics``, ``H.hessian``, ``H.omega``;
* Exposes small helpers so you can later swap in **explicit** gradients without touching
  the aligned-spin Hamiltonian code paths.

Equations (same as aligned, χ = 0)
----------------------------------
**Conservative** (quasi-circular at :math:`p_{r^*}=0`): find :math:`(r,p_\\phi)` such that

    ∂H/∂p_φ = ω ,    ∂H/∂r = 0 .

**Dissipative**: find :math:`p_{r^*}` such that the radial velocity from RR matches
:math:`\\partial H/\\partial p_{r^*}` (Eq. (68) in Khalil2021, as in the aligned module).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Tuple

import numpy as np
from scipy.optimize import root, root_scalar

logger = logging.getLogger(__name__)

# Fixed non-spin (aligned on z-axis but χ = 0)
_CHI1: float = 0.0
_CHI2: float = 0.0


def conservative_residuals_from_H(
    r: float,
    pphi: float,
    omega: float,
    H: Any,
    m_1: float,
    m_2: float,
) -> Tuple[float, float]:
    """
    Conservative part of QC initial conditions using **only** ``H.dynamics`` with χ = 0.

    Residuals: ( ∂H/∂p_φ − ω ,  ∂H/∂r ) at q = (r,0), p = (0, pphi).

    Replace this body with **explicit** derivatives in r, pphi (and frozen a_i) when ready.
    """
    q = np.array([r, 0.0])
    p = np.array([0.0, pphi])
    grad = H.dynamics(q, p, _CHI1, _CHI2, m_1, m_2)
    dHdr = grad[0]
    dHdpphi = grad[3]
    return dHdpphi - omega, dHdr


def IC_cons_nonspin_custom(
    u: np.ndarray,
    omega: float,
    H: Any,
    m_1: float,
    m_2: float,
) -> np.ndarray:
    """
    Same structure as :func:`initial_conditions_aligned_opt.IC_cons`, but χ fixed to 0
    and **no** spin arguments passed through to the Hamiltonian beyond zeros.
    """
    r, pphi = u
    d1, d2 = conservative_residuals_from_H(r, pphi, omega, H, m_1, m_2)
    return np.array([d1, d2])


def IC_diss_nonspin_custom(
    u: float,
    r: float,
    pphi: float,
    H: Any,
    RR: Callable[..., Any],
    m_1: float,
    m_2: float,
    params: Any,
) -> float:
    """
    Dissipative balance for pr (same algebra as ``IC_diss`` in aligned_opt), χ = 0.
    """
    pr = u
    q = np.array([r, 0.0])
    p = np.array([pr, pphi])
    hess = H.hessian(q, p, _CHI1, _CHI2, m_1, m_2)
    d2Hdr2 = hess[0, 0]
    d2HdrdL = hess[3, 0]
    dLdr = -d2Hdr2 / d2HdrdL
    p_circ = np.array([0.0, p[1]])
    dynamics = H.dynamics(q, p, _CHI1, _CHI2, m_1, m_2)
    H_val = dynamics[4]
    omega = dynamics[3]
    omega_circ = H.omega(q, p_circ, _CHI1, _CHI2, m_1, m_2)
    RR_f = RR.RR(q, p, omega, omega_circ, H_val, params)
    xi = dynamics[5]
    rdot = 1 / xi * RR_f[1] / dLdr
    dHdpr = dynamics[2]
    return rdot - dHdpr


def computeIC_nonspin_custom_opt(
    omega: float,
    H: Any,
    RR: Callable[..., Any],
    chi_1: float,
    chi_2: float,
    m_1: float,
    m_2: float,
    **kwargs: Any,
) -> Tuple[float, float, float]:
    """
    Non-spin custom initial conditions: (r0, pphi0, pr0).

    Parameters ``chi_1``, ``chi_2`` are kept for the same call signature as
    :func:`initial_conditions_aligned_opt.computeIC_opt` but **must be zero** for this
    Hamiltonian; non-zero values are ignored and zeros are used internally.

    Optional kwargs:
        ``params``: EOBParams, required for the dissipative step (same as aligned).
    """
    if abs(chi_1) > 0.0 or abs(chi_2) > 0.0:
        logger.warning(
            "computeIC_nonspin_custom_opt: chi_1 and chi_2 should be 0 for non-spin "
            "custom H; using 0 internally."
        )

    # Initial guess (Newtonian), same as aligned
    r_guess = omega ** (-2.0 / 3)
    z = [r_guess, np.sqrt(r_guess)]

    res_cons = root(
        IC_cons_nonspin_custom,
        z,
        args=(omega, H, m_1, m_2),
        tol=6e-12,
    )
    if not res_cons.success:
        logger.error(
            "Conservative IC (nonspin custom) failed for "
            f"m1={m_1}, m2={m_2}, omega={omega}"
        )

    r0, pphi0 = res_cons.x

    res_diss = root_scalar(
        IC_diss_nonspin_custom,
        bracket=[-3e-2, 0],
        args=(r0, pphi0, H, RR, m_1, m_2, kwargs["params"]),
        xtol=1e-12,
        rtol=1e-10,
    )
    if not res_diss.converged:
        logger.error(
            "Dissipative IC (nonspin custom) failed for "
            f"m1={m_1}, m2={m_2}, omega={omega}"
        )

    pr0 = float(res_diss.root)
    return float(r0), float(pphi0), pr0


__all__ = [
    "conservative_residuals_from_H",
    "IC_cons_nonspin_custom",
    "IC_diss_nonspin_custom",
    "computeIC_nonspin_custom_opt",
]
