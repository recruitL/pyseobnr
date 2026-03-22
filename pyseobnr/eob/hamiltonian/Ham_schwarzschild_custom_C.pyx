# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
"""
Schwarzschild-metric effective Hamiltonian for minimal EOB testing.

H_eff: effective Hamiltonian in Schwarzschild (geometric units, M=1 for the metric).
  A(r) = 1 - 2/r
  H_eff = sqrt(A + p_r*^2/A + L^2*A/r^2)
  xi = A  (tortoise factor: p_r = p_r* / xi)

Relation to real two-body Hamiltonian (user-requested):
  H_EOB = M * sqrt(1 + 2*nu*(H_eff/mu - 1))
  In the standard EOB conventions used elsewhere in this codebase,
  H_eff is already made dimensionless by the reduced mass mu, so we
  treat H_eff_schw computed below as this dimensionless H_eff.

The code stores H = H_EOB/nu (same convention as Ham_align), so
  H = M * sqrt(1 + 2*nu*(H_eff - 1)) / nu
"""
from libc.math cimport sqrt

from ..utils.containers cimport EOBParams, qp_param_t
from .Hamiltonian_C cimport (
    Hamiltonian_C,
    Hamiltonian_C_call_return_t,
    Hamiltonian_C_grad_return_t,
    Hamiltonian_C_dynamics_return_t,
    Hamiltonian_C_auxderivs_return_t,
)

import numpy as np
cimport numpy as np


# --------------- Schwarzschild H_eff and H_EOB (for .pxd / docs) ---------------
# A = 1 - 2/r
# H_eff = sqrt(A + prst^2/A + L^2*A/r^2)
# xi = A
# Here H_eff is treated as the usual EOB effective Hamiltonian per reduced mass,
# so we directly use it in H_EOB = M * sqrt(1 + 2*nu*(H_eff - 1)).
# H = H_EOB/nu = M * sqrt(1 + 2*nu*(H_eff - 1)) / nu


cdef inline double _A_schw(double r):
    if r <= 2.0:
        return 0.0
    return 1.0 - 2.0 / r


cdef inline double _H_eff_schw(double r, double prst, double L, double A):
    cdef double term = A + prst * prst / A + L * L * A / (r * r)
    if term <= 0.0:
        return 1.0
    return sqrt(term)


cdef class Ham_schwarzschild_custom_C(Hamiltonian_C):
    """
    EOB Hamiltonian with Schwarzschild H_eff and
    H_EOB = M * sqrt(1 + 2*nu*(H_eff/mu - 1)), mu = M*nu.
    """

    def __cinit__(self, EOBParams eob_params not None):
        self.EOBpars = eob_params

    cpdef Hamiltonian_C_call_return_t _call(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
    ):
        cdef double r = q[0]
        cdef double prst = p[0]
        cdef double L = p[1]
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double A = _A_schw(r)
        cdef double H_eff_schw = _H_eff_schw(r, prst, L, A)
        # Treat H_eff_schw as the dimensionless effective Hamiltonian per mu
        cdef double H_EOB = M * sqrt(1.0 + 2.0 * nu * (H_eff_schw - 1.0))
        cdef double H = H_EOB / nu
        cdef double xi = A
        if r <= 2.0 or H != H:
            raise ValueError("Incorrect domain (r or H)")
        return (H, xi, A, A, 0.0, 0.0, H_eff_schw, 0.0)

    cpdef Hamiltonian_C_grad_return_t grad(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
    ):
        cdef double r = q[0]
        cdef double prst = p[0]
        cdef double L = p[1]
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double A = _A_schw(r)
        cdef double H_eff_schw = _H_eff_schw(r, prst, L, A)
        # H = M/nu * sqrt(1 + 2*nu*(H_eff_schw - 1))
        cdef double den = sqrt(1.0 + 2.0 * nu * (H_eff_schw - 1.0))
        if den <= 0.0 or r <= 2.0:
            raise ValueError("Incorrect domain (grad)")
        cdef double dA_dr = 2.0 / (r * r)
        cdef double dH_eff_dr = (1.0 / (2.0 * H_eff_schw)) * (
            dA_dr
            - 2.0 * prst * prst * dA_dr / (A * A)
            + L * L * (dA_dr / (r * r) - 2.0 * A / (r * r * r))
        )
        cdef double dH_eff_dprst = prst / (A * H_eff_schw)
        cdef double dH_eff_dL = L * A / (H_eff_schw * r * r)
        # Chain rule: H = (M/nu) * den,  den = sqrt(1 + 2*nu*(H_eff_schw - 1))
        # dH/dx = (M/nu) * (1/(2*den)) * 2*nu * dH_eff_schw/dx = M * dH_eff_schw/dx / den
        cdef double fac = M / den
        cdef double dHdr = fac * dH_eff_dr
        cdef double dHdphi = 0.0
        cdef double dHdpr = fac * dH_eff_dprst
        cdef double dHdpphi = fac * dH_eff_dL
        return (dHdr, dHdphi, dHdpr, dHdpphi)

    cpdef double csi(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
    ):
        cdef double r = q[0]
        return _A_schw(r)

    cpdef Hamiltonian_C_dynamics_return_t dynamics(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
    ):
        cdef Hamiltonian_C_grad_return_t g = self.grad(q, p, chi_1, chi_2, m_1, m_2)
        cdef Hamiltonian_C_call_return_t c = self._call(q, p, chi_1, chi_2, m_1, m_2)
        cdef double xi = c[1]
        cdef double H_val = c[0]
        cdef double omega = self.omega(q, p, chi_1, chi_2, m_1, m_2)
        return (g[0], g[1], g[2], omega, H_val, xi)

    cpdef double omega(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
    ):
        cdef Hamiltonian_C_grad_return_t g = self.grad(q, p, chi_1, chi_2, m_1, m_2)
        return g[3]

    cpdef hessian(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
    ):
        cdef double eps = 1e-8
        cdef np.ndarray[np.float64_t, ndim=2] hess = np.zeros((4, 4), dtype=np.float64)
        cdef Hamiltonian_C_grad_return_t g0, g1
        cdef qp_param_t qp, pp
        cdef int i, j
        cdef double coord[4]
        coord[0] = q[0]
        coord[1] = q[1]
        coord[2] = p[0]
        coord[3] = p[1]
        for i in range(4):
            for j in range(4):
                if i < 2:
                    qp = (coord[0] + (eps if i == 0 else 0.0), coord[1] + (eps if i == 1 else 0.0))
                    pp = (coord[2], coord[3])
                else:
                    qp = (coord[0], coord[1])
                    pp = (coord[2] + (eps if i == 2 else 0.0), coord[3] + (eps if i == 3 else 0.0))
                g1 = self.grad(qp, pp, chi_1, chi_2, m_1, m_2)
                if i < 2:
                    qp = (coord[0] - (eps if i == 0 else 0.0), coord[1] - (eps if i == 1 else 0.0))
                    pp = (coord[2], coord[3])
                else:
                    qp = (coord[0], coord[1])
                    pp = (coord[2] - (eps if i == 2 else 0.0), coord[3] - (eps if i == 3 else 0.0))
                g0 = self.grad(qp, pp, chi_1, chi_2, m_1, m_2)
                hess[j, i] = (g1[j] - g0[j]) / (2.0 * eps)
                hess[i, j] = hess[j, i]
        return hess

    cpdef Hamiltonian_C_auxderivs_return_t auxderivs(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
    ):
        cdef double r = q[0]
        cdef double dA_dr = 2.0 / (r * r) if r > 2.0 else 0.0
        return (dA_dr, 0.0, 0.0, dA_dr, 0.0, 0.0, 0.0)
