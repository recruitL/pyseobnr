# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
"""
Non-spin custom Hamiltonian: only the Hamiltonian is evaluated in the non-spin
limit (chi_1=chi_2=0). Radiation reaction, flux, dynamics structure, and
post-adiabatic logic are unchanged and inherit from the align-spin model.

Use this with SEOBNRv5HM_opt by passing H=Ham_nonspin_custom_C and RR=SEOBNRv5RRForce().
To plug in a fully custom non-spin Hamiltonian formula, replace _call (and
optionally grad, dynamics, omega, csi, hessian, auxderivs) in this file.
"""
from ..utils.containers cimport EOBParams, qp_param_t
from .Hamiltonian_C cimport (
    Hamiltonian_C,
    Hamiltonian_C_call_return_t,
    Hamiltonian_C_grad_return_t,
    Hamiltonian_C_dynamics_return_t,
    Hamiltonian_C_auxderivs_return_t,
)
from .Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C cimport (
    Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C,
)


cdef class Ham_nonspin_custom_C(Hamiltonian_C):
    """
    Non-spin Hamiltonian that delegates to the aligned-spin Hamiltonian
    with chi_1=chi_2=0. RR and all other parts of the model inherit from
    the align-spin implementation.
    """

    cdef Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C _align_ham

    def __cinit__(self, EOBParams eob_params not None):
        self.EOBpars = eob_params
        self._align_ham = Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C(eob_params)

    cpdef Hamiltonian_C_call_return_t _call(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
    ):
        return self._align_ham._call(q, p, 0.0, 0.0, m_1, m_2)

    cpdef Hamiltonian_C_grad_return_t grad(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
    ):
        return self._align_ham.grad(q, p, 0.0, 0.0, m_1, m_2)

    cpdef hessian(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
    ):
        return self._align_ham.hessian(q, p, 0.0, 0.0, m_1, m_2)

    cpdef double csi(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
    ):
        return self._align_ham.csi(q, p, 0.0, 0.0, m_1, m_2)

    cpdef Hamiltonian_C_dynamics_return_t dynamics(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
    ):
        return self._align_ham.dynamics(q, p, 0.0, 0.0, m_1, m_2)

    cpdef double omega(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
    ):
        return self._align_ham.omega(q, p, 0.0, 0.0, m_1, m_2)

    cpdef Hamiltonian_C_auxderivs_return_t auxderivs(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2,
    ):
        return self._align_ham.auxderivs(q, p, 0.0, 0.0, m_1, m_2)
