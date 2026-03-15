# cython: language_level=3
"""
Schwarzschild H_eff Hamiltonian for minimal EOB testing.

Formulas (same as in .pyx):
  A(r) = 1 - 2/r
  H_eff = sqrt(A + p_r*^2/A + L^2*A/r^2)
  xi = A
  mu = M*nu (reduced mass)
  H_eff/mu - 1  =>  Heff_dimless = H_eff/(M*nu)
  H_EOB = M * sqrt(1 + 2*nu*(H_eff/mu - 1))
  H = H_EOB/nu (stored value)
"""
from .Hamiltonian_C cimport Hamiltonian_C
from ..utils.containers cimport EOBParams

cdef class Ham_schwarzschild_custom_C(Hamiltonian_C):
    pass
