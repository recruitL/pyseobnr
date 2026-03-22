# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
# cython: profile=False, linetrace=False, binding=True
"""
Non-spin custom EOB Hamiltonian: HS + afun_cafun (a2,a3,a4) and
H = M*sqrt(1+2*nu*(H_eff-1))/nu.  Uses scipy.special (Spence, elliptic K/E).
"""

import cython

import numpy as np
cimport numpy as np

from libc.math cimport sqrt, log, asinh, acosh, fabs
from scipy.special.cython_special cimport spence, ellipk, ellipe

from ..utils.containers cimport EOBParams, qp_param_t
from .Hamiltonian_C cimport (
  Hamiltonian_C,
  Hamiltonian_C_call_return_t,
  Hamiltonian_C_grad_return_t,
  Hamiltonian_C_dynamics_return_t,
  Hamiltonian_C_auxderivs_return_t
)

##########

cdef double PI = 3.141592653589793238462643383279502884

cdef inline double HS(double r, double phi, double pr, double pphi) noexcept:
    cdef double l2 = pphi * pphi
    cdef double u = 1.0 / r
    cdef double tu = 2.0 * u
    cdef double u2 = u * u
    return sqrt((1.0 - tu) * (1.0 + (1.0 - tu) * pr * pr + l2 * u2))

cdef tuple afun_cafun(double nu, double gamma0):
    """
    input: gamma0 is the effective energy which number type is float
    output: the coefficient a2, a3, a4, .0.0.0
    """
    # 能量的解析延拓
    cdef double gamma = sqrt(2 - gamma0 * gamma0)

    # 高次幂
    cdef double gamma2 = gamma * gamma
    cdef double gamma3 = gamma2 * gamma
    cdef double gamma4 = gamma3 * gamma
    cdef double gamma5 = gamma4 * gamma
    cdef double gamma6 = gamma5 * gamma
    cdef double gamma7 = gamma6 * gamma
    cdef double gamma8 = gamma7 * gamma
    cdef double gamma9 = gamma8 * gamma
    cdef double gamma10 = gamma9 * gamma
    cdef double gamma11 = gamma10 * gamma
    cdef double gamma12 = gamma11 * gamma
    cdef double gamma13 = gamma12 * gamma
    cdef double gamma14 = gamma13 * gamma
    cdef double gamma15 = gamma14 * gamma
    cdef double gamma16 = gamma15 * gamma
    cdef double gamma17 = gamma16 * gamma
    cdef double gamma18 = gamma17 * gamma
    cdef double gamma19 = gamma18 * gamma
    cdef double gamma20 = gamma19 * gamma
    cdef double gamma21 = gamma20 * gamma
    
    # EOB 能量
    cdef double Gamma = sqrt(1.0 + 2.0 * nu * (gamma - 1.0))
    cdef double Gamma2 = Gamma * Gamma
    cdef double Gamma3 = Gamma2 * Gamma
    cdef double Gamma4 = Gamma3 * Gamma
    cdef double Gamma5 = Gamma4 * Gamma

    # 常数
    cdef double pi = PI
    cdef double pi2 = pi * pi
    
    # 一些重复表达式
    cdef double gammam1 = gamma - 1.0
    cdef double gammap1 = gamma + 1.0
    cdef double gammam1_2 = gammam1 * gammam1
    cdef double sqgm1d2 = sqrt(gammam1 / 2.0)
    
    # 三角/反三角函数
    cdef double arcsinh_sqgm1d2 = asinh(sqgm1d2) 
    cdef double arccosh_g = acosh(gamma)
    
    # 定义
    cdef double epsilon = gamma2 - 1.0
    cdef double epsilon1d2 = sqrt(epsilon)
    cdef double epsilon3d2 = sqrt(epsilon) * epsilon
    cdef double epsilon5d2 = epsilon3d2 * epsilon
    cdef double epsilon7d2 = epsilon5d2 * epsilon
    cdef double epsilon2 = epsilon * epsilon
    cdef double epsilon3 = epsilon2 * epsilon
    cdef double epsilon4 = epsilon3 * epsilon
    
    # P30 
    cdef double nudG = nu / Gamma2
    cdef double cp0 = (18.0 * gamma2 - 1.0)/ (2.0 * Gamma2)
    cdef double cp1 = 8.0 * nu * (3.0 + 12.0 * gamma2 - 4.0 * gamma4) / (Gamma2 * epsilon1d2) * arcsinh_sqgm1d2
    cdef double cp2 = nudG * (1.0 - 103.0 / 3.0 * gamma - 18.0 * gamma2 - 2.0 / 3.0 * gamma3 
                         + 3.0 * Gamma * (1.0 - 2.0 * gamma2) * (1.0 - 5.0 * gamma2) / ((1.0 + Gamma) * gammap1))
    cdef double p30 = cp0 + cp1 + cp2

    # 辐射反作用项
    cdef double ex_2gm1 = 2.0 * gamma2 - 1.0
    cdef double ex_2gm1_2 = ex_2gm1 * ex_2gm1
    ## chi_rr^3
    cdef double chirr3 = - 2.0 * nudG * (gamma / 3.0) * (ex_2gm1_2 / epsilon) * ((5.0 * gamma2 - 8.0)/ gamma + 2.0 * ((9.0 - 6.0 * gamma2)/epsilon1d2) * arcsinh_sqgm1d2 )
    ## chi_rr^4
    # h1 = -3.0 + 377.0*gamma2 - 1017.0*gamma4 + 515.0*gamma6
    cdef double h2 = 169.0 + 380.0*gamma2
    cdef double h3 = 834.0 + 2095.0*gamma + 1200.0*gamma2
    cdef double h4 = 1183.0 + 2929.0*gamma + 2660.0*gamma2 + 1200.0*gamma3
    cdef double h5 = -12.0 + 76.0*gamma - 129.0*gamma2 + 60.0*gamma3 + 30.0*gamma4 - 25.0*gamma6
    cdef double h6 = 1151.0 - 3336.0*gamma + 3148.0*gamma2 - 912.0*gamma3 + 339.0*gamma4 - 552.0*gamma5 + 210.0*gamma6
    cdef double h7 = -1.0*gamma*(-3.0 + 2.0*gamma2)*(4.0 - 15.0*gamma + 15.0*gamma2)
    # h8 = -1049.0 - 496.0*gamma + 8700.0*gamma2 - 16658.0*gamma3 + 9563.0*gamma4 + 13176.0*gamma5 - 15822.0*gamma6 - 1338.0*gamma7 + 3456.0*gamma8 + 420.0*gamma9
    cdef double h9 = ( - 210.0 - 210.0*gamma + 885.0*gamma2 + 885.0*gamma3 - 3457.0*gamma4 - 3457.0*gamma5 + 9593.0*gamma6 
          + 9593.0*gamma7 + 3259.0*gamma8 - 181493.0*gamma9 + 535259.0*gamma10 - 500785.0*gamma11 - 32675.0*gamma12 
          + 333545.0*gamma13 - 304761.0*gamma14 + 232751.0*gamma15 + 74431.0*gamma16 - 216185.0*gamma17 
          - 34080.0*gamma18 + 116100.0*gamma19 + 11340.0*gamma20 - 22680.0*gamma21 
          )
    # h10 = -129.0 + 366.0*gamma + 444.0*gamma2 - 1432.0*gamma3 + 27.0*gamma4 + 970.0*gamma5 + 50.0*gamma6 - 280.0*gamma7
    cdef double h11 =( 2074.0 + 10643.0*gamma 
          + 2835.0*gamma11 + 18958.0*gamma2 + 11391.0*gamma3 + 5242.0*gamma4 
          - 9826.0*gamma5 + 1818.0*gamma6 + 13198.0*gamma7 - 700.0*gamma8 - 10065.0*gamma9 )
    cdef double h12 = gamma*(5369.0 + 945.0*gamma10 + 8077.0*gamma2 - 5014.0*gamma4 + 4874.0*gamma6 - 2955.0*gamma8)
    cdef double h13 = gamma*(-1965.0 + 2169.0*gamma + 1289.0*gamma2 - 2211.0*gamma3 - 856.0*gamma4 + 90.0*gamma5 + 580.0*gamma6 + 280.0*gamma7)
    cdef double h14 = gamma*(-3.0 + 2.0*gamma2)*(85.0 - 82.0*gamma - 716.0*gamma2 + 380.0*gamma3 + 1537.0*gamma4 - 610.0*gamma5 - 890.0*gamma6 + 280.0*gamma7)
    cdef double h15 = -5.0 + 76.0*gamma - 150.0*gamma2 + 60.0*gamma3 + 35.0*gamma4
    cdef double h16 = gamma*(-3.0 + 2.0*gamma2)*(11.0 - 30.0*gamma2 + 35.0*gamma4)
    cdef double h17 = 299.0 - 1216.0*gamma + 1732.0*gamma2 - 960.0*gamma3 + 690.0*gamma4 - 860.0*gamma6 + 315.0*gamma8
    cdef double h18 = 21.0 + 65.0*gamma2 - 145.0*gamma4 + 315.0*gamma6
    # h19 = 102.0 - 4983.0*gamma + 12882.0*gamma2 - 11744.0*gamma3 - 2154.0*gamma4 + 20405.0*gamma5 - 17562.0*gamma6 + 234.0*gamma7 + 1932.0*gamma8 + 840.0*gamma9
    cdef double h20 = (-45.0 - 9872.0*gamma10 + 16138.0*gamma11 + 14128.0*gamma12 + 7824.0*gamma13 - 23840.0*gamma14 + 4320.0*gamma15 + 3600.0*gamma16 
           + 207.0*gamma2 - 1471.0*gamma4 + 13349.0*gamma6 - 37478.0*gamma7 + 63848.0*gamma8 - 47540.0*gamma9)
    # h21 = -124.0 + 285.0*gamma + 660.0*gamma2 - 1480.0*gamma3 - 400.0*gamma4 + 1425.0*gamma5 - 350.0*gamma7
    cdef double h22 = 1759.0 + 6744.0*gamma + 3692.0*gamma2 + 2044.0*gamma3 + 2787.0*gamma4 + 1112.0*gamma5 + 210.0*gamma6 - 300.0*gamma7
    cdef double h23 = gamma*(-852.0 - 283.0*gamma2 - 140.0*gamma4 + 75.0*gamma6)
    cdef double h24 = gamma*(-3.0 + 2.0*gamma2)*(1151.0 - 3504.0*gamma + 3148.0*gamma2 - 576.0*gamma3 + 339.0*gamma4 - 720.0*gamma5 + 210.0*gamma6)
    cdef double h25 = gamma*(-3.0 + 2.0*gamma2)*(96.0 - 93.0*gamma - 768.0*gamma2 + 432.0*gamma3 + 1632.0*gamma4 - 705.0*gamma5 - 960.0*gamma6 + 350.0*gamma7)
    cdef double h26 = (3.0 - 2.0*gamma2)**2 * gamma2 * (11.0 - 30.0*gamma2 + 35.0*gamma4)
    cdef double h27 = 8.0 + 19.0*gamma + 60.0*gamma2 + 15.0*gamma3
    cdef double h28 = gamma*(63.0 + 768.0*gamma2 - 645.0*gamma4 + 70.0*gamma6)
    cdef double h29 = 60.0 + 333.0*gamma2 + 90.0*gamma4 - 75.0*gamma6
    cdef double h30 = 12.0 + 76.0*gamma + 129.0*gamma2 + 60.0*gamma3 - 30.0*gamma4 + 25.0*gamma6
    # h61 = 35.0*(-1.0 + gamma)*(1.0 + gamma)*(1.0 - 18.0*gamma2 + 33.0*gamma4)
    cdef double h62 = (-45.0 - 
           15056.0*gamma10 - 25145.0*gamma11 + 27952.0*gamma12 + 33249.0*gamma13 - 35360.0*gamma14 + 4320.0*gamma15 + 3600.0*gamma16 
           + 207.0*gamma2 - 1471.0*gamma4 + 13349.0*gamma6 - 38135.0*gamma7 + 64424.0*gamma8 - 32177.0*gamma9 )
    cdef double h63 = gamma2*(-3.0 + 2.0*gamma2)*(-1.0 + 2.0*gamma2)*(11.0 - 30.0*gamma2 + 35.0*gamma4)
    cdef double h64 = -102.0 + 2681.0*gamma - 6210.0*gamma2 + 10052.0*gamma3 - 9366.0*gamma4 - 8491.0*gamma5 + 15018.0*gamma6 + 702.0*gamma7 - 4140.0*gamma8
    cdef double h65 = 124.0 - 295.0*gamma - 508.0*gamma2 + 1200.0*gamma3 + 216.0*gamma4 - 755.0*gamma5 - 240.0*gamma6 + 210.0*gamma7
    cdef double h66 = gamma*(-3.0 + 2.0*gamma2)*(-1.0 + 2.0*gamma2)*(11.0 - 30.0*gamma2 + 35.0*gamma4)
    cdef double h67 = (1.0 - 1.0*gamma)*(-947.0 - 3177.0*gamma + 14910.0*gamma2 - 26710.0*gamma3 + 18929.0*gamma4 + 21667.0*gamma5 - 30840.0*gamma6 - 2040.0*gamma7 + 7596.0*gamma8 + 420.0*gamma9)
    cdef double h68 = (-1.0 + gamma)*(253.0 - 661.0*gamma - 952.0*gamma2 + 2632.0*gamma3 + 189.0*gamma4 - 1725.0*gamma5 - 290.0*gamma6 + 490.0*gamma7)

    # part 1 of chi_rr^4 
    cdef double ln_gp1d2 = log(gammap1 / 2.0)
    cdef double n96d72 = 96.0 * epsilon7d2
    cdef double n16d52 = 16.0 * epsilon5d2
    cdef double n8d4 = 8.0 * epsilon4
    cdef double cfour64 = h64 / n96d72
    cdef double cfour65 = h65 / n16d52
    cdef double cfour63 = h63 / n8d4
    cdef double cfour25 = - h25 / (4.0 * n8d4)
    cdef double cfour67 = h67 / n96d72
    cdef double cfour68 = h68 / n16d52
    cdef double cfour14 = - (h14 * gammap1 + h25 * (gamma - 3.0))/(4.0 * n8d4)
    cdef double cfour66 = h66 * gammam1_2 /n8d4
    cdef double coeff = pi * nu * epsilon2 / Gamma5
    cdef double chirr4_p1 = coeff * ( cfour64 + cfour65 * ln_gp1d2 + cfour63 * arcsinh_sqgm1d2 + cfour25 * arccosh_g
                                            + nu * (cfour67 + cfour68 * ln_gp1d2 + cfour14 * arccosh_g + cfour66 * arcsinh_sqgm1d2) )
    
    # part 2 of chi_rr^4
    cdef double ln_g = log(gamma)
    cdef double ln_gp1d2_2 = ln_gp1d2 * ln_gp1d2
    cdef double gm1dgp1 = gammam1 / gammap1
    cdef double polylog2 = spence(1 - gm1dgp1)          # Li_2(z)
    cdef double polylog2_m = spence(1 + gm1dgp1) 
    cdef double arccosh_g_2 = arccosh_g * arccosh_g
    cdef double coe3 = gammap1 / (64.0 * epsilon3) 
    cdef double cff_lngp1d2 = (h11 + 2.0 * epsilon * h22) * coe3 
    cdef double cff_lng = - 2.0 * (h12 - 8.0 * epsilon * h23) * coe3
    cdef double cff_arcch = 3.0 * (2.0 * gammam1_2 * h13 - gammap1 * h24) / n96d72
    cdef double cff_arcch_lngp1d2 = 3.0 * (h16 + h28) * gammap1 / (2.0 * n16d52) 
    cdef double cff_0 = - (h9 - 4.0 * gamma2 * gammap1 * h20) / (1536.0 * gamma9 * epsilon3)
    cdef double cff_lngp1d2_2 = - 3.0 * (h15 - 4.0 * h27)/(16.0 * gammam1)
    cdef double cff_arcch_2 = - 3.0 * h26 * gammap1 / (8.0 * n8d4)
    cdef double cff_polym = 3.0 / 64.0 * gammap1 * h18 + h29 * gammap1 / (8.0 * epsilon)
    cdef double cff_poly = 3.0 * (h17 + 8.0 * h30) / (128.0 * gammam1) 

    cdef double chirr4_p2 = nu * coeff * (cff_lngp1d2 * ln_gp1d2 + cff_lng * ln_g + cff_arcch * arccosh_g + cff_arcch_lngp1d2 * arccosh_g * ln_gp1d2
                 + cff_0 + cff_lngp1d2_2 * ln_gp1d2_2 + cff_arcch_2 * arccosh_g_2 + cff_polym * polylog2_m + cff_poly * polylog2)
    cdef double chirr4 = chirr4_p1 + chirr4_p2

    # const_4
    cdef double sq_gmp = sqrt(gm1dgp1)
    cdef double polylog2_s = spence(1 - sq_gmp)
    cdef double ln_gm1d2 = log(gammam1/2)
    cdef double coeff3 = nu * 8.0/3.0 * epsilon2/Gamma3
    cdef double eplic_K = ellipk(gm1dgp1)
    cdef double eplic_K_2 = eplic_K * eplic_K
    cdef double eplic_E = ellipe(gm1dgp1)
    cdef double eplic_E_2 = eplic_E * eplic_E
    cdef double n32d2 = 32.0 * epsilon2
    cdef double n16d1 = 16.0 * epsilon
    cdef double cff3_kk = - 3.0 * h3 * eplic_K_2 / n32d2
    cdef double cff4_kk = 3.0 * h4 * eplic_E * eplic_K / n32d2
    cdef double cff5_0 = - pi2 * h5/n16d1
    cdef double cff27_ln2 = - 12.0 * h27 * ln_gp1d2_2 / n16d1 
    cdef double cff6_lnm = - h6 * ln_gm1d2 / n32d2
    cdef double cff15_lnmp = 3.0 * h15 * ln_gp1d2 * ln_gm1d2/n16d1
    cdef double cff22_lnp = - h22 * ln_gp1d2/n32d2
    cdef double cff23_lng = - 8.0 * h23 * ln_g / n32d2
    cdef double cff26_arcc2 = 3.0 * h26 * arccosh_g_2 / (8.0 * n8d4)
    cdef double cff24_arcc = 3.0 * h24 * arccosh_g / n96d72  
    cdef double cff16_arcclnm = - 3.0 * h16 * arccosh_g * ln_gm1d2 / (2.0 * n16d52)
    cdef double cff28_arcclnp = - 3.0 * h28 * arccosh_g * ln_gp1d2 / (2.0 * n16d52)
    cdef double cff62_0 = - h62 / (384.0 * gamma7 * epsilon3)
    cdef double cff2_ee = -21.0 * gammap1 * h2 * eplic_E_2 / (2.0 * n32d2)
    cdef double cff7_lis = - 3.0 * epsilon1d2 * h7 * polylog2_s * gammam1 / (2.0 * epsilon3) 
    cdef double cff29_lim = -2.0 * h29 * polylog2_m / n16d1
    cdef double cff730_li = (3.0 * epsilon1d2 * h7 * gammam1 / (8.0 * epsilon3) - 3.0 * h30 / n16d1) * polylog2 

    cdef double cc4p1 = coeff3 * ( (cff3_kk + cff4_kk + cff5_0 + cff27_ln2 + cff6_lnm + cff15_lnmp)
                      + (cff22_lnp + cff23_lng + cff26_arcc2 + cff24_arcc + cff16_arcclnm + cff28_arcclnp)
                      + (cff62_0 + cff2_ee + cff7_lis + cff29_lim + cff730_li) )
    cdef double cc4p2 = 2.0 * (2.0 * gamma2 - 1.0) * (cp1 + cp2)
    cdef double cc4 = cc4p1 - cc4p2
    
    cdef double x1 = (3.0 - 2.0 * Gamma - 3.0 * (15.0 - 8.0 * Gamma) * gamma2 + 6.0 * (25.0 -16.0 * Gamma) * gamma4) / (Gamma * (3.0 * gamma2 - 1.0))
    cdef double x2 = 4.0 / (epsilon * (5.0 * gamma2 - 1.0))
    cdef double a2 = 3.0 * (1.0 - Gamma) * (1.0 - 5.0 * gamma2) / (Gamma * (3.0 * gamma2 - 1.0))
    cdef double a2_2 = a2 * a2
    cdef double a3 = 3.0 / (2.0 * (4.0 * gamma2 - 1.0)) * (x1 - 2.0 * p30 - 2.0 * chirr3 / epsilon1d2)
    cdef double x21 = 2.0 * (2.0 * gamma2 - 1.0) / epsilon1d2 * chirr3
    cdef double x22 = 1.0 / 48.0 * epsilon * (6.0 * a2 * (1.0 - 65.0 * gamma2) + 3.0 * a2_2 * (17.0 * gamma2 - 1.0) + 4.0 * a3 * (1.0 - 41.0 * gamma2))
    cdef double x3 = 35.0 - 630.0 * gamma2 + 1155.0 * gamma4 + Gamma * (1.0 + Gamma) * (1.0 - 130.0 * gamma2 + 129.0 * gamma4) 
    cdef double a4 = x2 * (x21 - 8.0 * chirr4 / (3.0 * pi) - cc4 + x22) - x2 * (1.0 - Gamma) / (16.0 * Gamma3) * x3

    return a2, a3, a4

##########

@cython.cpow(True)
cpdef (double, double) evaluate_H(
    qp_param_t q,
    qp_param_t p,
    double chi_1,
    double chi_2,
    double m_1,
    double m_2,
    double M,
    double nu,
    double X_1,
    double X_2,
    double a6,
    double dSO
):
    """
    Evaluate the Hamiltonian and xi

    Args:
      q (tuple[double, double]): Canonical positions (r,phi).
      p (tuple[double, double]): Canonical momenta (prstar,pphi).
      chi1 (double): Dimensionless z-spin of the primary.
      chi2 (double): Dimensionless z-spin of the secondary.
      m_1 (double): Primary mass component.
      m_2 (double): Secondary mass component.
      M (double): Total mass.
      nu (double): Reduced mass ratio.
      X_1 (double): m_1/M
      X_2 (double): m_2/M
      a6 (double): nonspinning calibration parameter
      dSO (double): spin-orbit calibration parameter

    Returns:
        (tuple)  H,xi

    """
    # 坐标参数
    cdef double r = q[0]
    if r <= 0:
        raise ValueError("Incorrect domain")
    cdef double phi = 0.0
    cdef double prst = p[0]
    cdef double L = p[1]

    cdef double L2 = L * L 
    cdef double prst2 = prst * prst 
    
    # \gamma_0 = \hat{H}_{\text{Sch}} 
    cdef double A0 = 1.0 - 2.0/r
    cdef double pr0 = prst / A0
    cdef double hs = HS(r, phi, pr0, L)

    cdef double r2 = r * r
    cdef double r3 = r * r2
    cdef double r4 = r * r3

    cdef double c1 = 1.0 + L2 /r2 + 2.0 * prst2
    cdef double a2_ini, dummy1, dummy2
    a2_ini, dummy1, dummy2 = afun_cafun(nu, hs)

    cdef double heff = hs + a2_ini / (2.0 * hs) * c1 / r2

    cdef double a2, a3, a4
    a2, a3, a4 = afun_cafun(nu, heff)

    cdef double A = 1.0 - 2.0 / r + a2/r2 + a3/r3 + a4/r4
    cdef double xi = A 
    if xi != xi:
        raise ValueError("Incorrect domain (xi)")

    cdef double Heven = sqrt( A * (1.0 + L2 / r2) + prst2 )
    if Heven != Heven:
      raise ValueError("Incorrect domain (Heven)")
    cdef double Hodd = 0.0

    cdef double Heff = Heven + Hodd

    # Evaluate H_real/nu
    cdef double H = M * sqrt(1+2*nu*(Heff-1)) / nu
    if H != H:
        raise ValueError("Incorrect domain (H)")

    return H, xi

@cython.cpow(True)
cdef class Ham_nonspin_custom_C(Hamiltonian_C):
    """
    Custom non-spin EOB Hamiltonian (user-defined A(r) via afun_cafun / HS / heff).
    """

    def __cinit__(self, EOBParams eob_params not None):
        self.EOBpars = eob_params

    def __call__(
      self,
      qp_param_t q,
      qp_param_t p,
      double chi_1,
      double chi_2,
      double m_1,
      double m_2,
      bint verbose=False):

        cdef:
            double H
            double xi
            double A
            double Bnp
            double Bnpa
            double Qq
            double Heven
            double Hodd

        H, xi, A, Bnp, Bnpa, Qq, Heven, Hodd = self._call(q, p, chi_1, chi_2, m_1, m_2)
        if not verbose:
            return H, xi

        return H, xi, A, Bnp, Bnpa, Qq, Heven, Hodd

    cpdef Hamiltonian_C_call_return_t _call(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2):

        """
        Evaluate the aligned-spin SEOBNRv5HM Hamiltonian as well as several potentials.
        See Sec. 1B and 1C of [SEOBNRv5HM-notes]_.

        Args:
          q (tuple[double, double]): Canonical positions (r,phi).
          p (tuple[double, double]): Canonical momenta  (prstar,pphi).
          chi1 (double): Dimensionless z-spin of the primary.
          chi2 (double): Dimensionless z-spin of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.
          verbose (bint): Output additional potentials.

        Returns:
           (tuple)  H,xi, A, Bnp, Bnpa, Qq, Heven, Hodd

        """

        # Coordinate definitions
        cdef double r = q[0]
        if r <= 0:
            raise ValueError("Incorrect domain")
        cdef double phi = 0.0 
        cdef double prst = p[0]
        cdef double L = p[1]

        cdef double a6 = self.EOBpars.c_coeffs.a6
        cdef double dSO = self.EOBpars.c_coeffs.dSO

        # Extra quantities used in the Hamiltonian
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2

        cdef double L2 = L * L 
        cdef double prst2 = prst * prst 
    
        # \gamma_0 = \hat{H}_{\text{Sch}} 
        cdef double A0 = 1.0 - 2.0/r
        cdef double pr0 = prst / A0
        cdef double hs = HS(r, phi, pr0, L)

        cdef double r2 = r * r
        cdef double r3 = r * r2
        cdef double r4 = r * r3
    
        cdef double c1 = 1.0 + L2 /r2 + 2.0 * prst2
        cdef double a2_ini, dummy1, dummy2
        a2_ini, dummy1, dummy2 = afun_cafun(nu, hs)

        cdef double heff = hs + a2_ini / (2.0 * hs) * c1 / r2

        cdef double a2, a3, a4
        a2, a3, a4 = afun_cafun(nu, heff)
    
        cdef double A = 1.0 - 2.0 / r + a2/r2 + a3/r3 + a4/r4

        cdef double xi = A
        if xi != xi:
          raise ValueError("Incorrect domain (xi)")

        cdef double Heven = sqrt( A * (1.0 + L2 / r2) + prst2 )
        if Heven != Heven:
            raise ValueError("Incorrect domain (Heven)")
        cdef double Hodd = 0.0 
        cdef double Heff = Heven + Hodd

        cdef double Bnp = A - 1.0
        cdef double Bnpa = 0.0
        cdef double Qq = 0.0

        # Evaluate H_real/nu
        cdef double H = M * sqrt(1+2*nu*(Heff-1)) / nu
        if H != H:
            raise ValueError("Incorrect domain (H)")

        return H, xi, A, Bnp, Bnpa, Qq, Heven, Hodd

    cpdef Hamiltonian_C_grad_return_t grad(self, qp_param_t q,qp_param_t p,double chi_1,double chi_2,double m_1,double m_2):

        """
        Compute the gradient of the Hamiltonian in polar coordinates.

        Args:
          q (tuple[double, double]): Canonical positions (r,phi).
          p (tuple[double, double]): Canonical momenta  (prstar,pphi).
          chi1 (double): Dimensionless z-spin of the primary.
          chi2 (double): Dimensionless z-spin of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.

        Returns:
           (tuple) dHdr, dHdphi, dHdpr, dHdpphi

        """

        # Coordinate definitions
        cdef double r = q[0]
        cdef double prst = p[0]
        cdef double L = p[1]

        # 参数
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu

        # 动量的高次幂
        cdef double L2 = L * L 
        cdef double L3 = L * L2
        cdef double L4 = L * L3
        cdef double prst2 = prst * prst 
    
        # \gamma_0 = \hat{H}_{\text{Sch}} 
        cdef double A0 = 1.0 - 2.0/r
        cdef double pr0 = prst / A0
        cdef double hs = HS(r, 0 , pr0, L)

        cdef double u = 1.0 / r
        cdef double u2 = u * u
        cdef double u3 = u * u2
        cdef double u4 = u * u3
        cdef double u5 = u * u4
        cdef double u6 = u * u5

        # Evaluate Hamiltonian
        cdef double a6 = 0.0
        cdef double dSO = 0.0
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2
        cdef double H = evaluate_H(q,p,chi_1,chi_2,m_1,m_2,M,nu,X_1,X_2,a6,dSO)[0]
        
        cdef double c1 = 1.0 + L2 * u2 + 2.0 * prst2
        cdef double a2_ini, dummy1, dummy2
        a2_ini, dummy1, dummy2 = afun_cafun(nu, hs)

        cdef double heff = hs + a2_ini / (2.0 * hs) * c1 * u2

        cdef double a2, a3, a4
        a2, a3, a4 = afun_cafun(nu, heff)
        
        cdef double A = 1.0 - 2.0 * u + a2 * u2 + a3 * u3 + a4 * u4 
        cdef double A2 = A * A

        # 有效能量
        cdef double Heff = sqrt(A * (1.0 + L2 * u2 ) + prst2)
        cdef double Heff2 = Heff * Heff
        cdef double Heff3 = Heff * Heff2
        
        # 函数 A' , A''
        cdef double Ap = 2.0 * u2 - 2.0 * a2 * u3 - 3.0 * a3 * u4 - 4.0 * a4 * u5
        cdef double Ap2 = Ap * Ap
        cdef double App = - 4.0 * u3 + 6.0 * a2 * u4 + 12.0 * a3 * u5 + 20.0 * a4 * u6

        # Heff Jacobian expressions
        cdef double dHeffdr = (Ap + Ap * L2 * u2 - 2.0 * A * L2 * u3)/(2.0 * Heff)
        cdef double dHeffdphi = 0
        cdef double dHeffdpr = prst / Heff
        cdef double dHeffdpphi = (A * L * u2) / Heff

        cdef double dHdr = M * M * dHeffdr / (nu*H)
        cdef double dHdphi = M * M * dHeffdphi / (nu*H)
        cdef double dHdpr = M * M * dHeffdpr / (nu*H)
        cdef double dHdpphi = M * M * dHeffdpphi / (nu*H)

        if (dHdr != dHdr) or (dHdphi != dHdphi) or (dHdpr != dHdpr) or (dHdpphi != dHdpphi):
            raise ValueError("Incorrect domain")

        return dHdr, dHdphi, dHdpr, dHdpphi
    
    # Hessian metrix 
    cpdef hessian(self, qp_param_t q,qp_param_t p,double chi_1,double chi_2,double m_1,double m_2):

        """
        Evaluate the Hessian of the Hamiltonian. 

        Args:
          q (tuple[double, double]): Canonical positions (r,phi).
          p (tuple[double, double]): Canonical momenta  (prstar,pphi).
          chi1 (double): Dimensionless z-spin of the primary.
          chi2 (double): Dimensionless z-spin of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.

        Returns:
           (np.array)  d2Hdr2, d2Hdrdphi, d2Hdrdpr, d2Hdrdpphi, d2Hdrdphi, d2Hdphi2, d2Hdphidpr, d2Hdphidpphi, d2Hdrdpr, d2Hdphidpr, d2Hdpr2, d2Hdprdpphi, d2Hdrdpphi, d2Hdphidpphi, d2Hdprdpphi, d2Hdpphi2

        """

        # Coordinate definitions
        cdef double r = q[0]
        cdef double prst = p[0]
        cdef double L = p[1]

        # 参数
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu

        # 动量的高次幂
        cdef double L2 = L * L 
        cdef double L3 = L * L2
        cdef double L4 = L * L3
        cdef double prst2 = prst * prst 
    
        # \gamma_0 = \hat{H}_{\text{Sch}} 
        cdef double A0 = 1.0 - 2.0/r
        cdef double pr0 = prst / A0
        cdef double hs = HS(r, 0 , pr0, L)

        cdef double u = 1.0 / r
        cdef double u2 = u * u
        cdef double u3 = u * u2
        cdef double u4 = u * u3
        cdef double u5 = u * u4
        cdef double u6 = u * u5

        # H_EOB
        cdef double H = self._call(q,p,chi_1,chi_2,m_1,m_2)[0]
        
        cdef double c1 = 1.0 + L2 * u2 + 2.0 * prst2
        cdef double a2_ini, dummy1, dummy2
        a2_ini, dummy1, dummy2 = afun_cafun(nu, hs)

        cdef double heff = hs + a2_ini / (2.0 * hs) * c1 * u2

        cdef double a2, a3, a4
        a2, a3, a4 = afun_cafun(nu, heff)
        
        cdef double A = 1.0 - 2.0 * u + a2 * u2 + a3 * u3 + a4 * u4 
        cdef double A2 = A * A

        # 有效能量
        cdef double Heff = sqrt(A * (1.0 + L2 * u2 ) + prst2)
        cdef double Heff2 = Heff * Heff
        cdef double Heff3 = Heff * Heff2
        
        # 函数 A' , A''
        cdef double Ap = 2.0 * u2 - 2.0 * a2 * u3 - 3.0 * a3 * u4 - 4.0 * a4 * u5
        cdef double Ap2 = Ap * Ap
        cdef double App = - 4.0 * u3 + 6.0 * a2 * u4 + 12.0 * a3 * u5 + 20.0 * a4 * u6

        # Heff Jacobian expressions
        cdef double dHeffdr = (Ap + Ap * L2 * u2 - 2.0 * A * L2 * u3)/(2.0 * Heff)
        cdef double dHeffdphi = 0
        cdef double dHeffdpr = prst / Heff
        cdef double dHeffdpphi = (A * L * u2) / Heff
        
        cdef double x0 = App * (0.5 + 0.5 * L2 * u2) - 2.0 * Ap * L2 * u3 + 3.0 * A * L2 * u4
        cdef double x1 = Ap2 * (-0.25 - 0.5 * L2 * u2 - 0.25 * L4 * u4) + Ap * (A * L2 * u3 + A * L4 * u5) - 1.0 * A2 * L4 * u6
        # 对角元
        cdef double d2Heffdr2 = x0 / Heff + x1 / Heff3
        cdef double d2Heffdphi2 = 0
        cdef double d2Heffdpr2 = 1.0 / Heff - (1.0 * prst2) / Heff3
        cdef double d2Heffdpphi2 = (A * u2) / Heff - (1.0 * A2 * L2 * u4) / Heff3

        # 非对角元
        cdef double x2 = (Ap * L * u2 - 2.0 * A * L * u3)
        cdef double x3 = (Ap * (-0.5 * A * L * u2 - 0.5 * A * L3 * u4) + A2 * L3 * u5)
        cdef double d2Heffdrdphi = 0
        cdef double d2Heffdrdpr = (Ap * (-0.5 * prst - 0.5 * L2 * prst * u2) + A * L2 * prst * u3) / Heff3
        cdef double d2Heffdrdpphi = x2 / Heff + x3 / Heff3

        cdef double d2Heffdprdpphi = -(1.0 * A * L * prst * u2)/Heff3
        cdef double d2Heffdphidpr = 0
        cdef double d2Heffdphidpphi = 0
        

        # Compute H Hessian
        cdef double d2Hdr2 = (-(dHeffdr**2/H**3)*(M**2/nu) + d2Heffdr2/H) * M*M / nu
        cdef double d2Hdphi2 = (-(dHeffdphi**2/H**3)*(M**2/nu) + d2Heffdphi2/H) * M*M / nu
        cdef double d2Hdpr2 = (-(dHeffdpr**2/H**3)*(M**2/nu) + d2Heffdpr2/H) * M*M / nu
        cdef double d2Hdpphi2 = (-(dHeffdpphi**2/H**3)*(M**2/nu) + d2Heffdpphi2/H) * M*M / nu
        cdef double d2Hdrdphi = (-(dHeffdr*dHeffdphi/H**3)*(M**2/nu) + d2Heffdrdphi/H) * M*M / nu
        cdef double d2Hdrdpr = (-(dHeffdr*dHeffdpr/H**3)*(M**2/nu) + d2Heffdrdpr/H) * M*M / nu
        cdef double d2Hdrdpphi = (-(dHeffdr*dHeffdpphi/H**3)*(M**2/nu) + d2Heffdrdpphi/H) * M*M / nu
        cdef double d2Hdphidpr = (-(dHeffdphi*dHeffdpr/H**3)*(M**2/nu) + d2Heffdphidpr/H) * M*M / nu
        cdef double d2Hdphidpphi = (-(dHeffdphi*dHeffdpphi/H**3)*(M**2/nu) + d2Heffdphidpphi/H) * M*M / nu
        cdef double d2Hdprdpphi = (-(dHeffdpr*dHeffdpphi/H**3)*(M**2/nu) + d2Heffdprdpphi/H) * M*M / nu

        return np.array([d2Hdr2, d2Hdrdphi, d2Hdrdpr, d2Hdrdpphi, d2Hdrdphi, d2Hdphi2, d2Hdphidpr, d2Hdphidpphi, d2Hdrdpr, d2Hdphidpr, d2Hdpr2, d2Hdprdpphi, d2Hdrdpphi, d2Hdphidpphi, d2Hdprdpphi, d2Hdpphi2]).reshape(4, 4)

    cpdef double csi(self, qp_param_t q, qp_param_t p, double chi_1, double chi_2, double m_1, double m_2):
        """
        Compute the tortoise factor :math:`\\xi` to convert between :math:`p_r` and :math:`p_{r_*}`.

        Args:
          q (tuple[double, double]): Canonical positions (r,phi).
          p (tuple[double, double]): Canonical momenta  (prstar,pphi).
          chi1 (double): Dimensionless z-spin of the primary.
          chi2 (double): Dimensionless z-spin of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.

        Returns:
           (double) xi
        """

        # Coordinate definitions
        cdef double r = q[0]
        if r <= 0:
            raise ValueError("Incorrect domain")
        cdef double phi = 0.0
        cdef double prst = p[0]
        cdef double L = p[1]

        # Extra quantities used in the Hamiltonian
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu

        cdef double L2 = L * L 
        cdef double prst2 = prst * prst 
    
        # \gamma_0 = \hat{H}_{\text{Sch}} 
        cdef double A0 = 1.0 - 2.0/r
        cdef double pr0 = prst / A0
        cdef double hs = HS(r, phi, pr0, L)

        cdef double r2 = r * r
        cdef double r3 = r * r2
        cdef double r4 = r * r3
    
        cdef double c1 = 1.0 + L2 /r2 + 2.0 * prst2
        cdef double a2_ini, dummy1, dummy2
        a2_ini, dummy1, dummy2 = afun_cafun(nu, hs)

        cdef double heff = hs + a2_ini / (2.0 * hs) * c1 / r2

        cdef double a2, a3, a4
        a2, a3, a4 = afun_cafun(nu, heff)

        cdef double xi = 1.0 - 2.0 / r + a2/r2 + a3/r3 + a4/r4
        if xi != xi:
          raise ValueError("Incorrect domain (xi)")

        return xi

    cpdef Hamiltonian_C_dynamics_return_t dynamics(self, qp_param_t q,qp_param_t p,double chi_1,double chi_2,double m_1,double m_2):

        """
        Compute the dynamics from the Hamiltonian: dHdr, dHdphi, dHdpr, dHdpphi, H and xi.

        Args:
          q (tuple[double, double]): Canonical positions (r,phi).
          p (tuple[double, double]): Canonical momenta  (prstar,pphi).
          chi1 (double): Dimensionless z-spin of the primary.
          chi2 (double): Dimensionless z-spin of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.

        Returns:
           (tuple) dHdr, dHdphi, dHdpr, dHdpphi, H and xi
        """

        # Coordinate definitions
        cdef double r = q[0]
        cdef double prst = p[0]
        cdef double L = p[1]

        # 参数
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu

        # 动量的高次幂
        cdef double L2 = L * L 
        cdef double L3 = L * L2
        cdef double L4 = L * L3
        cdef double prst2 = prst * prst 
    
        # \gamma_0 = \hat{H}_{\text{Sch}} 
        cdef double A0 = 1.0 - 2.0/r
        cdef double pr0 = prst / A0
        cdef double hs = HS(r, 0 , pr0, L)

        cdef double u = 1.0 / r
        cdef double u2 = u * u
        cdef double u3 = u * u2
        cdef double u4 = u * u3
        cdef double u5 = u * u4
        cdef double u6 = u * u5
        
        cdef double c1 = 1.0 + L2 * u2 + 2.0 * prst2
        cdef double a2_ini, dummy1, dummy2
        a2_ini, dummy1, dummy2 = afun_cafun(nu, hs)

        cdef double heff = hs + a2_ini / (2.0 * hs) * c1 * u2

        cdef double a2, a3, a4
        a2, a3, a4 = afun_cafun(nu, heff)
        
        cdef double A = 1.0 - 2.0 * u + a2 * u2 + a3 * u3 + a4 * u4 
        cdef double A2 = A * A

        # 有效能量
        cdef double Heff = sqrt(A * (1.0 + L2 * u2 ) + prst2)
        cdef double Heff2 = Heff * Heff
        cdef double Heff3 = Heff * Heff2

        # Evaluate Hamiltonian
        cdef double a6 = 0.0
        cdef double dSO = 0.0
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2
        cdef double H,xi
        H,xi = evaluate_H(q,p,chi_1,chi_2,m_1,m_2,M,nu,X_1,X_2,a6,dSO)
        
        # 函数 A' , A''
        cdef double Ap = 2.0 * u2 - 2.0 * a2 * u3 - 3.0 * a3 * u4 - 4.0 * a4 * u5
        cdef double Ap2 = Ap * Ap
        cdef double App = - 4.0 * u3 + 6.0 * a2 * u4 + 12.0 * a3 * u5 + 20.0 * a4 * u6

        # Heff Jacobian expressions
        cdef double dHeffdr = (Ap + Ap * L2 * u2 - 2.0 * A * L2 * u3)/(2.0 * Heff)
        cdef double dHeffdphi = 0
        cdef double dHeffdpr = prst / Heff
        cdef double dHeffdpphi = (A * L * u2) / Heff
        cdef double  M2 = M * M
        cdef double  nuH = nu * H
        # Compute H Jacobian
        cdef double  dHdr = M2 * dHeffdr / nuH
        cdef double  dHdphi = M2 * dHeffdphi / nuH
        cdef double  dHdpr = M2 * dHeffdpr / nuH
        cdef double  dHdpphi = M2 * dHeffdpphi / nuH

        if (dHdr != dHdr) or (dHdphi != dHdphi) or (dHdpr != dHdpr) or (dHdpphi != dHdpphi):
            raise ValueError("Incorrect domain")

        return dHdr, dHdphi, dHdpr, dHdpphi, H, xi

    cpdef double omega(self, qp_param_t q,qp_param_t p,double chi_1,double chi_2,double m_1,double m_2):

        """
        Compute the orbital frequency from the Hamiltonian.

        Args:
          q (tuple[double, double]): Canonical positions (r,phi).
          p (tuple[double, double]): Canonical momenta  (prstar,pphi).
          chi1 (double): Dimensionless z-spin of the primary.
          chi2 (double): Dimensionless z-spin of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.

        Returns:
           (double) dHdpphi
        """
        # Coordinate definitions
        cdef double r = q[0]
        cdef double prst = p[0]
        cdef double L = p[1]

        # 参数
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu

        # 动量的高次幂
        cdef double L2 = L * L 
        cdef double L3 = L * L2
        cdef double L4 = L * L3
        cdef double prst2 = prst * prst 
    
        # \gamma_0 = \hat{H}_{\text{Sch}} 
        cdef double A0 = 1.0 - 2.0/r
        cdef double pr0 = prst / A0
        cdef double hs = HS(r, 0 , pr0, L)

        cdef double u = 1.0 / r
        cdef double u2 = u * u
        cdef double u3 = u * u2
        cdef double u4 = u * u3
        cdef double u5 = u * u4
        cdef double u6 = u * u5

        # Evaluate Hamiltonian
        cdef double a6 = 0.0
        cdef double dSO = 0.0
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2
        cdef double H = evaluate_H(q,p,chi_1,chi_2,m_1,m_2,M,nu,X_1,X_2,a6,dSO)[0]
        
        cdef double c1 = 1.0 + L2 * u2 + 2.0 * prst2
        cdef double a2_ini, dummy1, dummy2
        a2_ini, dummy1, dummy2 = afun_cafun(nu, hs)

        cdef double heff = hs + a2_ini / (2.0 * hs) * c1 * u2

        cdef double a2, a3, a4
        a2, a3, a4 = afun_cafun(nu, heff)
        
        cdef double A = 1.0 - 2.0 * u + a2 * u2 + a3 * u3 + a4 * u4 

        # 有效能量
        cdef double Heff = sqrt(A * (1.0 + L2 * u2 ) + prst2)
    
        # Heff Jacobian expressions
        cdef double dHeffdpphi = (A * L * u2) / Heff

        cdef double omega = M * M * dHeffdpphi / (nu*H)

        return omega

    cpdef Hamiltonian_C_auxderivs_return_t auxderivs(
        self,
        qp_param_t q,
        qp_param_t p,
        double chi_1,
        double chi_2,
        double m_1,
        double m_2):

        """
        Compute derivatives of the potentials which are used in the post-adiabatic approximation.

        Args:
          q (tuple[double, double]): Canonical positions (r,phi).
          p (tuple[double, double]): Canonical momenta  (prstar,pphi).
          chi1 (double): Dimensionless z-spin of the primary.
          chi2 (double): Dimensionless z-spin of the secondary.
          m_1 (double): Primary mass component.
          m_2 (double): Secondary mass component.

        Returns:
           (tuple) dAdr, dBnpdr, dBnpadr, dxidr, dQdr, dQdprst, dHodddr

        """

        # Coordinate definitions
        cdef double r = q[0]
        cdef double prst = p[0]
        cdef double L = p[1]

        # 参数
        cdef double M = self.EOBpars.p_params.M
        cdef double nu = self.EOBpars.p_params.nu

        # 动量的高次幂
        cdef double L2 = L * L 
        cdef double L3 = L * L2
        cdef double L4 = L * L3
        cdef double prst2 = prst * prst 
    
        # \gamma_0 = \hat{H}_{\text{Sch}} 
        cdef double A0 = 1.0 - 2.0/r
        cdef double pr0 = prst / A0
        cdef double hs = HS(r, 0 , pr0, L)

        cdef double u = 1.0 / r
        cdef double u2 = u * u
        cdef double u3 = u * u2
        cdef double u4 = u * u3
        cdef double u5 = u * u4
        cdef double u6 = u * u5

        # Evaluate Hamiltonian
        cdef double a6 = 0.0
        cdef double dSO = 0.0
        cdef double X_1 = self.EOBpars.p_params.X_1
        cdef double X_2 = self.EOBpars.p_params.X_2
        cdef double H = evaluate_H(q,p,chi_1,chi_2,m_1,m_2,M,nu,X_1,X_2,a6,dSO)[0]
        
        cdef double c1 = 1.0 + L2 * u2 + 2.0 * prst2
        cdef double a2_ini, dummy1, dummy2
        a2_ini, dummy1, dummy2 = afun_cafun(nu, hs)

        cdef double heff = hs + a2_ini / (2.0 * hs) * c1 * u2

        cdef double a2, a3, a4
        a2, a3, a4 = afun_cafun(nu, heff)
        
        cdef double A = 1.0 - 2.0 * u + a2 * u2 + a3 * u3 + a4 * u4 

        # 有效能量
        cdef double Heff = sqrt(A * (1.0 + L2 * u2 ) + prst2)
        
        # 函数 A' , A''
        cdef double Ap = 2.0 * u2 - 2.0 * a2 * u3 - 3.0 * a3 * u4 - 4.0 * a4 * u5

        cdef double dxidr = Ap
        cdef double dHodddr = 0.0
        cdef double dQdprst = 0.0
        cdef double dQdr = 0.0
        cdef double dBnpadr = Ap
        cdef double dBnpdr = 0.0
        cdef double dAdr = Ap

        return dAdr,dBnpdr,dBnpadr,dxidr,dQdr,dQdprst,dHodddr
