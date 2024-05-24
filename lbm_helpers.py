import numpy as np

from constants import w, Nl, cx, cy

from sim_variables import LatticeVariables

def compute_feq(rho, u, v, c_s):
    # Use a fourth order expansion for the equilibrium distribution

    # Compute the equilibrium distribution
    Feq = np.zeros((u.shape[0], u.shape[1], Nl), dtype=np.float64)
    for i in range(Nl):
        cu = cx[i] * u + cy[i] * v
        Feq[:, :, i] = rho * w[i] * ( 1 
                                  + cu / c_s**2
                                    + 0.5 * (cu / c_s**2)**2
                                    - 0.5 * (u**2 + v**2) / c_s**2)

    return Feq
