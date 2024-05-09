import numpy as np

from constants import w, Nl, cx, cy

def compute_feq(rho, u, v, normalize=True):
    nx, ny = u.shape
    feq = np.zeros((nx, ny, Nl), dtype=np.float64)
    u_sq = u ** 2 + v ** 2
    for vel_direction in range(Nl):
        cu = cx[vel_direction] * u + cy[vel_direction] * v
        feq[:, :, vel_direction] += rho * w[vel_direction] * (1 + 3 * cu + 9 / 2 * cu ** 2 - 3 / 2 * u_sq)

    if normalize:
        rho_eq = np.sum(feq, axis=2)
        for l in range(Nl):
            feq[:, :, l] *= rho / rho_eq

    return feq
