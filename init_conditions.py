"""
Setup various initial conditions for the simulation.
"""

from lbm_helpers import compute_feq
from grid_quantity import GridQuantity
import numpy as np


def setup_cos_cos(nx: int, ny: int, rho_0: np.float64 = 1.0):
    u = np.zeros((nx, ny), dtype=np.float64)
    v = np.zeros((nx, ny), dtype=np.float64)

    for x in range(nx):
        for y in range(ny):
            u[x, y] = 2.0 * np.cos(2 * np.pi * y / ny)
            v[x, y] = 2.0 * np.cos(2 * np.pi * x / nx)

    f = compute_feq(rho_0, u, v)

    # Normalize the distribution functions
    rho = np.sum(f, axis=2)
    for c_i in range(9):
        f[:, :, c_i] *= rho_0 / rho

    # Set up the ink
    x_arr, y_arr = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    top = y_arr >= ny / 2
    # bottom = Y < ny / 2

    ink = GridQuantity(nx, ny, 0, 0, periodic=True)
    ink.quantity[top] = 1.0

    return f, ink



def setup_kelvin_helmholtz(nx: int, ny: int, rho_0: np.float64 = 1.0):
    x_arr, y_arr = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    x_arr_norm = x_arr / nx
    y_arr_norm = y_arr / ny

    width = 1.0 / np.sqrt(8.0)
    layer = np.logical_and(0.5 - width / 2 < y_arr_norm, y_arr_norm <= 0.5 + width / 2)

    u = np.zeros((nx, ny), dtype=np.float64)
    v = np.zeros((nx, ny), dtype=np.float64)
    u[layer] = 1.0
    u[~layer] = -1.0

    for i in range(nx):
        for j in range(ny):
            x, y = x_arr_norm[i, j], y_arr_norm[i, j]
            v[i, j] = 0.5 * np.sin(4 * np.pi * x)

    ink = GridQuantity(nx, ny, 0, 0, periodic=True)
    ink.quantity[layer] = 1.0

    # angle =  np.pi/4
    # u_grid = GridQuantity(nx, ny, 0, 0, periodic=True)
    # v_grid = GridQuantity(nx, ny, 0, 0, periodic=True)
    # u_grid.quantity = u
    # v_grid.quantity = v
    # u_grid_rotated = u_grid.rotated(angle, nx / 2, ny / 2)
    # v_grid_rotated = v_grid.rotated(angle, nx / 2, ny / 2)
    # u_prime, v_prime = u_grid_rotated.quantity, v_grid_rotated.quantity

    # u = np.cos(-angle) * u_prime + np.sin(-angle) * v_prime
    # v = np.sin(-angle) * u_prime - np.cos(-angle) * v_prime

    # ink = ink.rotated(angle, nx / 2, ny / 2)


    f = compute_feq(rho_0, u, v)

    return f, ink
