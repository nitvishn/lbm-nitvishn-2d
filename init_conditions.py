"""
Setup various initial conditions for the simulation.
"""

from lbm_helpers import compute_feq
from grid_quantity import GridQuantity
import numpy as np
from constants import u_max_upperbound


def setup_cos_cos(nx: int, ny: int, cx, cy, rho_0: np.float64 = 1.0):
    u = np.zeros((nx, ny), dtype=np.float64)
    v = np.zeros((nx, ny), dtype=np.float64)

    for x in range(nx):
        for y in range(ny):
            u[x, y] = 0.5 * np.cos(2 * np.pi * y / ny)
            v[x, y] = 0.5 * np.cos(2 * np.pi * x / nx)

    u, v = u * 0.01, v * 0.01

    u_max = np.sqrt(np.max(u**2 + v**2)) + 1e-6
    v *= u_max_upperbound / u_max
    u *= u_max_upperbound / u_max

    f = compute_feq(rho_0, u, v, cx, cy)

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

    # print("In setup_cos_cos")
    # print(f"Max f: {np.max(f)}, Min f: {np.min(f)}")
    # print(f"Max u: {np.max(u)}, Min u: {np.min(u)}")
    # print(f"Max v: {np.max(v)}, Min v: {np.min(v)}")

    return f, ink



def setup_kelvin_helmholtz(resolution: int):
    """
    I want to set this up so that the simulation happens in a unit square, 
    in meters. 
    We are in AIR, so the speed of sound is 343 m/s, 
    and the kinematic viscosity is 1.5e-5 m^2/s.
    """

    horizontal_velocity = 8.0 # m/s
    vertical_velocity = 0.1 # m/s, then we'll multiply by sin(x)
    density = 1.293 / 165 # kg/m^3

    l_ref = 1.0 # m
    
    rho_ref = 1.0 # kg/m^3
    
    speed_of_sound = 343.0 # m/s
    kinematic_viscosity = 1.5e-5  # kinematic viscosity, m^2/s

    # u_ref = 2 * np.linalg.norm([horizontal_velocity, vertical_velocity]) # m/s
    u_ref = speed_of_sound / 2.0

    Re = u_ref * l_ref / kinematic_viscosity

    print(f"Reynolds number: {Re}")

    delta_x = l_ref / resolution

    x_arr, y_arr, = np.meshgrid(np.arange(0, 1, delta_x), np.arange(0, 1, delta_x), indexing='ij')
    nx, ny = x_arr.shape

    layer = np.logical_and(y_arr >= 0.25, y_arr <= 0.75)

    u = np.zeros((nx, ny), dtype=np.float64) 
    v = np.zeros((nx, ny), dtype=np.float64)
    u[layer] = horizontal_velocity
    u[~layer] = -horizontal_velocity

    for i in range(nx):
        for j in range(ny):
            x, y = x_arr[i, j], y_arr[i, j]
            v[i, j] = vertical_velocity * np.sin(4 * np.pi * x)

    ink = GridQuantity(nx, ny, 0, 0, periodic=True)
    ink.quantity[layer] = 1.0

    rho = np.ones((nx, ny), dtype=np.float64) * density

    return u, v, rho, kinematic_viscosity, ink, rho_ref, u_ref, l_ref, speed_of_sound, delta_x




def setup_kelvin_helmholtz_rotated(nx: int, ny: int, cx, cy, rho_0: np.float64 = 1.0):
    x_arr, y_arr = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    x_arr_norm = x_arr / nx
    y_arr_norm = y_arr / ny

    width = nx / 4
    layer = np.logical_or(
        np.logical_or(
            np.logical_and(
                x_arr - width / 2 <= y_arr, 
                y_arr <= x_arr + width / 2),
            np.logical_and(
                x_arr - width / 2 <= y_arr + ny, 
                y_arr + ny <= x_arr + width / 2)
        ),
                np.logical_and(
        x_arr - width / 2 <= y_arr - ny, 
        y_arr - ny <= x_arr + width / 2)
    )

    u = np.zeros((nx, ny), dtype=np.float64)
    v = np.zeros((nx, ny), dtype=np.float64)
    u[layer] = 1.0
    u[~layer] = -1.0
    v[layer] = 1.0
    v[~layer] = -1.0

    for i in range(nx):
        for j in range(ny):
            x, y = x_arr_norm[i, j], y_arr_norm[i, j]
            coeff = (x + y) / 2

            u[i, j] += 0.5 * np.sqrt(2.0) + np.sin(4 * np.pi * coeff)
            v[i, j] += 0.5 * np.sqrt(2.0) + np.sin(4 * np.pi * coeff)

    ink = GridQuantity(nx, ny, 0, 0, periodic=True)
    other_layer = y_arr >= ny / 2
    ink.quantity[layer] = 1.0
    ink.quantity += ink.rotated(-np.pi / 4, nx // 2, ny // 2).quantity

    u *= 0.01
    v *= 0.01

    f = compute_feq(rho_0, u, v, cx, cy)

    return f, ink



def setup_kelvin_helmholtz_no_unit_conversion(resolution: int):
    """
    I want to set this up so that the simulation happens in a unit square, 
    in meters. 
    We are in AIR, so the speed of sound is 343 m/s, 
    and the kinematic viscosity is 1.5e-5 m^2/s.
    """

    horizontal_velocity = 0.2 # m/s
    vertical_velocity = 0.01 # m/s, then we'll multiply by sin(x)
    density = 1.0 # kg/m^3

    l_ref = 1.0 # metres
    t_ref = 0.1 # seconds, the timescale of the simulation
    
    rho_ref = 1.0 # kg/m^3
    
    speed_of_sound = 343.0 # m/s
    kinematic_viscosity = 1.5e-5  # kinematic viscosity, m^2/s

    # u_ref = 2 * np.linalg.norm([horizontal_velocity, vertical_velocity]) # m/s
    u_ref = 2 * horizontal_velocity

    Re = u_ref * l_ref / kinematic_viscosity
    print(f"Reynolds number: {Re}")

    delta_x = l_ref / resolution

    x_arr, y_arr, = np.meshgrid(np.arange(0, delta_x * resolution, delta_x), np.arange(0, delta_x * resolution, delta_x), indexing='ij')
    nx, ny = x_arr.shape

    layer = np.logical_and(y_arr >= 0.25, y_arr <= 0.75)

    u = np.zeros((nx, ny), dtype=np.float64) 
    v = np.zeros((nx, ny), dtype=np.float64)
    u[layer] = horizontal_velocity
    u[~layer] = -horizontal_velocity

    for i in range(nx):
        for j in range(ny):
            x, y = x_arr[i, j], y_arr[i, j]
            # x, y are in the range [0, 1]
            assert 0 <= x <= 1
            v[i, j] = vertical_velocity * np.sin(4 * np.pi * x)

    ink = GridQuantity(nx, ny, 0, 0, periodic=True)
    ink.quantity[layer] = 1.0

    rho = np.ones((nx, ny), dtype=np.float64) * density

    return u, v, rho, kinematic_viscosity, ink, rho_ref, u_ref, l_ref, speed_of_sound, delta_x
