"""
Setup various initial conditions for the simulation.
"""

from lbm_helpers import compute_feq
from grid_quantity import GridQuantity
import numpy as np
from sim_variables import SimulationVariables


def setup_cos_cos(nx: int, ny: int, cx, cy, rho_0: np.float64 = 1.0):

    # BROKEN PLEASE FIX THIS
    u = np.zeros((nx, ny), dtype=np.float64)
    v = np.zeros((nx, ny), dtype=np.float64)

    for x in range(nx):
        for y in range(ny):
            u[x, y] = 0.5 * np.cos(2 * np.pi * y / ny)
            v[x, y] = 0.5 * np.cos(2 * np.pi * x / nx)

    u, v = u * 0.01, v * 0.01

    u_max = np.sqrt(np.max(u**2 + v**2)) + 1e-6

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



def setup_kelvin_helmholtz_rotated(resolution: int, 
                                   horizontal_velocity: float = 0.5, 
                                   vertical_velocity: float = 0.005, 
                                   medium="air"):
    """
    I want to set this up so that the simulation happens in a unit square, 
    in meters. 
    """

    l_ref = 1.0 # metres

    if medium == "air":
        density = 1.293 # kg/m^3
        rho_ref = 1.293 # kg/m^3
        speed_of_sound = 343.0 # m/s
        kinematic_viscosity = 1.5 * 1e-5
    elif medium == "water":
        density = 1000.0 # kg/m^3
        rho_ref = 1000.0 # kg/m^3
        speed_of_sound = 1500.0 # m/s
        kinematic_viscosity = 1002.0 * 1e-6  # kinematic viscosity, m^2/s

    # u_ref = 2 * np.linalg.norm([horizontal_velocity, vertical_velocity]) # m/s
    u_ref = (horizontal_velocity ** 2 + vertical_velocity ** 2) ** 0.5

    Re = u_ref * l_ref / kinematic_viscosity
    print(f"Reynolds number: {Re}")

    delta_x = l_ref / resolution

    x_arr, y_arr, = np.meshgrid(np.arange(0, delta_x * resolution, delta_x), np.arange(0, delta_x * resolution, delta_x), indexing='ij')
    nx, ny = x_arr.shape

    layer = np.logical_or(
                np.logical_or(
                        np.logical_and(
                            y_arr + np.sqrt(2)/6 >= 1 - x_arr, 
                            y_arr - np.sqrt(2)/6 <= 1 - x_arr
                        ),
                        # False
                        y_arr <= np.sqrt(2)/6 - x_arr
                    ),
                    # False 
                y_arr >= 2 - np.sqrt(2)/6 - x_arr
            )

    u = np.zeros((nx, ny), dtype=np.float64) 
    v = np.zeros((nx, ny), dtype=np.float64)
    u[layer] = -horizontal_velocity / np.sqrt(2)
    v[layer] = horizontal_velocity / np.sqrt(2)
    u[~layer] = horizontal_velocity / np.sqrt(2)
    v[~layer] = -horizontal_velocity / np.sqrt(2)

    for i in range(nx):
        for j in range(ny):
            x, y = x_arr[i, j], y_arr[i, j]

            z = (1 - (y - x)) / 2
            # x, y are in the range [0, 1]
            assert 0 <= x <= 1
            assert 0 <= y <= 1
            assert 0 <= z <= 1
            v[i, j] += vertical_velocity * np.sin(4 * np.pi * z) / np.sqrt(2)
            u[i, j] += vertical_velocity * np.sin(4 * np.pi * z) / np.sqrt(2)

    ink = GridQuantity(nx, ny, 0, 0, periodic=True)
    ink.quantity[layer] = 1.0

    rho = np.ones((nx, ny), dtype=np.float64) * density

    physical_vars = SimulationVariables(l_ref, u_ref, rho_ref, speed_of_sound, kinematic_viscosity, delta_x)
    physical_vars.set_space(x_arr, y_arr, rho)
    physical_vars.set_velocity(u, v)

    return physical_vars, ink




def setup_kelvin_helmholtz(resolution: int, 
                           horizontal_velocity: float = 0.5, 
                           vertical_velocity: float = 0.005,
                           medium="air"):
    """
    I want to set this up so that the simulation happens in a unit square, 
    in meters. 
    """

    l_ref = 1.0 # metres

    if medium == "air":
        density = 1.293 # kg/m^3
        rho_ref = 1.293 # kg/m^3
        speed_of_sound = 343.0 # m/s
        kinematic_viscosity = 1.5 * 1e-5
    elif medium == "water":
        density = 1000.0 # kg/m^3
        rho_ref = 1000.0 # kg/m^3
        speed_of_sound = 1500.0 # m/s
        kinematic_viscosity = 1002.0 * 1e-6  # kinematic viscosity, m^2/s

    # u_ref = 2 * np.linalg.norm([horizontal_velocity, vertical_velocity]) # m/s
    u_ref = (horizontal_velocity ** 2 + vertical_velocity ** 2) ** 0.5

    Re = u_ref * l_ref / kinematic_viscosity
    print(f"Reynolds number: {Re}")

    delta_x = l_ref / resolution

    x_arr, y_arr, = np.meshgrid(np.arange(0, delta_x * resolution, delta_x), np.arange(0, delta_x * resolution, delta_x), indexing='ij')
    nx, ny = x_arr.shape

    layer = np.logical_and(y_arr >= 1/3, y_arr <= 2/3)

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

    physical_vars = SimulationVariables(l_ref, u_ref, rho_ref, speed_of_sound, kinematic_viscosity, delta_x)
    physical_vars.set_space(x_arr, y_arr, rho)
    physical_vars.set_velocity(u, v)

    return physical_vars, ink
