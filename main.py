# Imports 
import numpy as np

from constants import Nl, cx_fixed, cy_fixed, u_max_upperbound
from writer import plot_velocities, array_to_png
from grid_quantity import GridQuantity, advect
from init_conditions import setup_cos_cos, setup_kelvin_helmholtz, setup_kelvin_helmholtz_no_unit_conversion, setup_kelvin_helmholtz_rotated
from lbm_helpers import compute_feq

import matplotlib.pyplot as plt

import time

# Simulation variables
resolution = 100
nit = 5000  # Number of iterations
repeat = 1  # Number of times to tile the rendering

output_modby = 1

pause = 0.0 # seconds

cx, cy = cx_fixed.copy(), cy_fixed.copy()

# The initial conditions
u, v, rho, kinematic_viscosity, ink, rho_ref, u_ref, l_ref, speed_of_sound, delta_x = setup_kelvin_helmholtz_no_unit_conversion(resolution)
t = 0

nx, ny = u.shape

speed_of_sound_lbm = 1.0 / np.sqrt(3)
u_ref_lbm = 0.5 * speed_of_sound_lbm
# speed_of_sound_lbm = 1.0 / np.sqrt(3)
u_lbm = u * (u_ref_lbm / u_ref)
v_lbm = v * (u_ref_lbm / u_ref)
rho_lbm = rho / delta_x
kinematic_viscosity_lbm = kinematic_viscosity * (u_ref_lbm / (u_ref * delta_x))
delta_t = delta_x * u_ref_lbm / u_ref 

advect_dt = 1.0

# print(u_lbm)
# print(v_lbm)
# print(kinematic_viscosity_lbm)
# print(rho_lbm)
print(f"Speed of sound: {speed_of_sound_lbm}")

F = compute_feq(rho_lbm, u_lbm, v_lbm, cx, cy, speed_of_sound_lbm, normalize=False)

# print(F)

# Simulation code
rho_lbm = np.sum(F, axis=2)
u_lbm = np.sum(F * cx, axis=2) / rho_lbm
v_lbm = np.sum(F * cy, axis=2) / rho_lbm

u_grid = GridQuantity(nx, ny, 0, 0, periodic=True)
v_grid = GridQuantity(nx, ny, 0, 0, periodic=True)
u_max_init = np.max(u_lbm**2 + v_lbm**2)

new_ink = GridQuantity(nx, ny, 0, 0, periodic=True)


M = np.array(
    [
        [1.0 for j in range(Nl)], # m0, 1
        [cx[j] for j in range(Nl)], # m1, c_j_x
        [cy[j] for j in range(Nl)], # m2, c_j_y
        [cx[j] ** 2 + cy[j] ** 2 for j in range(Nl)], # m3, c_j_x^2 + c_j_y^2
        [cx[j] ** 2 - cy[j] ** 2 for j in range(Nl)], # m4, c_j_x^2 - c_j_y^2
        [cx[j] * cy[j] for j in range(Nl)], # m5, c_j_x * c_j_y
        [cx[j]**2 * cy[j] for j in range(Nl)], # m6, c_j_x^2 * c_j_y
        [cx[j] * cy[j]**2 for j in range(Nl)], # m7, c_j_x * c_j_y^2
        [cx[j]**2 * cy[j]**2 for j in range(Nl)], # m8, c_j_x^2 * c_j_y^2
    ]
, dtype=np.float64)

second_order_relaxation = 1.0 / (3 * kinematic_viscosity_lbm + 0.5)
third_order_relaxation = 0.005

R = np.diag([0.0, 
             2.0, 2.0, 
             second_order_relaxation, second_order_relaxation, second_order_relaxation, 
             third_order_relaxation, third_order_relaxation, third_order_relaxation])
# R = np.eye(Nl) * 0.6 
R = R.astype(np.float64)
M_inv = np.linalg.inv(M)

C = - M_inv @ R @ M


# Clear the output/frames directory
import os
import shutil
shutil.rmtree("output/frames", ignore_errors=True)
os.makedirs("output/frames")

uref_list = []

# # Main loop
for it in range(nit):
    # Step 1: Determine whether to rescale the simulation
    rho_lbm = np.sum(F, axis=2)
    u_lbm = np.sum(F * cx, axis=2) / rho_lbm
    v_lbm = np.sum(F * cy, axis=2) / rho_lbm

    u_max = np.sqrt(np.max(u_lbm**2 + v_lbm**2)) + 1e-6
    if (u_max > u_max_upperbound * 1.1 or u_max < u_max_upperbound * 0.9):

        u_lbm *= u_max_upperbound / u_max 
        v_lbm *= u_max_upperbound / u_max
        u_ref_lbm = u_ref_lbm * u_max / u_max_upperbound
        u_max = np.sqrt(np.max(u_lbm**2 + v_lbm**2)) + 1e-6
        
        F = compute_feq(rho_lbm, u_lbm, v_lbm, cx, cy, speed_of_sound_lbm)

    # Step 1.5: Some basic checks
    # if u_ref > 1e12 or u_ref < 1e-12:
    #     print(f"u_ref {u_ref} too high")
    #     break
    # if np.isnan(F).any():
    #     print("NaNs detected in F")
    #     break
    # if F.min() < 0:
    #     print("Negative F detected")
    #     break

    # Step 2: Compute equilibrium distribution
    rho_lbm = np.sum(F, axis=2)
    u_lbm = np.sum(F * cx, axis=2) / rho_lbm
    v_lbm = np.sum(F * cy, axis=2) / rho_lbm
    Feq = compute_feq(rho_lbm, u_lbm, v_lbm, cx, cy, speed_of_sound_lbm, normalize=False)

    # Step 3: Advect the ink 
    u_grid.quantity = u_lbm
    v_grid.quantity = v_lbm
    advect(new_ink, u_grid, v_grid, advect_dt, ink)
    ink.quantity = new_ink.quantity

    # Step 4: Collision 
    F += np.einsum('ij,xyj->xyi', C, F - Feq)

    # Step 5: Streaming 
    for l in range(Nl):
        F[:,:,l] = np.roll(F[:,:,l], (cx[l], cy[l]), axis=(0, 1))

    # Output so that the world can see!
    if it % output_modby == 0:        
        print(f"Writing frame {it}")
        print(f"Uref: {u_ref}")
        print(f"Uref_lbm: {u_ref_lbm}")

        # uref_list.append(u_ref)

        print(f"Max u: {np.max(u_lbm)}, Min u: {np.min(u_lbm)}")
        print(f"Max v: {np.max(v_lbm)}, Min v: {np.min(v_lbm)}")
        # print(f"Max rho: {np.max(rho_lbm)}, Min rho: {np.min(rho_lbm)}")
        # print(f"Max ink: {np.max(ink.quantity)}, Min ink: {np.min(ink.quantity)}")
        print(f"Max F: {np.max(F)}, Min F: {np.min(F)}")
        # print(f"Max Feq: {np.max(Feq)}, Min Feq: {np.min(Feq)}")

        # print(f"F update norm: {np.linalg.norm((Feq - F)/tau)}")

        # array_to_png(u_lbm.T, f"output/frames/u_frame_{it}.png", repeat=repeat, low=-u_max_init, high=u_max_init)

        # array_to_png(u_lbm.T, f"output/u.png", repeat=repeat, low=-u_max_init, high=u_max_init)
        # array_to_png(v_lbm.T, f"output/v.png", repeat=repeat, low=-u_max_init, high=u_max_init)
        # array_to_png(ink.quantity.T, f"output/current.png", repeat=repeat)
        array_to_png(ink.quantity.T, f"output/frames/frame_{it}.png", repeat=repeat)
        
        if pause:
            time.sleep(pause)
