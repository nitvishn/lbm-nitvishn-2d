# Imports 
import numpy as np

from constants import Nl, cx, cy
from writer import plot_velocities, array_to_png
from grid_quantity import GridQuantity, advect
from init_conditions import setup_cos_cos, setup_kelvin_helmholtz,setup_kelvin_helmholtz_rotated
from lbm_helpers import compute_feq

import matplotlib.pyplot as plt

from simulation import Simulation

import time


# Clear output/frames
import os
import shutil
shutil.rmtree("output/frames", ignore_errors=True)
os.makedirs("output/frames")

# Simulation variables
resolution = 200  # Resolution of the simulation
nit = 100000  # Number of iterations
repeat = 1  # Number of times to tile the rendering

framerate = 12 # frames per second
dt_frame = 1 / framerate

time_since_last_frame = 0

real_time = time.time()

output_modby = 10

pause = 0.0 # seconds

physical_vars, ink = setup_kelvin_helmholtz(resolution)

sim = Simulation(resolution, physical_vars)

u_max = np.max(sim.physical_vars.u ** 2 + sim.physical_vars.v ** 2) ** 0.5

for it in range(nit):
    dt = sim.step()
    time_since_last_frame += dt

    u, v = sim.physical_vars.u, sim.physical_vars.v
    u_grid = GridQuantity(resolution, resolution, 0, 0, periodic=True)
    v_grid = GridQuantity(resolution, resolution, 0, 0, periodic=True)

    u_grid.quantity = u / sim.physical_vars.delta_x
    v_grid.quantity = v / sim.physical_vars.delta_x

    new_ink = GridQuantity(resolution, resolution, 0, 0, periodic=True)
    advect(new_ink, u_grid, v_grid, dt, ink)
    ink = new_ink

    if it % output_modby == 0:
        print(f"Outputting frame on iteration {it} of {nit}") 

        print("ETA: ", (time.time() - real_time) / (it + 1) * (nit - it - 1))

        print(f"Time since last frame: {time_since_last_frame}")
        print(f"Lattice uref: {sim.lattice_vars.u_ref}")
        print(f"Physical uref: {sim.physical_vars.u_ref}")
        print(f"Max u: {np.max(u)}, Min u: {np.min(u)}")
        print(f"Max v: {np.max(v)}, Min v: {np.min(v)}")
        print(f"Max rho: {np.max(sim.lattice_vars.rho)}, Min rho: {np.min(sim.lattice_vars.rho)}")
        print(f"Max F: {np.max(sim.F)}, Min F: {np.min(sim.F)}")
        print(f"Max vel at init: {u_max}")
        print(f"Max vel now: {np.max(u ** 2 + v ** 2) ** 0.5}")

        time_since_last_frame = 0

        array_to_png(u.T, f"output/u.png", low=-u_max, high=u_max)
        # array_to_png(u.T, f"output/frames/frame_{it}.png", low=-u_max, high=u_max)
        array_to_png(v.T, f"output/v.png", low=-u_max, high=u_max)

        array_to_png(ink.quantity.T, f"output/current.png")