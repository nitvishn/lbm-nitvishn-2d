# Imports 
import os 
import shutil

import numpy as np

from constants import Nl, cx, cy
from writer import plot_velocities, array_to_png
from grid_quantity import GridQuantity, advect
from init_conditions import setup_cos_cos, setup_kelvin_helmholtz, setup_kelvin_helmholtz_rotated
from lbm_helpers import compute_feq

from merge_frames import merge_frames

import matplotlib.pyplot as plt

from simulation import Simulation

import time


def main(sim_id: int, sim_length: float, resolution: int, init_function, framerate=30, erase_dir=False, **kwargs):
    # If outputs/output_{sim_id} exists, raise an error
    if os.path.exists(f"outputs/output_{sim_id}") and not erase_dir:
        raise ValueError(f"outputs/output_{sim_id} already exists")
    
    if erase_dir:
        shutil.rmtree(f"outputs/output_{sim_id}", ignore_errors=True)

    os.makedirs(f"outputs/output_{sim_id}/frames")

    # Simulation variables
    dt_frame = 1 / framerate

    frame_number = 0
    iteration_number = 0

    time_since_last_frame = 0

    physical_time = 0.0
    real_time = time.time()

    physical_vars, ink = init_function(resolution, **kwargs)

    sim = Simulation(resolution, physical_vars)

    u_max = np.max(sim.physical_vars.u ** 2 + sim.physical_vars.v ** 2) ** 0.5

    while physical_time < sim_length:
        dt = sim.step()
        time_since_last_frame += dt
        physical_time += dt

        u, v = sim.physical_vars.u, sim.physical_vars.v
        u_grid = GridQuantity(resolution, resolution, 0, 0, periodic=True)
        v_grid = GridQuantity(resolution, resolution, 0, 0, periodic=True)

        u_grid.quantity = u / sim.physical_vars.delta_x
        v_grid.quantity = v / sim.physical_vars.delta_x

        new_ink = GridQuantity(resolution, resolution, 0, 0, periodic=True)
        advect(new_ink, u_grid, v_grid, dt, ink)
        ink = new_ink

        if time_since_last_frame >= dt_frame:
            print(f"Frame {frame_number} of {sim_length * framerate} at {physical_time} seconds, lagging by {time_since_last_frame - dt_frame} seconds")

            # print(f"Time since last frame: {time_since_last_frame} seconds over {dt_frame} seconds")
            # print(f"Physical time: {physical_time}")
            # print(f"Time since last frame: {time_since_last_frame}")
            # print(f"Lattice uref: {sim.lattice_vars.u_ref}")
            # print(f"Physical uref: {sim.physical_vars.u_ref}")
            # print(f"Max u: {np.max(u)}, Min u: {np.min(u)}")
            # print(f"Max v: {np.max(v)}, Min v: {np.min(v)}")
            # print(f"Max rho: {np.max(sim.lattice_vars.rho)}, Min rho: {np.min(sim.lattice_vars.rho)}")
            # print(f"Max F: {np.max(sim.F)}, Min F: {np.min(sim.F)}")
            # print(f"Max vel at init: {u_max}")
            # print(f"Max vel now: {np.max(u ** 2 + v ** 2) ** 0.5}")

            time_since_last_frame = 0

            # array_to_png(u.T, f"output/u.png", low=-u_max, high=u_max)
            # array_to_png(v.T, f"output/v.png", low=-u_max, high=u_max)

            # array_to_png(ink.quantity.T, f"output/current.png")
            array_to_png(ink.quantity.T, f"outputs/output_{sim_id}/frames/frame_{frame_number}.png")
            frame_number += 1

        if iteration_number % 100 == 0:
            print(f"Iteration {iteration_number} at {physical_time} seconds")
            print(f"Max vel now: {np.max(u ** 2 + v ** 2) ** 0.5}")
            print(f"Physical uref: {sim.physical_vars.u_ref}")

        iteration_number += 1

    # Merge the frames into a video
    init_function_name = init_function.__name__
    kwargs_str = "_".join([f"{key}={value}" for key, value in kwargs.items()]) if kwargs else ""
    merge_frames(f"outputs/output_{sim_id}/frames", f"outputs/output_{sim_id}/{init_function_name}_{resolution}_px_{sim_length}s_{kwargs_str}.mp4", framerate)


if __name__ == "__main__":
    # Do Kelvin-Helmholtz instability at various resolutions
    # sim_id = 8
    # length = 20 # seconds
    # resolution = 100
    # main(sim_id, length, resolution, init_function=setup_kelvin_helmholtz, erase_dir=True, medium="air")

    # sim_id = 7
    # length = 20 # seconds
    # resolution = 300
    # main(sim_id, length, resolution, init_function=setup_kelvin_helmholtz, erase_dir=True)

    # sim_id = 3
    # length = 8 # seconds
    # resolution = 700
    # main(sim_id, length, resolution, init_function=setup_kelvin_helmholtz, erase_dir=True)

    # # This one's just for fun
    # sim_id = 4
    # length = 20 # seconds
    # resolution = 1000
    # main(sim_id, length, resolution, init_function=setup_kelvin_helmholtz, erase_dir=True)


    # Rotated Kelvin-Helmholtz vs normal Kelvin-Helmholtz
    # sim_id = 10
    # length = 20 # seconds
    # resolution = 300
    # main(sim_id, length, resolution, init_function=setup_kelvin_helmholtz, erase_dir=True, 
    #      medium="air", horizontal_velocity=0.25, vertical_velocity=0.005)
    
    # sim_id = 11
    # length = 20 # seconds
    # resolution = 300
    # main(sim_id, length, resolution, init_function=setup_kelvin_helmholtz_rotated, erase_dir=True, 
    #      medium="air", horizontal_velocity=0.25, vertical_velocity=0.005)
    

    # Water vs air Kelvin-Helmholtz
    length = 20 # seconds
    resolution = 400
    
    # sim_id = 12
    # main(sim_id, length, resolution, init_function=setup_kelvin_helmholtz, erase_dir=True, 
        #  medium="water", horizontal_velocity=0.25, vertical_velocity=0.0025)
    
    sim_id = 13
    main(sim_id, length, resolution, init_function=setup_kelvin_helmholtz, erase_dir=True, 
         medium="air", horizontal_velocity=0.25, vertical_velocity=0.0025, framerate=60)

