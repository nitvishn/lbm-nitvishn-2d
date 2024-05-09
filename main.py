# Imports 
import numpy as np

from constants import Nl, cx, cy
from writer import plot_velocities, array_to_png
from grid_quantity import GridQuantity, advect
from init_conditions import setup_cos_cos, setup_kelvin_helmholtz
from lbm_helpers import compute_feq

# Simulation variables
nx = 300
ny = nx  # Square grid
tau = 0.6  # Relaxation time
rho_0 = 1.0  # Initial density, 1 everywhere
nit = 2000  # Number of iterations
advect_dt = 1.0  # Advection timestep
repeat = 2  # Number of times to tile the rendering

# The initial conditions
F, ink = setup_kelvin_helmholtz(nx, ny)

# Simulation code
u, v = np.sum(F * cx, axis=2), np.sum(F * cy, axis=2)
u_grid = GridQuantity(nx, ny, 0, 0, periodic=True)
v_grid = GridQuantity(nx, ny, 0, 0, periodic=True)

new_ink = GridQuantity(nx, ny, 0, 0, periodic=True)

# Main loop
for it in range(nit):
    # Step 1: Compute macroscopic variables
    u = np.sum(F * cx, axis=2)
    v = np.sum(F * cy, axis=2)
    rho = np.sum(F, axis=2)

    # Compute equilibrium distribution
    Feq = compute_feq(rho, u, v)

    # Advect the ink 
    u_grid.quantity = u
    v_grid.quantity = v
    advect(new_ink, u_grid, v_grid, advect_dt, ink)
    ink.quantity = new_ink.quantity

    # Step 4: Collision step
    F += (Feq - F) / tau

    # Step 5: Renormalize
    # rho = compute_rho(F)
    # for l in range(Nl):
    #     F[:,:,l] *= rho_0/rho

    # Step 5: Streaming step
    for c_i in range(Nl):
        F = np.roll(F, (cx[c_i], cy[c_i]), axis=(0, 1))

    print(f"Iteration {it}")
    if it % 20 == 0:
        print(f"Writing frame {it}")
        array_to_png(v.T, f"output/v.png", repeat=repeat)
        array_to_png(u.T, f"output/u.png", repeat=repeat)
        array_to_png(ink.quantity.T, f"output/current.png", repeat=repeat)
