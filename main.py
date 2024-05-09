# Imports 
import numpy as np
from writer import plot_velocities
from grid_quantity import GridQuantity
from writer import array_to_png
import time

def compute_u(F):
    u = np.sum(F*cx, axis=2)
    return u

def compute_v(F):
    v = np.sum(F*cy, axis=2)
    return v

def compute_rho(F):
    rho = np.sum(F, axis=2)
    return rho


def compute_Feq(rho,u,v):
    # TODO: Check if this is correct
    Feq = np.zeros((Nx,Ny,Nl), dtype=np.float64)
    u_sq = u**2 + v**2
    for l in range(Nl):
         cu = cx[l]*u + cy[l]*v
         Feq[:,:,l] += rho * w[l]*(1 + 3*cu + 9/2*cu**2 - 3/2*u_sq)
    return Feq


def advect(u: GridQuantity, v: GridQuantity, dt: float, q: GridQuantity):
	"""
	Advects the quantity q using the velocities u and v. 
	Semi-Lagrangian advection is used, which means that the 
	velocity field is used to trace back in time where the
	quantity q came from.
	"""

	q_advected_vec = GridQuantity(q.x_size, q.y_size, q.stagger_x, q.stagger_y, q.periodic)

	x_arr, y_arr = np.meshgrid(range(q.x_size), range(q.y_size), indexing='ij')
	x_arr, y_arr = q.grid_to_world(x_arr, y_arr)

	x_mid_arr, y_mid_arr = x_arr - 0.5 * dt * u.sample(x_arr, y_arr), y_arr - 0.5 * dt * v.sample(x_arr, y_arr)

	x_p_arr, y_p_arr = x_arr - dt * u.sample(x_mid_arr, y_mid_arr), y_arr - dt * v.sample(x_mid_arr, y_mid_arr)

	q_advected_vec.quantity = q.sample(x_p_arr, y_p_arr)

	return q_advected_vec


# Constants
Nx = 400
Ny = Nx # Square grid
tau = 0.9 # Relaxation time
rho_0 = np.ones((Nx, Ny)) # Initial density, 1 everywhere

# Velocity directions corresponding to here, right, up, left, down, bottom-right, top-right, top-left, bottom-left.
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]) # weights
Nl = 9
cx = np.array([0, 1, 0, -1,  0,  1, 1, -1, -1])
cy = np.array([0, 0, 1,  0, -1, -1, 1,  1, -1])

# Initialization
X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
top = Y >= Ny/2
bottom = Y < Ny/2

# Ink
ink = GridQuantity(Nx, Ny, 0, 0, periodic=True)
ink.quantity[top] = 1.0

# F and stuff
u = np.zeros((Nx, Ny), dtype=np.float64)
for x in range(Nx):
        for y in range(Ny):
            u[x, y] = 2.0 * np.cos(2 * np.pi * y / Ny)
# u[top] = 8.0
# u[bottom] = 8.0
v = np.zeros((Nx, Ny), dtype=np.float64)
for x in range(Nx):
    for y in range(Ny):
        v[x, y] = 2.0 * np.cos(4 * np.pi * x / Nx)
F = compute_Feq(rho_0, u, v)
# F = np.ones((Nx, Ny, Nl))
# F[top, 1] = 2.0
# F[bottom, 3] = 2.0

rho = compute_rho(F)
for l in range(Nl):
    F[:,:,l] *= rho_0/rho
rho = compute_rho(F)
# print(rho)

u, v = compute_u(F), compute_v(F)

# plot_velocities(X, Y, u, v)

# Main loop
nit = 2000
for it in range(nit):
    # Step 1: Compute macroscopic variables
    u = compute_u(F)
    u_grid = GridQuantity(Nx, Ny, 0, 0, periodic=True)
    u_grid.quantity = u

    v = compute_v(F)
    v_grid = GridQuantity(Nx, Ny, 0, 0, periodic=True)
    v_grid.quantity = v

    rho = compute_rho(F)

    # Step 2: Compute equilibrium distribution
    Feq = compute_Feq(rho, u, v)

    # Step 3: Write and plot
    # plot_velocities(X, Y, u, v)

    # print(f"u: {u[0, 0]}")
    # print(f"v: {v[0, 0]}")
    # print(f"rho: {rho[0, 0]}")
    # print(f"F: {F[0, 0, :]}")
    # print(f"Feq: {Feq[0, 0, :]}")
    # array_to_png(ink.quantity.T, f"output/frames/frame_{it}.png", repeat=3)
    # plot_velocities(X, Y, u, v)

    # Advect the ink
    ink = advect(u_grid, v_grid, 0.1, ink)

    # Step 4: Collision step
    F += (Feq - F)/tau

    # print(f"F after collision: {F[0, 0, :]}")

    # Step 5: Renormalize
    # rho = compute_rho(F)
    # for l in range(Nl):
    #     F[:,:,l] *= rho_0/rho

    # print(f"F after renormalization: {F[0, 0, :]}")

    # Step 5: Streaming step
    for l in range(Nl):
        F = np.roll(F, (cx[l], cy[l]), axis=(0, 1))

    # time.sleep(0.1)
    # print(it)

    print(it)
    if it % 50 == 0:
        print(f"Writing frame {it}")
        array_to_png(ink.quantity.T, f"output/current.png", repeat=1)
