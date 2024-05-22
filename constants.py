import numpy as np

# Idk
u_max_upperbound = 0.2

# Velocity directions corresponding to here, right, up, left, down, bottom-right, top-right, top-left, bottom-left.

w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)
Nl = 9
cx_fixed = np.array([0, 1, 0, -1, 0, 1, 1, -1, -1])
cy_fixed = np.array([0, 0, 1, 0, -1, -1, 1, 1, -1])
