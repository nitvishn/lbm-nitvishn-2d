import numpy as np

# Velocity directions corresponding to here, right, up, left, down, bottom-right, top-right, top-left, bottom-left.

w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])  # weights
Nl = 9
cx = np.array([0, 1, 0, -1, 0, 1, 1, -1, -1])
cy = np.array([0, 0, 1, 0, -1, -1, 1, 1, -1])
