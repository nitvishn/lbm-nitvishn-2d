import numpy as np 

# Import uref_1 from output/uref_1.npy
uref_1 = np.load("output/uref_1.0.npy")
uref_09 = np.load("output/uref_0.9.npy")
uref_11 = np.load("output/uref_1.1.npy")

# Cut them to the same length
min_length = min(len(uref_1), len(uref_09), len(uref_11))
uref_1 = uref_1[:min_length]
uref_09 = uref_09[:min_length]
uref_11 = uref_11[:min_length]

dt_1 = 1/uref_1 
dt_09 = 1/uref_09
dt_11 = 1/uref_11

# Plot dt_1, dt_09, dt_11
import matplotlib.pyplot as plt
plt.plot(dt_1, label="rho_0 = 1.0")
plt.plot(dt_09, label="rho_0 = 0.9")
plt.plot(dt_11, label="rho_0 = 1.1")
plt.legend()
plt.xlabel("Frame Number")

# Log scale
plt.yscale("log")

plt.ylabel("Time Step Size (log)")

plt.show()