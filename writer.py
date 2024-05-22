import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

def plot_velocities(X, Y, u, v):
    Nx, Ny = u.shape

    max_u = 2.0 + np.max(np.abs(u))
    max_v = 2.0 + np.max(np.abs(v))

    # Plot the velocity field as a quiver plot
    plt.figure()
    plt.quiver(X, Y, u/max_u, v/max_v, scale=1, scale_units='xy')
    # Equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def array_to_png(data, output_filename, repeat=1, low=0.0, high=1.0):

    # Normalize the data array to be between 0 and 1
    divby = high - low
    data = (data - low) / (high - low)

    # Repeat the data array in both dimensions
    new_data = np.zeros((data.shape[0] * repeat, data.shape[1] * repeat))
    nx, ny = data.shape
    for i in range(repeat):
        for j in range(repeat):
            new_data[i*nx:(i+1)*nx, j*ny:(j+1)*ny] = data
    
    data = new_data

    data = np.flipud(data)  # Flip the data array vertically to match the PNG format
    # Define the color mapping from 0 (white) to 1 (dark ink blue)
    color_0 = np.array([114, 27, 228])  # RGB for 0
    color_1 = np.array([141, 228, 27])  # RGB for 1
    
    color_0_arr = np.tile(color_0, (data.shape[0], data.shape[1], 1))
    color_1_arr = np.tile(color_1, (data.shape[0], data.shape[1], 1))
    colors = color_0_arr * (1 - data[:, :, None]) + color_1_arr * data[:, :, None]

    # Create an RGB image where each pixel is determined by the interpolation of white and dark ink blue
    height, width = data.shape
    image = Image.new("RGB", (width, height))
    image.putdata([tuple(colors[i, j, :].astype(int)) for i in range(height) for j in range(width)])

    # Save the image
    image.save(output_filename)