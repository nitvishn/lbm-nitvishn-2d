import numpy as np


class GridQuantity:

    def __init__(self, x_size, y_size, stagger_x: float, stagger_y: float, periodic=False):
        # stagger_x and stagger_y are offsets such that
        # grid index (i, j) corresponds to the world coordinate (i + stagger_x, j + stagger_y)
        # example: stagger_x = -0.5, stagger_y = 0 corresponds to the u-velocity grid
        self.x_size = x_size
        self.y_size = y_size
        self.quantity = np.zeros((x_size, y_size), dtype=np.float64)
        self.stagger_x = stagger_x
        self.stagger_y = stagger_y
        self.periodic = periodic

    def sample(self, x_arr: np.array, y_arr: np.array):
        """
        Interpolates the quantity at the given world coordinates.
        The x, y coordinates are a meshgrid of world coordinates.
        """
        x_arr, y_arr = self.world_to_grid(x_arr, y_arr)
        # x_arr, y_arr are now grid coordinates

        x_arr, y_arr = self.wrap_to_grid(x_arr, y_arr)

        i_arr, j_arr = x_arr.astype(int), y_arr.astype(int)
        i_arr_next, j_arr_next = (i_arr + 1) % self.x_size, (j_arr + 1) % self.y_size

        s = x_arr - i_arr
        t = y_arr - j_arr
        # s, t = 0.5, 0.5

        q00 = self.quantity[i_arr, j_arr]
        q01 = self.quantity[i_arr, j_arr_next]
        q10 = self.quantity[i_arr_next, j_arr]
        q11 = self.quantity[i_arr_next, j_arr_next]

        return (1 - s) * (1 - t) * q00 + s * (1 - t) * q10 + (1 - s) * t * q01 + s * t * q11

    def grid_to_world(self, i, j):
        """
        Converts the grid coordinates to world coordinates.
        """
        return i + self.stagger_x, j + self.stagger_y

    def world_to_grid(self, x, y):
        """
        Converts the world coordinates to grid coordinates.
        """
        return x - self.stagger_x, y - self.stagger_y

    def project_to_grid(self, x, y):
        """
        Projects the world coordinates to the nearest grid coordinates.
        The grid coordinates are NOT necessarily integers.
        """
        assert not self.periodic

        x, y = self.world_to_grid(x, y)
        x = np.clip(x, 0, self.x_size - 1)
        y = np.clip(y, 0, self.y_size - 1)
        return x, y

    def wrap_to_grid(self, x, y):
        """
        Wraps the given GRID coordinates to the grid coordinates.
        """
        assert self.periodic

        # If values of x_arr, y_arr are close to 0 then clip them to 0
        x_arr = np.where(np.isclose(x, 0), 0, x)
        y_arr = np.where(np.isclose(y, 0), 0, y)

        # Wrap x_arr, y_arr 
        x_arr = x_arr % self.x_size
        y_arr = y_arr % self.y_size

        return x_arr, y_arr
    
    def rotated(self, angle: np.float64, center_x: np.float64, center_y: np.float64):
        """
        Returns the rotated grid quantity by the given angle around the given center in world coordinates.
        """
        if self.stagger_x != 0 or self.stagger_y != 0:
            raise NotImplementedError("Rotating staggered grids is not implemented.")

        new_quantity = GridQuantity(self.x_size, self.y_size, self.stagger_x, self.stagger_y, self.periodic)

        x_arr, y_arr = np.meshgrid(range(self.x_size), range(self.y_size), indexing='ij') # The new quantity coordinates
        x_arr = x_arr.astype(np.float64)
        y_arr = y_arr.astype(np.float64)

        # Want to move the new quantity coordinates to the old quantity coordinates
        x_arr -= center_x
        y_arr -= center_y

        x_arr_rot = np.cos(-angle) * x_arr - np.sin(-angle) * y_arr
        y_arr_rot = np.sin(-angle) * x_arr + np.cos(-angle) * y_arr

        x_arr_rot += center_x
        y_arr_rot += center_y

        # Now it's in the old quantity coordinates
        new_quantity.quantity = self.sample(x_arr_rot, y_arr_rot)

        return new_quantity
        


def advect(new_q: GridQuantity, u: GridQuantity, v: GridQuantity, dt: float, q: GridQuantity):
    """
    Advects the quantity q using the velocities u and v.
    Semi-Lagrangian advection is used, which means that the
    velocity field is used to trace back in time where the
    quantity q came from.

    STORES THE RESULT IN new_q.
    """

    x_arr, y_arr = np.meshgrid(range(q.x_size), range(q.y_size), indexing='ij')
    x_arr, y_arr = q.grid_to_world(x_arr, y_arr)

    x_mid_arr, y_mid_arr = x_arr - 0.5 * dt * u.sample(x_arr, y_arr), y_arr - 0.5 * dt * v.sample(x_arr, y_arr)

    x_p_arr, y_p_arr = x_arr - dt * u.sample(x_mid_arr, y_mid_arr), y_arr - dt * v.sample(x_mid_arr, y_mid_arr)

    new_q.quantity = q.sample(x_p_arr, y_p_arr)



if __name__ == "__main__":
    from writer import array_to_png

    # Test the rotate function
    nx, ny = 100, 100

    q = GridQuantity(nx, ny, 0, 0, periodic=True)
    x, y = np.meshgrid(range(nx), range(ny), indexing='ij')

    width = 0.3

    y_norm = y / ny
    x_norm = x / nx
    layer = np.logical_and(0.5 - width / 2 <= x_norm, x_norm < 0.5 + width / 2)
    q.quantity[layer] = 1.0

    angle = np.pi / 4

    array_to_png(q.quantity, "output/rotated_before.png", repeat=3)

    q_rotated = q.rotated(angle, nx / 2, ny / 2)

    array_to_png(q_rotated.quantity, "output/rotated_after.png", repeat=3)