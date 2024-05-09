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
        
        # If values of x_arr, y_arr are close to 0 then clip them to 0
        x_arr = np.where(np.isclose(x_arr, 0), 0, x_arr)
        y_arr = np.where(np.isclose(y_arr, 0), 0, y_arr)
        
        # Wrap x_arr, y_arr 
        x_arr = x_arr % self.x_size
        y_arr = y_arr % self.y_size

        i_arr, j_arr = x_arr.astype(int), y_arr.astype(int)
        i_arr_next, j_arr_next = (i_arr + 1) % self.x_size, (j_arr + 1) % self.y_size
        
        s = x_arr - i_arr
        t = y_arr - j_arr
        
        q00 = self.quantity[i_arr, j_arr]
        q01 = self.quantity[i_arr, j_arr_next]
        q10 = self.quantity[i_arr_next, j_arr]
        q11 = self.quantity[i_arr_next, j_arr_next]
        
        return (1 - s) * (1 - t) * q00 + s * (1 - t) * q10 + (1 - s) * t * q01 + s * t * q11
        
        q = np.zeros(x_arr.shape)
        for i in range(x_arr.shape[0]):
            for j in range(y_arr.shape[1]):
                x, y = x_arr[i, j], y_arr[i, j]
                q[i, j] = self.interpolate(x, y)
        
        # Hopefully a faster way to do this
        # q = np.vectorize(self.interpolate)(x_arr, y_arr)
        
        
        return q
    
    
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
        Wraps the world coordinates to the grid coordinates.
        """
        assert self.periodic
        
        x, y = self.world_to_grid(x, y)
        
        if np.isclose(x, 0):
            x = 0
        if np.isclose(y, 0):
            y = 0
        
        x = x % self.x_size
        y = y % self.y_size
        return x, y
    
        
    def interpolate(self, x, y):
        """
        Bilinear interpolation of the quantity at the given world coordinates.
        The x, y coordinates are in world coordinates.
        """
        x, y = self.project_to_grid(x, y) if not self.periodic else self.wrap_to_grid(x, y)
        # x, y are now grid coordinates
        
        i, j = int(x), int(y)
        i_next, j_next = i + 1, j + 1
        
        if i_next >= self.x_size:
            i_next = i if not self.periodic else 0
        if j_next >= self.y_size:
            j_next = j if not self.periodic else 0
        
        s = x - i
        t = y - j
        
        q00 = self.quantity[i, j]
        q01 = self.quantity[i, j_next]
        q10 = self.quantity[i_next, j]
        q11 = self.quantity[i_next, j_next]
        
        val = (1 - s) * (1 - t) * q00 + s * (1 - t) * q10 + (1 - s) * t * q01 + s * t * q11
        
        return val