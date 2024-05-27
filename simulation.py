from sim_variables import SimulationVariables, LatticeVariables
import numpy as np

from constants import Nl, cx, cy

from lbm_helpers import compute_feq

class Simulation():
    def __init__(self, resolution, physical_vars: SimulationVariables):
        self.resolution = resolution # Assume that this is width 
        self.physical_vars = physical_vars

        self.lattice_vars = LatticeVariables(physical_vars)

        self.F = compute_feq(self.lattice_vars.rho, 
                             self.lattice_vars.u, 
                             self.lattice_vars.v, 
                             self.lattice_vars.speed_of_sound)
        
        self.lattice_vars.recompute_from_distribution(self.F)

        self.C = self.compute_collision_operator()

    
    def step(self):
        # Step 1: Determine whether to change the timescale
        u_max = np.sqrt(np.max(self.lattice_vars.u**2 + self.lattice_vars.v**2)) + 1e-6
        if u_max > 0.22 or u_max < 0.18:

            # F_before_rescale = self.F.copy()
            F_eq_before_rescale = compute_feq(self.lattice_vars.rho, 
                            self.lattice_vars.u, 
                            self.lattice_vars.v, 
                            self.lattice_vars.speed_of_sound)

            self.physical_vars.u_ref = 5 * self.physical_vars.u_ref * u_max
            self.lattice_vars.u *= 0.2 / u_max
            self.lattice_vars.v *= 0.2 / u_max
            
            # Recompute the distribution from the rescaled lattice velocities.
            # Page 12 in the paper talks about rescaling the distribution functions.
            F_eq_after_rescale = compute_feq(self.lattice_vars.rho, 
                            self.lattice_vars.u, 
                            self.lattice_vars.v, 
                            self.lattice_vars.speed_of_sound)
            
            # The below does not seem to work.
            # self.F = F_eq_after_rescale + (F_eq_before_rescale - self.F) * (F_eq_after_rescale / F_eq_before_rescale)

            # This is the normal way to rescale the distribution functions.
            self.F = F_eq_after_rescale 
        
            # This line is pretty much just to recompute 
            # the kinematic viscosity but it also resets u, v, and rho.
            # No matter since we recompute them from the distribution anyway.
            # I know it's confusing. 
            self.lattice_vars.set_from_physical(self.physical_vars) 

            self.C = self.compute_collision_operator() # Since the kinematic viscosity has changed

            self.lattice_vars.recompute_from_distribution(self.F) 

        # Step 2: Collision
        self.F_eq = compute_feq(self.lattice_vars.rho, 
                            self.lattice_vars.u, 
                            self.lattice_vars.v, 
                            self.lattice_vars.speed_of_sound)
        self.F += np.einsum('ij,xyj->xyi', self.C, self.F - self.F_eq)

        # Step 3: Streaming
        for l in range(Nl):
            self.F[:,:,l] = np.roll(self.F[:,:,l], (cx[l], cy[l]), axis=(0, 1))

        # Step 4: Recompute variables for display and next iteration
        self.lattice_vars.recompute_from_distribution(self.F)
        self.physical_vars.set_variables(self.lattice_vars)

        # Return the timestep taken
        dt = self.physical_vars.delta_x * self.lattice_vars.u_ref / self.physical_vars.u_ref

        return dt
    
    def compute_collision_operator(self):
        M = np.array(
            [
                [1.0 for j in range(Nl)], # m0, 1
                [cx[j] for j in range(Nl)], # m1, c_j_x
                [cy[j] for j in range(Nl)], # m2, c_j_y
                [cx[j] ** 2 + cy[j] ** 2 for j in range(Nl)], # m3, c_j_x^2 + c_j_y^2
                [cx[j] ** 2 - cy[j] ** 2 for j in range(Nl)], # m4, c_j_x^2 - c_j_y^2
                [cx[j] * cy[j] for j in range(Nl)], # m5, c_j_x * c_j_y
                [cx[j]**2 * cy[j] for j in range(Nl)], # m6, c_j_x^2 * c_j_y
                [cx[j] * cy[j]**2 for j in range(Nl)], # m7, c_j_x * c_j_y^2
                [cx[j]**2 * cy[j]**2 for j in range(Nl)], # m8, c_j_x^2 * c_j_y^2
            ]
        , dtype=np.float64)

        second_order_relaxation = 1.0 / (3 * self.lattice_vars.kinematic_viscosity + 0.5)
        third_order_relaxation = 0.005

        R = np.diag([0.0, 
                    2.0, 2.0, 
                    second_order_relaxation, second_order_relaxation, second_order_relaxation, 
                    third_order_relaxation, third_order_relaxation, third_order_relaxation])
        R = R.astype(np.float64)
        M_inv = np.linalg.inv(M)

        C = - M_inv @ R @ M

        return C 