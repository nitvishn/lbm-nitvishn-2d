import numpy as np 

from constants import cx, cy

class SimulationVariables(object):
    def __init__(self, l_ref, u_ref, rho_ref, speed_of_sound, kinematic_viscosity, delta_x):
        self.l_ref = l_ref
        self.u_ref = u_ref
        self.rho_ref = rho_ref
        self.speed_of_sound = speed_of_sound
        self.kinematic_viscosity = kinematic_viscosity
        self.delta_x = delta_x

        self.x = None
        self.y = None
        self.rho = None
        
        self.u = None
        self.v = None

    def set_space(self, x, y, rho):
        self.x = x
        self.y = y
        self.rho = rho

    def set_velocity(self, u, v):
        self.u = u
        self.v = v
        
    def get_tref(self):
        return self.l_ref / self.u_ref
    

    def set_variables(self, lattice_vars: 'LatticeVariables'):
        self.u = lattice_vars.u * self.u_ref / lattice_vars.u_ref
        self.v = lattice_vars.v * self.u_ref / lattice_vars.u_ref
        self.rho = lattice_vars.rho * self.rho_ref
    


class LatticeVariables(object):
    def __init__(self, physical_variables: SimulationVariables) -> None:
        self.speed_of_sound = 1.0 / np.sqrt(3)
        self.u_ref = 0.2 # Stays fixed throughout the simulation, according to LBM paper
        
        self.set_from_physical(physical_variables)


    def set_from_physical(self, physical_variables: SimulationVariables):
        self.kinematic_viscosity = physical_variables.kinematic_viscosity * (self.u_ref / (physical_variables.u_ref * physical_variables.delta_x))

        self.u = physical_variables.u * (self.u_ref / physical_variables.u_ref)
        self.v = physical_variables.v * (self.u_ref / physical_variables.u_ref)
        self.rho = physical_variables.rho / physical_variables.rho_ref

    def recompute_from_distribution(self, F):
        self.rho = np.sum(F, axis=2)
        self.u = np.sum(F * cx, axis=2) / self.rho
        self.v = np.sum(F * cy, axis=2) / self.rho