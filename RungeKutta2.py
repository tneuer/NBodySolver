#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : RungeKutta2.py
    # Creation Date : Mit 31 Okt 2018 16:13:04 CET
    # Last Modified : Mit 31 Okt 2018 18:40:48 CET
    # Description :
"""
#==============================================================================

import json

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from collections import deque

from NBody_solver import N_Body_Gravitationsolver

np.set_printoptions(precision=16)

class N_Body_Gravitation_RK2(N_Body_Gravitationsolver):
    """ Solver of PDE's via the Runge-Kutta2 method.
    Designed to solve gravitational N-Body systems.
    Many basic functionalities like calculating the distances, accelarations, energies
    are taken from the parent class.

    See https://www.ctcms.nist.gov/~langer/oof2man/RegisteredClass-RK2.html
    """

    def __init__(self, dt, initials, verbose=True):
        N_Body_Gravitationsolver.__init__(self, dt, initials, verbose)

    def get_next_steps(self, steps):
        """ Evolves the system according to the potential with Runge-Kutta 2 method.

        This function should not be called directly but use:

            _instancename_.evolve(self, steps=None, t_end=None, saveOnly=None, mass_sun=None)

        from parent class N_Body_Gravitationsolver.

        Arguments
        ---------
        steps : int [None]
            Evolve the system for "steps" steps.

        Returns
        -------
        None
            Body is save by calling the get_next_acc() and save_system_information()
            methods.
        """
        for step in range(steps):
            # Actual calulation: Runge-Kutta 2

            # Step 1
            k1 = [self.vel, self.get_next_acc()]

            # Step 2
            next_pos = self.pos + k1[0] * self.dt * 0.5
            next_vel = self.vel + k1[1] * self.dt * 0.5
            self.disps, self.dists = self.get_relative_distances(positions=next_pos)
            k2 = [
                next_vel,
                self.get_next_acc()
                ]

            self.pos = self.pos + k2[0] * self.dt
            self.vel = self.vel + k2[1] * self.dt

            # Saving of statistics
            self.save_system_information(self.pos, self.vel)



if __name__ == "__main__":
    dt = 60*60*24
    RK2 = N_Body_Gravitation_RK2(dt, "./default_initial_short.json", verbose=True)

    results = RK2.evolve(steps=1000, saveOnly=None)

    RK2.plot_trajectories(draw_forces=False, draw_energies=False, show=True)





