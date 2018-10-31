#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : RungeKutta4.py
    # Creation Date : Mit 31 Okt 2018 18:42:26 CET
    # Last Modified : Mit 31 Okt 2018 22:44:42 CET
    # Description : Implementation of Runge-Kutta 4
"""
#==============================================================================

import json

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from collections import deque

from NBody_solver import N_Body_Gravitationsolver

np.set_printoptions(precision=16)

class N_Body_Gravitation_RK4(N_Body_Gravitationsolver):
    """ Solver of PDE's via the Runge-Kutta4 method.
    Designed to solve gravitational N-Body systems.
    Many basic functionalities like calculating the distances, accelarations, energies
    are taken from the parent class.

    https://www.ctcms.nist.gov/~langer/oof2man/RegisteredClass-RK4.html
    """

    def __init__(self, dt, initials, verbose=True):
        N_Body_Gravitationsolver.__init__(self, dt, initials, verbose)

    def get_next_steps(self, steps):
        """ Evolves the system according to the potential with Runge-Kutta 4 method.

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
            k1 = [
                self.vel * self.dt,
                self.get_next_acc() * self.dt
                ]

            # Step 2
            next_pos = self.pos + k1[0] * 0.5
            next_vel = self.vel + k1[1] * 0.5
            self.disps, self.dists = self.get_relative_distances(positions=next_pos)
            k2 = [
                next_vel * self.dt,
                self.get_next_acc(save=False) * self.dt
                ]

            # Step 3
            next_pos = self.pos + k2[0] * 0.5
            next_vel = self.vel + k2[1] * 0.5
            self.disps, self.dists = self.get_relative_distances(positions=next_pos)
            k3 = [
                next_vel * self.dt,
                self.get_next_acc(save=False) * self.dt
                ]

            # Step 4
            next_pos = self.pos + k3[0]
            next_vel = self.vel + k3[1]
            self.disps, self.dists = self.get_relative_distances(positions=next_pos)
            k4 = [
                next_vel * self.dt,
                self.get_next_acc(save=False) * self.dt
                ]

            # Move forward
            self.pos = self.pos + 1/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            self.vel = self.vel + 1/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])

            # Saving of statistics
            self.save_system_information(self.pos, self.vel)



if __name__ == "__main__":
    dt = 60*60*24
    RK4 = N_Body_Gravitation_RK4(dt, "./default_initial_short.json", verbose=True)
    results = RK4.evolve(steps=365, saveOnly=270)
    figs = RK4.plot_trajectories(draw_forces=True, draw_energies=False, show=True, save=True)
