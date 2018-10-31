#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : LeapFrog.py
    # Creation Date : Son 28 Okt 2018 15:41:12 CET
    # Last Modified : Mit 31 Okt 2018 16:11:25 CET
    # Description : Implement leapfrog algorithm
"""
#==============================================================================

import json

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from collections import deque

from NBody_solver import N_Body_Gravitationsolver

np.set_printoptions(precision=16)

class N_Body_Gravitation_LF(N_Body_Gravitationsolver):
    """ Solver of PDE's via the LeapFrog method.
    Designed to solve gravitational N-Body systems.
    Many basic functionalities like calculating the distances, accelarations, energies
    are taken from the parent class.

    See https://en.wikipedia.org/wiki/Leapfrog_integration
    """

    def __init__(self, dt, initials, verbose=True):
        N_Body_Gravitationsolver.__init__(self, dt, initials, verbose)

    def get_next_steps(self, steps):
        """ Evolves the system according to the potential with leapfrog method.

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
        # Leapfrog specific initialisation
        if not self.alreadyRun:
            self.acc = self.get_next_acc()
            self.vel = self.vel + 0.5 * self.acc * self.dt

        for step in range(steps):
            # Actual calulation: Leapfrog
            self.pos = self.pos + self.vel * self.dt
            self.disps, self.dists = self.get_relative_distances()
            self.acc = self.get_next_acc()
            self.vel = self.vel + self.acc * self.dt

            # Saving of statistics
            self.save_system_information(self.pos, self.vel)

            if self.verbose and step % 500 == 0:
                print(step, "/", steps)



if __name__ == "__main__":
    dt = 60*60*24
    leapfrog = N_Body_Gravitation_LF(dt, "./default_initial_short.json", verbose=True)

    results = leapfrog.evolve(steps=1000, saveOnly=None)

    leapfrog.plot_trajectories(draw_forces=True, draw_energies=True, show=True)

