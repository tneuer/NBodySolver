#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : Comparator.py
    # Creation Date : Mit 31 Okt 2018 18:56:14 CET
    # Last Modified : Mit 31 Okt 2018 21:35:23 CET
    # Description : Compares different solvers which inherit from NBody_solver
"""
#==============================================================================

import os
import time
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from LeapFrog import N_Body_Gravitation_LF
from RungeKutta2 import N_Body_Gravitation_RK2
from RungeKutta4 import N_Body_Gravitation_RK4

from NBody_solver import N_Body_Gravitationsolver

matplotlib.rcParams["axes.labelsize"] = 40
matplotlib.rcParams["axes.titlesize"] = 20
matplotlib.rcParams["text.color"] = "#AFFFF2"


class Comparator():
    """ Class which makes comparisons between the different solvers.

    The sun_earth.json has to be in the folder to work properly.
    """

    def __init__(self, solvers, dt, steps):
        """ Constructor for the comparator.

        The sun_earth.json has to be in the folder to work properly.

        Arguments
        ---------
        solvers : dict
            Dictionary where keys represent the solvers name and the value is a class
            of this solver. Thos solvers need to inherit from the N_Body_Gravitationsolver base clase.
            An instance of the class is created by the Comparator class itself.
        dt : float
            timesteps used for the algorithm
        steps : int
            Number of steps over which the solvers are compared.
        """
        try:
            self.dt = float(dt)
        except TypeError:
            raise TypeError("'dt' has to be a float value.")

        if not isinstance(steps, int) or steps==0:
            raise TypeError("'step' has to be integer and not 0.")
        else:
            self.steps = steps

        if self.steps>10000:
            self.verbose = True
        else:
            self.verbose = False

        try:
            self.initials = N_Body_Gravitationsolver.read_initials_from_json("./sun_earth.json")
        except FileNotFoundError:
            raise FileNotFoundError("Get the sun_earth.json from me personally!")

        self.solvers = {
                key: solver(dt=self.dt, initials=self.initials, verbose=self.verbose)
                for key, solver in solvers.items()
                }


    def compare(self, show=True, save=True):
        """ Compare the energies and earth trajectories with the given method

        Arguments
        ---------
        show : bool
            If True the show mehtod is called, else only figures and axes are returned
        save : bool or string [True]
            If True the figured is saved as "Energies_method1_method2_..._steps.png" and
            "Trajectories_method1_method2_..._steps.png".
        """
        ####
        # Setup energy plot
        ####
        fig_energies = plt.figure(figsize=(30,20))
        fig_energies.tight_layout()
        ax_energies = plt.gca()
        ax_energies.grid()
        ax_energies.set_xlabel("Time [kDays / Years]")
        ax_energies.set_ylabel("Energy [Joule]")

        ####
        # Setup trajectory plot
        ####
        fig_trajectories = plt.figure(figsize=(30,20)); fig_trajectories.tight_layout()
        fig_trajectories.patch.set_facecolor("k")
        ax_trajectories = plt.gca()
        ax_trajectories.grid()
        ax_trajectories.set_facecolor("k")
        ax_trajectories.axis("off")
        r_max = -np.inf
        names = []
        for name, solver in self.solvers.items():
            names.append(name)
            print("Evolving {}...".format(name))
            startTimer = time.clock()
            positions, _, _, energies, timesteps = solver.evolve(self.steps)
            endTimer = time.clock()

            time_needed = endTimer - startTimer
            hours = int(time_needed/3600)
            time_needed %= 3600
            minutes = int(time_needed/60)
            time_needed %= 60
            seconds = int(time_needed)
            time_needed %= 1
            milliseconds = int(np.round(time_needed*1000))
            time_needed = "{0:2d}:{1:2d}:{2:2d}:{3:3d}".format(hours, minutes, seconds, milliseconds)
            print("Needed {} hh:mm:ss:msmsms".format(time_needed))
            del solvers[name]

            # Plot energies
            energies = energies.sum(axis=0)
            timesteps = np.round(np.array(timesteps)/(60*60*24), 2)
            ax_energies.plot(timesteps, energies, label=name)

            # Plot trajectories
            curr_max = np.max(positions)
            if curr_max > r_max:
                r_max = curr_max

            # Plot earth at index 1
            xs = np.array(positions)[:, 1, 0]
            ys = np.array(positions)[:, 1, 1]
            label = name + "({} h:m:s:ms)".format(time_needed)
            ax_trajectories.plot(xs, ys, linestyle="dashed", label=label, linewidth=5, alpha=0.8)


        # Include initial energy and legend into energy plot
        ax_energies.axhline(y=energies[0], color="red", linestyle="dashed", linewidth=3)
        xticks_values = np.linspace(timesteps[0], timesteps[-1], 10).astype(int)
        xticks_labels = [
                "{} / {}".format(np.round(t/1000, 2), np.round(t/365, 1))
                for t in xticks_values
                ]
        yticks_labels = ["{}".format(e) for e in ax_energies.get_xticks()]
        ax_energies.set_xticks(xticks_values)
        ax_energies.set_xticklabels(xticks_labels, fontsize=15)
        ax_energies.set_yticklabels(yticks_labels, fontsize=20)
        legend = ax_energies.legend(fontsize=30)
        plt.setp(legend.get_texts(), color="k")

        # Plot sun
        ax_trajectories.plot(
                np.array(positions)[:, 0, 0],
                np.array(positions)[:, 0, 1],
                marker="o", markersize=5, alpha=1, color="y"
                )

        # Define axis for trajectories plot
        ax_trajectories.set_xlim(-r_max, r_max)
        ax_trajectories.set_ylim(-r_max, r_max)
        ax_trajectories.legend(loc=2, bbox_to_anchor=(0.72, 1.15), fontsize=20)
        ax_trajectories.set_title(
                    "Days: {} - {} | Years: {} - {} | Stepsize {}".format(
                    timesteps[0], timesteps[-1],
                    np.round(timesteps[0]/365, 2), np.round(timesteps[-1]/365, 2),
                    self.dt),
                    color="#AFFFF2"
                    )
        legend = ax_trajectories.legend(loc=2, bbox_to_anchor=(0.72, 1.15), fontsize=20)
        plt.setp(legend.get_texts(), color="k")

        if save:
            filepath_traj = "Trajectories_{}_{}.png".format("_".join(names), self.steps)
            filepath_ener = "Energies_{}_{}.png".format("_".join(names), self.steps)

            if os.path.exists(filepath_traj):
                os.remove(filepath_traj)
            fig_trajectories.savefig(filepath_traj, facecolor=fig_trajectories.get_facecolor(), bbox_inches="tight")

            if os.path.exists(filepath_ener):
                os.remove(filepath_ener)
            fig_energies.savefig(filepath_ener, bbox_inches="tight")

        if show:
            plt.show()

        return fig_energies, ax_energies


if __name__ == "__main__":
    dt = 60*60*48
    steps = 500000
    solvers = {
            "LeapFrog": N_Body_Gravitation_LF,
            "RK2": N_Body_Gravitation_RK2,
            "RK4": N_Body_Gravitation_RK4
            }

    comparator = Comparator(solvers, dt, steps)

    comparator.compare(show=True, save=True)



