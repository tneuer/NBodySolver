#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : NBody_solver.py
    # Creation Date : Mit 31 Okt 2018 08:42:46 CET
    # Last Modified : Mit 31 Okt 2018 16:12:26 CET
    # Description : Superclass for all other integrators whic mainly handles initialization.
"""
#==============================================================================

import json

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from collections import deque

class N_Body_Gravitationsolver():
    """ Provides general interface for initialization, plotting and saving data.

    The individual solvers only need to implement the update-step themselves.
    Designed to solve gravitational N-Body systems.

    TODO:
        - Plot option
        - write trajectories constantly to file
    """

    def __init__(self, dt, initials, verbose=True):
        """ Designed to solve gravitational N-Body systems.

        Arguments
        ---------
        dt : float
            Time step per integration
        initials : dict
            dict with the following keys:
                - "r": np.ndarray of dimension (n_bodies, dim)
                - "v": np.ndarray of dimension (n_bodies, dim)
                - "m": masses of the bodies
        verbose: bool
            If true, updates on progress is given.
        """
        if isinstance(initials, str):
            with open(initials, "r") as f:
                initials = json.load(f)

            names = []; masses = []; r_init = []; v_init = []; colors = []; sizes =[]

            for key, value in initials.items():
                names.append(key)
                masses.append(value["mass"])
                r_init.append(value["r_init"])
                v_init.append(value["v_init"])
                colors.append(value["color"])
                powers_comp_to_earth = np.log10(value["mass"]/5.9742e24)
                markersize = powers_comp_to_earth if powers_comp_to_earth>0 else -1/(powers_comp_to_earth-1)
                sizes.append(markersize)

            masses = np.array(masses); r_init = np.array(r_init);
            v_init = np.array(v_init); sizes=np.array(sizes)

            self.initials = {
                    "r_init": r_init,
                    "v_init": v_init,
                    "masses": masses,
                    "colors": colors,
                    "sizes": sizes,
                    "names": names
                    }
        else:
            self.initials = initials

        ####
        # Algorithmic parameters
        ####
        self.dt = dt
        self.pos = self.initials["r_init"]
        self.vel = self.initials["v_init"]
        self.mas = self.initials["masses"]
        self.sizes = self.initials["sizes"]
        self.colors = self.initials["colors"]
        self.names = self.initials["names"]
        self.mass_matrix = self.mas.reshape((1, -1, 1))*self.mas.reshape((-1, 1, 1))

        ####
        # Utility parameters
        ####
        self.G = 6.67428e-11
        self.verbose = verbose
        self.alreadyRun = False
        self.dim = self.pos.shape[1]
        self.n_bodies = self.pos.shape[0]

        ####
        # Assertions
        ####
        assert self.pos.shape == self.vel.shape, (
                "Initial position and velocity must have same dimension.")
        assert self.pos.shape[0] == len(self.mas), (
                "Mass input and positions indicate different number of bodies.")


    def evolve(self, steps=None, t_end=None, saveOnly=None, mass_sun=None):
        """ Evolves the system according to the potential.

        A get_next_steps(self, steps) method has to be implemented, which may use:
            Attributes:
            -----------
            - self.dt
            - self.pos / self.vel / self.mas
            Methods:
            --------
            - get_relative_distances(self):
            - get_next_acc(self)
            - get_kinetic_energy_of_system(self)
            - get_potential_energy_of_system(self)

        It returns (positions, velocities, forces, energies, timesteps)


        Arguments
        ---------
        Either steps XOR t_end is needed.

        steps : int [None]
            Evolve the system for "steps" steps.
        t_end : float [None]
            Evolve the system up to this time.
        saveOnly : int or None [None]
            Saves only the last "saveOnly" body positions, velocities, forces and system
            energy. If None all positions are saved.
        mass_sun : float or None [None]
            Adjust mass of the sun dynamically (in simulations).
            If None the intital value is used.

        Returns
        -------
        self.posCollect : np.ndarray (steps, n_bodies, dim)
            Array of the same shape as initials["r_init"] across a third dimension
            representing time. So each element of the trajectories array is an array
            of shape (n_bodies, dim), indicating the current position of all bodies.
        self.velCollect : np.ndarray (steps, n_bodies, dim)
            Array of the same shape as initials["v_init"] across a third dimension
            representing time. So each element of the trajectories array is an array
            of shape (n_bodies, dim), indicating the current velocity of all bodies.
        self.forces : np.ndarray (steps, n_bodies, dim)
            Array of the same shape as initials["r_init"] across a third dimension
            representing time. So each element of the trajectories array is an array
            of shape (n_bodies, dim), indicating the current force of all bodies.
        energies : np.ndarray (2, steps)
            Indicates the kinetic and potential energy of the system at every timestep.
        self.timesteps : np.ndarray (steps)
            One dimensional array indicating the timestamp of each step.
        """
        self.calledSave = 0
        if steps is None and t_end is not None:
            self.steps = int(np.ceil(t_end/self.dt))
            self.t_end = t_end
        elif t_end is None and steps is not None:
            self.steps = steps
            self.t_end = self.dt*steps
        else:
            raise ValueError("Need either steps OR t_end (excluse or).")

        if not isinstance(saveOnly, int) and saveOnly is not None:
            raise TypeError(
                    "'saveOnly' parameter has to be an integer as it determines"+
                    " the number of saved positions per body. None means all are saved."
                    )
        elif saveOnly == 0:
            raise ValueError(
                    "'saveOnly' is not allowed to be 0. At least 1 position needs to"+
                    " be saved."
                    )
        if self.alreadyRun:
            # Get latest position and velocity as 'intitial value'
            self.pos = self.posCollect[-1]
            self.vel = self.velCollect[-1]

            # Adjust length of saved inforamtions
            if saveOnly != self.posCollect.maxlen:
                self.forces = deque(self.forces, maxlen=saveOnly)
                self.timesteps = deque(self.timesteps, maxlen=saveOnly)
                self.posCollect = deque(self.posCollect, maxlen=saveOnly)
                self.velCollect = deque(self.velCollect, maxlen=saveOnly)
                self.kin_energies = deque(self.kin_energies, maxlen=saveOnly)
                self.pot_energies = deque(self.pot_energies, maxlen=saveOnly)
        else:
            # Initialize distances to calculate energies
            self.disps, self.dists = self.get_relative_distances()

            # Initializing of deques
            self.forces = deque(maxlen=saveOnly)
            self.timesteps = deque([0], maxlen=saveOnly)
            self.posCollect = deque([self.pos], maxlen=saveOnly)
            self.velCollect = deque([self.vel], maxlen=saveOnly)
            self.kin_energies = deque([self.get_kinetic_energy_of_system()], maxlen=saveOnly)
            self.pot_energies = deque([self.get_potential_energy_of_system()], maxlen=saveOnly)

        if mass_sun is not None: # Change mass of sun
            sun_index = np.argmax(self.mas)
            self.mas[sun_index] = mass_sun
            self.mass_matrix = self.mas.reshape((1, -1, 1))*self.mas.reshape((-1, 1, 1))

        # Assert existence of 'get_next_steps'
        get_next_steps = getattr(self, "get_next_steps", None)
        if callable(get_next_steps):
            get_next_steps(steps=steps)
        else:
            raise NotImplementedError(
                    "'get_next_steps(self, steps)' method must be implemented in child.")

        # Assert that save was called after every iteration
        if self.calledSave != steps:
            raise NotImplementedError(
                "Call the function self.save_system_information(positions, velocities)"+
                "after every timestep! Called {} / {} times.".format(self.calledSave, steps)
                )

        self.alreadyRun = True
        self.energies = np.array([np.array(self.kin_energies), np.array(self.pot_energies)])
        return self.posCollect, self.velCollect, self.forces, self.energies, self.timesteps


    def get_relative_distances(self):
        """ Calculate the distances between bodies.

        This is done completely with numpy number crunching methods but is still less
        effective than optimized code, as all distances are calculated and stored twice
        for the distance from $a$ to $b$ and als from $b$ to $a$.

        Returns
        -------
        disps : np.array
            Big object (potential bottleneck) which stores the difference of every
            body to all others coordinatewise. Concepually it looks like this:
                ("Body0": (
                    [0, 0, 0         ] <-- distance to itself
                    [x_01, y_01, z_01]
                    [...,      , ... ]
                    [x_0n, y_0n, z_0n])

                "Body1": (
                    [x_10, y_10, z_10] <-- identical to -1*[x_01, y_01, z_01]
                    [0, 0, 0         ] <-- distance to itself
                    [...,      , ... ]
                    [x_1n, y_1n, z_1n])
                    ...)

                "...": (
                    ...)

                "BodyN": (
                    [x_n0, y_n0, z_n0] <-- identical to -1*[x_0n, y_0n, z_0n]
                    [x_n1, y_n1, z_n1] <-- identical to -1*[x_n1, y_n1, z_n1]
                    [...,      , ... ]
                    [0, 0, 0         ]) <-- distance to itself
                )

            The keys are added for clarity but are not present in this object.

        dists : np.array
            distances between all particles with shape (n_bodies, n_bodies) and diagonal
            elements of np.inf:
                (
                    [np.inf, d_01, d_02, ..., d_0n]
                    [d_10, np.inf, d_12, ..., d_1n]
                    [...                       ...]
                    [d_n0, d_n1, d_n2, ..., np.inf]
                )
        """
        disps = self.pos.reshape((1, -1, self.dim)) - self.pos.reshape((-1, 1, self.dim))
        dists = norm(disps, axis=2)
        dists[dists == 0] = np.inf # Avoid divide by zero warnings
        return disps, dists


    def get_next_acc(self):
        """ Calculate the acceleration at the current position of each particle
        The forces are calculated on the fly if needed and immediately stored.

        Returns
        -------


        """
        forces = self.G*self.disps*self.mass_matrix/np.expand_dims(self.dists, 2)**3
        forces_per_particle = forces.sum(axis=1)
        self.forces.append(forces_per_particle)
        return forces_per_particle/self.mas.reshape(-1, 1)


    def get_kinetic_energy_of_system(self):
        """ Calculate kinetic energies of the system.
        If needed by the user, also individual_kin_energies are calculated on the fly.
        """
        velocities_squared = (self.vel**2).sum(axis=1)
        individual_kin_energies = velocities_squared * self.mas
        return individual_kin_energies.sum()


    def get_potential_energy_of_system(self):
        """ Calculate potential energies of the system.
        If needed by the user, also individual_pot_energies are calculated on the fly.
        """
        individual_pot_energies = -self.mass_matrix/np.expand_dims(self.dists, 2)
        return individual_pot_energies.sum()


    def save_system_information(self, positions, velocities):
        """ Save the information of the bodies and the system as whole.

        Append positions, velocities, energies (of the system) and timesteps
        to the according deques. The forces are added directly in the get_next_acc()
        method. Also count the number of function calls. In the end this has to be
        equal to the number of steps.
        """
        self.calledSave += 1
        self.posCollect.append(positions)
        self.velCollect.append(velocities)
        self.timesteps.append(self.timesteps[-1] + self.dt)
        self.kin_energies.append(self.get_kinetic_energy_of_system())
        self.pot_energies.append(self.get_potential_energy_of_system())


    def plot_trajectories(self, show=True, draw_forces=False, draw_energies=False):
        """ Plots the trajectories of all bodies in the system.

        The length of the drag can be chosen during the "evolve" vall with parameter
        saveOnly.

        Arguments
        ---------
        show : bool [True]
            Enables (disables) plt.show() function.
        draw_forces : List or bool [False]
            If True the forces on each body are shown. Also a list of bodynames is
            possible, so that only for those bodies the forces are drawen.
        draw_energies : bool [False]
            If True, an additional plot with energies is shown. The fig return is then a
            list of plt.figures and the ax return a list of plt.axes

        Returns
        -------
        fig : plt.figure
            matplotlib figure object
        ax : plt.ax
            matplotlib axes object
        """

        r_max = np.max(self.posCollect)
        fig = plt.figure()
        fig.patch.set_facecolor("k")
        ax = plt.axes(xlim=(-r_max, r_max), ylim=(-r_max, r_max))
        ax.set_facecolor("k")
        ax.axis("off")
        if draw_forces:
            if isinstance(draw_forces, list):
                planet_indices = np.where(np.isin(self.names, draw_forces))[0]
                max_force = np.max(np.array(self.forces)[:, planet_indices, :])
            else:
                planet_indices = list(range(self.n_bodies))
                max_force = np.max(self.forces)

        for i in range(self.n_bodies):
            xs = np.array(self.posCollect)[:, i, 0]
            ys = np.array(self.posCollect)[:, i, 1]
            if len(self.timesteps) < 20 or self.names[i] in ["sun", "Sun"]:
                msizes = self.sizes[i] * 3
                plt.plot(xs, ys,
                        color=self.colors[i],
                        marker="o",
                        markersize=msizes,
                        label=self.names[i]
                        )
            else:
                lwidths = self.sizes[i] * 3
                plt.plot(xs, ys,
                        color=self.colors[i],
                        linewidth=lwidths,
                        linestyle="dashed",
                        label=self.names[i]
                        )

            if draw_forces:
                if i in planet_indices:
                    for t in range(len(self.timesteps)):
                        fx = np.array(self.forces)[t, i, 0]
                        fy = np.array(self.forces)[t, i, 1]

                        fx_scaled = fx / max_force * r_max * 0.1
                        fy_scaled = fy / max_force * r_max * 0.1
                        plt.arrow(xs[t], ys[t], fx_scaled, fy_scaled, color="w")

        plt.plot([0], [0], color="w", label="forces", linestyle="solid")
        plt.legend(loc=2, bbox_to_anchor=(0.9, 1.15))

        if draw_energies:
            fig = [fig]; ax = [ax]
            total_energies = self.energies.sum(axis=0)

            f = plt.figure()
            a = plt.gca()
            plt.grid()
            plt.plot(self.timesteps, total_energies, marker="o", linestyle="dashed")

            fig.append(f); ax.append(a)

        if show:
            plt.show()

        return fig, ax


if __name__ == "__main__":
    dt = 60*60*24
    base_solver = N_Body_Gravitationsolver(dt, "./default_initial.json", verbose=False)

    # Has to raise NotImplementedError
    results = base_solver.evolve(steps=365, saveOnly=None)


