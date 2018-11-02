# ODESolver

### Description

This is a full set of solvers for the gravitational N-Body problems, solved for small systems with the typical Integrators Leapfrog, Runge-Kutta 2 and Runge-Kutta 4.

* The NBody_Solver.py file implements a parent class for all the others, which handles initialization (read-in) of the initial values, the time step and the reuse of older initial values . The children need to implement a get_next_steps(self, steps) method, where positions and velocities are calculated. In the for-loop the save_system_information(self, positions, velocities) needs to be called after every iteration.

* The LeapFrog.py module implements the leapfrog method solver.

* The RungeKutta2.py module implements the Runge-Kutta 2 solver.

* The RungeKutta4.py module implements the Runge-Kutta 4 solver.

  ```python
  if __name__ == "__main__":
      dt = 60*60*24
      RK4 = N_Body_Gravitation_RK4(dt, "./default_initial_short.json", verbose=True)
      results = RK4.evolve(steps=365, saveOnly=270)
      RK4.plot_trajectories(draw_forces=True, draw_energies=False, show=True, save=True)
  ```

  * The saveOnly options declares how many positions are saved

    ![Force RK4](https://raw.githubusercontent.com/tneuer/NBodySolver/master/Figures/Trajectories_N_Body_Gravitation_RK4_365.png)

* The Comparator.py module implements a class which compares the performance of the solvers in terms of energy and runtime. As input it takes a dictonary of solvers, the timestep and the number of steps.
  * The compare method evaluates the solvers for the given steps
  * Planet Earth is shown as comparison planet for the trajectories
  * The results can be directly saved



![Energies](https://raw.githubusercontent.com/tneuer/NBodySolver/master/Figures/Energies_500k.png)



![Trajectories](https://raw.githubusercontent.com/tneuer/NBodySolver/master/Figures/Trajectories_500k.png)



#### Features

* Variable drag
* Variable mass of central star
* Automaggtic recognition of unused dimensions



#### Dependencies

For the normal solvers:

- json

```bash
# Color package
>> pip install color
```



For the Dash simulation

- Dash

``` bash
# Dash (from https://dash.plot.ly/installation)
>> pip install dash==0.28.5  # The core dash backend
>> pip install dash-html-components==0.13.2  # HTML components
>> pip install dash-core-components==0.35.1  # Supercharged components
```





##### TODO

- ~~Implements logartihmic mass scaling for point size~~
- ~~Dark background colour~~
- ~~Change mass of sun~~
- ~~planets get smaller with bigger scale~~
- ~~refactor with Superclass ODESOLVER~~
- Comparison to true values in sun-earth system --> error calculation
- Focus coordinatesystem on heaviest object
- Different Solver in one plot or all in different
- ~~Timestep input~~
- ~~Automatic update to two body dimension~~
- ~~show planet position before adding~~
- draw every quarter year where earth should be
- Restart message 
- body error message
- No fail if body is added during runtime