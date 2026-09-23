# Assignment 2

## Running

```bash
$ uv run assignment_2.py --help
usage: assignment_2.py [-h] {trajectory,roa,return-map,lookup-table} ...

Simulate the inverted pendulum walker

positional arguments:
  {trajectory,roa,return-map,lookup-table}
    trajectory          animate a trajectory and plot its energy
    roa                 plot the upright controller's region of attraction
    return-map          plot the theta=0 Poincare return map
    lookup-table        compute minimum steps to the ankle controller's RoA and save the optimal alphas

options:
  -h, --help            show this help message and exit
```

You can configure parameters for each of these subcommands in `assignment_2.py`.
If you want to run the resulting controller, first run

```bash
uv run assignment_2.py roa
uv run assignment_2.py lookup-table
```

And then run

```bash
uv run assignment_2.py trajectory output/assignment_2/policy_map.npy
```

### Using an accelerator

The code is written in JAX, so you may use an accelerator if you wish. You can run

```bash
uv sync --extra gpu
```

To install JAX with CUDA 13 -- JAX will then use your GPU by default. However, I
have made sure that all subprograms run in a reasonable time (at most a few minutes
on my machine) with just the CPU, so this is not necessary to get the results in
this write up.

## Sketches

## RoA of ankle controller

The ankle torque is controlled via feedback linearization. We apply torque to
remove the effects of gravity and pendulum damping, and then add the same dynamics
with _inverted_ gravity so the pendulum behaves as if it is hanging down.

![The ankle torque controller's region of attraction.](assignment_2_results/upright_roa.png)
The ankle torque controller's region of attraction. Blue states are states which
the controller could stabilize the pendulum. We sweep
$\theta \in [-\frac{\pi}{2} + \gamma, \alpha + \gamma]$.
$\alpha = \frac{\pi}{8}$ for this plot. We terminate the trajectory immediately
if walker collides in either direction, as we are only interested in the RoA
within one step here. Note that the event dynamics do not handle collisions
backwards: we just manually check if the angle is beyond the backwards collision
bound.

## Poincaré section

I chose $\theta = 0$ as my Poincaré section. It is constant (unlike the
collision angle), allows us to use only $\dot{\theta}$ as the return map state
(since $\theta$ is fixed), and is transverse to all flows except those where
the pendulum has _just_ enough energy to stop at upright unstable equilibrium
(the blue highlighted ones in the below image). We do not really care about these
orbits, however, as they will stabilize anyways.
![Flow of uncontrolled inverted pendulum](assignment_2_results/orbits.png)

## Grid resolution

For the control lookup table grid, let's first see what happens with a very low
resolution. A good way to check how well we are doing is to compare what we
_think_ will happen based on our discretized state grid and transitions to what
actually happens. If we use just 4 uniform alpha values and 10 uniform $\dot{\theta}$
values, we get this plot:
![Plot](assignment_2_results/steps_to_stability_10_4.png)

The first plot shows how many steps we expect each of the 10 possible start states
to take based only on the state transition graph. If the resolution is too low,
we may expect the number of steps in this graph to sometimes be wrong. For example,
if a given initial velocity $\dot{\theta}_0$ ended up at $\dot{\theta}_1$ the next
time it hit the Poincaré section, but we rounded it down to the closest state
$\dot{\theta}' < \dot{\theta}_1$, we may think that we can stop faster than we can
in reality. The second plot is what _actually_ happens if we roll out the controller,
and it is much higher resolution than the first plot (though this is hard to show).
In this case, we are rolling out 2000 initial velocities spaced evenly. We can see
that we had to take _more_ steps than we predicted (for example, at the first
light green band) in some cases. This implies our $\dot{\theta}$ resolution is
too low.

Indeed, if we increase our resolution to 40, these issues seem to mostly disappear.
![Plot](assignment_2_results/steps_to_stability_40_4.png)

But at 35, they are still there.
![Plot](assignment_2_results/steps_to_stability_35_4.png)

What about the resolution of alpha? This surprisingly doesn't seem to matter
_too_ much, but choosing an extremely low value (like restricting the controller
to only 2 alphas but still with 40 $\dot{\theta}$) can still cause problems.
![Plot](assignment_2_results/steps_to_stability_40_2.png)
This is probably because the alpha selected for a given discretized $\dot{\theta}$
worked for that particular $\dot{\theta}$ but may not work for every continuous
state that corresponds to the discretized state. When choosing the $\alpha$ to use,
we intentionally select the median of the $\alpha$s that "work" (move us to a state
with the lowest possible already computed number of steps to convergence). This
gives us some robustness to this issue as the median $\alpha$ should somewhat work
for $\dot{\theta}$ higher or lower than the representative, but if there are very
few options for $\alpha$ that work then this is not very effective, and that is what
we are seeing here.

## Trajectory plot
