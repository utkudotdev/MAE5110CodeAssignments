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
