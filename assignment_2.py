import functools as ft
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from integrators import rk4
from models import inverted_pendulum_walker as model

# Fixed controls for this visualization example.
params = {
    "gravity": 9.81,  # m/s^2
    "length": 1.0,  # m
    "mass": 1.0,  # kg
    "incline": 0.06,  # rad
    "angle_of_attack": np.pi / 8,  # rad
    "ankle_torque": 0.0,  # N m
    "ankle_torque_damping": 5.0,
}

initial_state = np.array([0.0, 1.0])
timestep = 1e-4
sim_time = 3.0
num_timesteps = round(sim_time / timestep)


def ankle_torque_bounds(params):
    m = params["mass"]
    g = params["gravity"]
    l = params["length"]
    return -0.1 * m * g * l, 0.05 * m * g * l


def compute_ankle_torque(state, params):
    m = params["mass"]
    g = params["gravity"]
    l = params["length"]
    b = params["ankle_torque_damping"]
    theta, theta_dot = state

    control = -2.0 * m * g * l * jnp.sin(theta) - b * theta_dot
    lower, upper = ankle_torque_bounds(params)
    clipped = jnp.clip(control, lower, upper)

    return clipped


@ft.partial(jax.jit, static_argnames=["num_timesteps"])
def simulate(initial_state, timestep, num_timesteps, params):
    """Simulates the systems starting from `initial_state` for `num_timesteps`
    steps of length `timestep`. Returns `(time_traj, state_traj, impacts)`.

    `time_traj` is of length `num_timesteps + 1` and indicates the time at which
    each state in `state_traj` occurred. `state_traj` is of shape (2, `num_timesteps + 1`),
    one row for theta and another for theta dot. `impacts` is of length `num_timesteps + 1`
    and indicates timesteps at which the event guard triggered (the corresponding state is
    *after* the event dynamics have been applied).
    """

    def step(carry, _):
        t, state = carry

        ankle_torque = compute_ankle_torque(state, params)
        params["ankle_torque"] = ankle_torque

        f = ft.partial(model.dynamics, params=params)
        next_state = rk4(f, t, state, timestep)

        event_guard_hit = model.event_guard(state, next_state, params)
        next_state = jax.lax.cond(
            event_guard_hit,
            lambda: model.event_dynamics(next_state, params),
            lambda: next_state,
        )

        next_t = t + timestep

        return (next_t, next_state), (next_t, next_state, event_guard_hit)

    _, (time_traj, state_traj, impacts) = jax.lax.scan(
        step, (0.0, initial_state), length=num_timesteps
    )

    time_traj = jnp.concatenate([jnp.array([0.0]), time_traj])
    state_traj = jnp.concatenate([initial_state.reshape((-1, 1)), state_traj.T], axis=1)
    impacts = jnp.concatenate([jnp.array([False]), impacts])

    return time_traj, state_traj, impacts


def calculate_absolute_energy(state_traj, impacts, params):
    """Return kinetic and potential energy in a fixed global reference frame."""
    gravity = params["gravity"]
    mass = params["mass"]
    length = params["length"]
    incline = params["incline"]
    angle_of_attack = params["angle_of_attack"]

    kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
    theta = state_traj[0]

    step_height = 2 * length * jnp.sin(angle_of_attack) * jnp.sin(incline)
    stance_height_changes = jnp.where(
        impacts,
        jnp.where(theta < incline, -step_height, step_height),
        0.0,
    )
    stance_height = jnp.cumsum(stance_height_changes)
    potential_energy = potential_energy + mass * gravity * stance_height

    return kinetic_energy, potential_energy


def plot_energy(time_traj, state_traj, impacts, params):
    kinetic_energy, potential_energy = calculate_absolute_energy(
        state_traj, impacts, params
    )

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(time_traj, kinetic_energy, label="Kinetic energy")
    ax.plot(time_traj, potential_energy, label="Potential energy")
    ax.plot(time_traj, kinetic_energy + potential_energy, label="Total energy")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("Absolute Energy")
    ax.grid(alpha=0.25)
    ax.legend()

    return fig


time_traj, state_traj, impacts = simulate(
    initial_state, timestep, num_timesteps, params
)

fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact.
    model.visualize(state_traj[:, index], params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")


# Simulate at a small timestep, but render only 25 frames per second.
fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

animation = FuncAnimation(
    fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
)
output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)
animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))
energy_fig = plot_energy(time_traj, state_traj, impacts, params)
energy_fig.savefig(output / "energy.png")

# To save an MP4 instead, install FFmpeg and use:
# animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
print(f"Saved {output / 'walker.gif'} and {output / 'energy.png'}.")
plt.show()
