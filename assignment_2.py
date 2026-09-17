import argparse
import enum
import functools as ft
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from integrators import rk4
from models import inverted_pendulum_walker as model


class RoAClassification(enum.IntEnum):
    OUTSIDE = 0
    STABILIZABLE = 1


@jax.tree_util.register_dataclass
@dataclass
class RoAResult:
    theta_values: jax.Array
    theta_dot_values: jax.Array
    classification_grid: jax.Array


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


def get_timestep_for_state(state, params, large_timestep, small_timestep):
    alpha = params["angle_of_attack"]
    gamma = params["incline"]
    upper_limit = gamma + alpha
    lower_limit = jnp.deg2rad(-90.0) + gamma
    total_range = upper_limit - lower_limit

    theta, theta_dot = state
    distance = jnp.where(
        theta_dot >= 0,
        upper_limit - theta,
        theta - lower_limit,
    )
    progress = (total_range - distance) / total_range
    return progress * small_timestep + (1 - progress) * large_timestep


@ft.partial(jax.jit, static_argnames=["num_timesteps"])
def simulate(initial_state, timestep, num_timesteps, params):
    """Simulates the systems starting from `initial_state` for `num_timesteps`
    steps of length `timestep`. Returns `(time_traj, state_traj, impacts)`.

    `time_traj` is of length `num_timesteps + 1` and indicates the time at which
    each state in `state_traj` occurred. `state_traj` is of shape (2, `num_timesteps + 1`),
    one row for theta and another for theta dot. `impacts` is of length `num_timesteps + 1`
    and contains the collision type at each timestep: -1 for backward, 0 for none,
    and 1 for forward. A state at an impact is recorded *after* event dynamics.
    """

    def step(carry, _):
        t, state = carry

        ankle_torque = compute_ankle_torque(state, params)
        params["ankle_torque"] = ankle_torque

        f = ft.partial(model.dynamics, params=params)
        next_state = rk4(f, t, state, timestep)

        collision_type = model.event_guard(state, next_state, params)
        next_state = jax.lax.cond(
            collision_type != model.NO_COLLISION,
            lambda: model.event_dynamics(next_state, collision_type, params),
            lambda: next_state,
        )

        next_t = t + timestep

        return (next_t, next_state), (next_t, next_state, collision_type)

    _, (time_traj, state_traj, impacts) = jax.lax.scan(
        step, (0.0, initial_state), length=num_timesteps
    )

    time_traj = jnp.concatenate([jnp.array([0.0]), time_traj])
    state_traj = jnp.concatenate([initial_state.reshape((-1, 1)), state_traj.T], axis=1)
    impacts = jnp.concatenate([jnp.array([model.NO_COLLISION]), impacts])

    return time_traj, state_traj, impacts


def calculate_absolute_energy(state_traj, impacts, params):
    """Return kinetic and potential energy in a fixed global reference frame."""
    gravity = params["gravity"]
    mass = params["mass"]
    length = params["length"]
    incline = params["incline"]
    angle_of_attack = params["angle_of_attack"]

    kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
    step_height = 2 * length * jnp.sin(angle_of_attack) * jnp.sin(incline)
    stance_height_changes = (impacts == model.FORWARD_COLLISION) * step_height
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


@jax.jit
def find_upright_roa(params, large_timestep, small_timestep, sim_time):
    gamma = params["incline"]
    alpha = params["angle_of_attack"]

    THETA_MIN, THETA_MAX = jnp.deg2rad(-90.0) + gamma, gamma + alpha
    NUM_THETAS = 200
    THETA_DOT_MIN, THETA_DOT_MAX = jnp.deg2rad(-100.0), jnp.deg2rad(300.0)
    NUM_THETA_DOTS = 200

    thetas = jnp.linspace(THETA_MIN, THETA_MAX, NUM_THETAS)
    theta_dots = jnp.linspace(THETA_DOT_MIN, THETA_DOT_MAX, NUM_THETA_DOTS)

    all_initial_states = jnp.stack(
        jnp.meshgrid(thetas, theta_dots, indexing="ij"), axis=-1
    )
    flat_initial_states = all_initial_states.reshape((-1, 2))

    stable = jax.vmap(can_stabilize, in_axes=(0, None, None, None, None))(
        flat_initial_states,
        params,
        large_timestep,
        small_timestep,
        sim_time,
    )
    classification_grid = stable.reshape((NUM_THETAS, NUM_THETA_DOTS)).astype(int)

    return RoAResult(
        theta_values=thetas,
        theta_dot_values=theta_dots,
        classification_grid=classification_grid,
    )


def plot_upright_roa(result: RoAResult):
    colors = {
        RoAClassification.OUTSIDE: "#e63946",
        RoAClassification.STABILIZABLE: "#2a9d8f",
    }
    labels = {
        RoAClassification.OUTSIDE: "Outside RoA",
        RoAClassification.STABILIZABLE: "Stabilizable",
    }

    color_map = ListedColormap(
        [colors[classification] for classification in RoAClassification]
    )
    color_norm = BoundaryNorm(np.arange(len(RoAClassification) + 1) - 0.5, color_map.N)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.pcolormesh(
        np.rad2deg(result.theta_values),
        np.rad2deg(result.theta_dot_values),
        result.classification_grid.T,
        cmap=color_map,
        norm=color_norm,
        shading="nearest",
    )

    legend_handles = [
        Patch(color=colors[classification], label=labels[classification])
        for classification in RoAClassification
        if np.any(result.classification_grid == classification.value)
    ]
    ax.legend(
        handles=legend_handles,
        title="Outcome",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    ax.set_title("Upright Controller Region of Attraction")
    ax.set_xlabel(r"Initial angle $\theta$ (deg)")
    ax.set_ylabel(r"Initial angular velocity $\dot{\theta}$ (deg/s)")
    ax.grid(alpha=0.25)

    return fig


def can_stabilize(
    initial_state, params, large_timestep, small_timestep, sim_time
) -> jax.Array:
    def wrap_angle(theta):
        return jnp.abs((theta + jnp.pi) % (2 * jnp.pi) - jnp.pi)

    def fell_over(state):
        theta, _ = state
        return wrap_angle(theta) > jnp.deg2rad(89)

    def stabilized(state):
        theta, theta_dot = state
        return (wrap_angle(theta) < np.deg2rad(1)) & (
            jnp.abs(theta_dot) < np.deg2rad(0.1)
        )

    def step(val):
        state, time, _, _ = val

        timestep = get_timestep_for_state(state, params, large_timestep, small_timestep)
        timestep = jnp.minimum(timestep, sim_time - time)

        ankle_torque = compute_ankle_torque(state, params)
        controlled_params = {**params, "ankle_torque": ankle_torque}

        f = ft.partial(model.dynamics, params=controlled_params)
        next_state = rk4(f, time, state, timestep)

        collision_type = model.event_guard(state, next_state, controlled_params)
        next_state = jax.lax.cond(
            collision_type != model.NO_COLLISION,
            lambda: model.event_dynamics(next_state, collision_type, controlled_params),
            lambda: next_state,
        )

        stable = stabilized(next_state)
        collision_occurred = collision_type != model.NO_COLLISION
        next_time = time + timestep
        done = (
            fell_over(next_state)
            | stable
            | collision_occurred
            | (next_time >= sim_time)
        )

        return next_state, next_time, stable, done

    _, _, stable, _ = jax.lax.while_loop(
        lambda t: ~t[3],
        step,
        (initial_state, 0.0, jnp.array(False), jnp.array(False)),
    )

    return stable


def create_walker_animation(time_traj, state_traj, params, timestep, fps=25):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    def draw_frame(index):
        # The massless swing leg is repositioned instantaneously at each impact.
        model.visualize(state_traj[:, index], params, ax=ax)
        ax.set_title(f"t = {time_traj[index]:.2f} s")

    # Simulate at a small timestep, but render only at the requested frame rate.
    frame_stride = round(1 / (fps * timestep))
    frame_indices = list(range(0, time_traj.size, frame_stride))
    if frame_indices[-1] != time_traj.size - 1:
        frame_indices.append(time_traj.size - 1)

    return FuncAnimation(
        fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
    )


def main():
    parser = argparse.ArgumentParser(
        description="Simulate the inverted pendulum walker"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("trajectory", help="animate a trajectory and plot its energy")
    subparsers.add_parser(
        "roa", help="plot the upright controller's region of attraction"
    )
    args = parser.parse_args()

    params = {
        "gravity": 9.81,  # m/s^2
        "length": 1.0,  # m
        "mass": 1.0,  # kg
        "incline": 0.06,  # rad
        "angle_of_attack": np.pi / 8,  # rad
        "ankle_torque": 0.0,  # N m
        "ankle_torque_damping": 5.0,
    }
    timestep = 1e-4
    sim_time = 10.0
    num_timesteps = round(sim_time / timestep)
    output = Path("output/assignment_2")
    output.mkdir(parents=True, exist_ok=True)

    if args.command == "trajectory":
        initial_state = np.array([jnp.deg2rad(-90.0) + params["incline"], 0.0])
        time_traj, state_traj, impacts = simulate(
            initial_state, timestep, num_timesteps, params
        )

        fps = 25
        animation = create_walker_animation(
            time_traj, state_traj, params, timestep, fps
        )
        animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))
        energy_fig = plot_energy(time_traj, state_traj, impacts, params)
        energy_fig.savefig(output / "energy.png")
        print(f"Saved {output / 'walker.gif'} and {output / 'energy.png'}.")
    elif args.command == "roa":
        large_timestep = 1e-3
        small_timestep = 1e-4
        roa_result = find_upright_roa(params, large_timestep, small_timestep, sim_time)
        roa_fig = plot_upright_roa(roa_result)
        roa_fig.savefig(output / "upright_roa.png")
        print(f"Saved {output / 'upright_roa.png'}.")

    plt.show()


if __name__ == "__main__":
    main()
