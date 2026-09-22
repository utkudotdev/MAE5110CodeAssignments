import argparse
import enum
import functools as ft
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
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


@jax.tree_util.register_dataclass
@dataclass
class ReturnMapResult:
    alpha_values: jax.Array
    theta_dot_values: jax.Array
    next_theta_dot_grid: jax.Array


def compute_ankle_torque(state, params):
    m = params["mass"]
    g = params["gravity"]
    l = params["length"]
    b = params["ankle_torque_damping"]
    theta, theta_dot = state

    control = -2.0 * m * g * l * jnp.sin(theta) - b * theta_dot
    lower, upper = params["ankle_torque_bounds"]
    clipped = jnp.clip(control, lower, upper)

    return clipped


def simulation_step(
    time, state, timestep, small_timestep, params, use_ankle_controller=True
):
    """Advance by ``timestep``, using ``small_timestep`` near an impact.

    The inexpensive full-size integration is retained when it does not cross an
    event guard. If it does, the step is replayed with `small_timestep` through
    the impact, event dynamics are applied, and the remainder of the original
    timestep is integrated. Thus the returned state is always at
    ``time + timestep``.

    At most one impact is supported within a timestep.
    """
    ankle_torque = jnp.where(
        use_ankle_controller,
        compute_ankle_torque(state, params),
        0.0,
    )
    controlled_params = {**params, "ankle_torque": ankle_torque}
    dynamics = ft.partial(model.dynamics, params=controlled_params)

    full_step_state = rk4(dynamics, time, state, timestep)
    full_step_collision = model.event_guard(state, full_step_state, controlled_params)

    def refine_collision():
        def searching_for_impact(carry):
            _, elapsed, collision_occurred = carry
            return (elapsed < timestep) & ~collision_occurred

        def take_small_step(carry):
            current_state, elapsed, _ = carry
            step_size = jnp.minimum(small_timestep, timestep - elapsed)
            next_state = rk4(dynamics, time + elapsed, current_state, step_size)
            collision_occurred = model.event_guard(
                current_state, next_state, controlled_params
            )
            next_state = jax.lax.cond(
                collision_occurred,
                lambda: model.event_dynamics(next_state, controlled_params),
                lambda: next_state,
            )
            return next_state, elapsed + step_size, collision_occurred

        impact_state, elapsed, collision_occurred = jax.lax.while_loop(
            searching_for_impact,
            take_small_step,
            (state, jnp.asarray(0.0), jnp.asarray(False)),
        )

        remaining_time = timestep - elapsed
        end_state = rk4(dynamics, time + elapsed, impact_state, remaining_time)
        return end_state, collision_occurred

    return jax.lax.cond(
        full_step_collision,
        refine_collision,
        lambda: (full_step_state, full_step_collision),
    )


@ft.partial(jax.jit, static_argnames=["num_timesteps"])
def simulate(initial_state, timestep, small_timestep, num_timesteps, params):
    """Simulates the systems starting from `initial_state` for `num_timesteps`
    steps of length `timestep`. Returns `(time_traj, state_traj, impacts)`.

    `time_traj` is of length `num_timesteps + 1` and indicates the time at which
    each state in `state_traj` occurred. `state_traj` is of shape (2, `num_timesteps + 1`),
    one row for theta and another for theta dot. `impacts` is of length `num_timesteps + 1`
    and is true when a forward impact occurred during that timestep. When an
    impact occurs, the corresponding state is recorded at the end of the
    timestep, after event dynamics and the remaining integration.
    """

    def step(carry, _):
        t, state = carry

        next_state, collision_occurred = simulation_step(
            t, state, timestep, small_timestep, params
        )

        next_t = t + timestep

        return (next_t, next_state), (next_t, next_state, collision_occurred)

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
    step_height = 2 * length * jnp.sin(angle_of_attack) * jnp.sin(incline)
    stance_height_changes = impacts * step_height
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


def get_next_poincare_velocity(
    theta_dot,
    alpha,
    params,
    large_timestep,
    small_timestep,
    max_sim_time,
):
    """Return theta dot at the next positive crossing of the theta=0 section."""
    POINCARE_THETA = 0.0
    FALL_ANGLE = jnp.deg2rad(-89.0)

    return_map_params = {
        **params,
        "angle_of_attack": alpha,
        "ankle_torque": jnp.asarray(0.0),
    }
    dynamics = ft.partial(model.dynamics, params=return_map_params)

    def refine_section_crossing(time, state, timestep):
        def before_section(carry):
            current_state, elapsed = carry
            return (elapsed < timestep) & (current_state[0] < POINCARE_THETA)

        def take_small_step(carry):
            current_state, elapsed = carry
            step_size = jnp.minimum(small_timestep, timestep - elapsed)
            next_state = rk4(dynamics, time + elapsed, current_state, step_size)
            return next_state, elapsed + step_size

        crossing_state, _ = jax.lax.while_loop(
            before_section,
            take_small_step,
            (state, jnp.asarray(0.0)),
        )
        return crossing_state

    def continue_simulation(carry):
        _, time, _, section_crossed, fell_backward = carry
        return ~section_crossed & ~fell_backward & (time < max_sim_time)

    def take_step(carry):
        state, time, impact_occurred, _, _ = carry
        timestep = jnp.minimum(large_timestep, max_sim_time - time)
        next_state, impact_this_step = simulation_step(
            time,
            state,
            timestep,
            small_timestep,
            return_map_params,
            use_ankle_controller=False,
        )

        section_crossed = (
            impact_occurred
            & (state[0] <= POINCARE_THETA)
            & (next_state[0] >= POINCARE_THETA)
        )
        next_state = jax.lax.cond(
            section_crossed,
            lambda: refine_section_crossing(time, state, timestep),
            lambda: next_state,
        )
        fell_backward = next_state[0] < FALL_ANGLE

        return (
            next_state,
            time + timestep,
            impact_occurred | impact_this_step,
            section_crossed,
            fell_backward,
        )

    final_state, _, _, section_crossed, _ = jax.lax.while_loop(
        continue_simulation,
        take_step,
        (
            jnp.array([POINCARE_THETA, theta_dot]),
            jnp.asarray(0.0),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(False),
        ),
    )

    return jnp.where(section_crossed, final_state[1], jnp.nan)


@jax.jit
def compute_return_map(params, large_timestep, small_timestep, max_sim_time):
    """Sweep the theta=0 Poincare return map over theta dot and alpha."""
    NUM_ALPHAS = 20
    NUM_THETA_DOTS = 200
    MAX_FROUDE = 2.0

    alpha_min, alpha_max = params["angle_of_attack_bounds"]
    alpha_values = jnp.linspace(alpha_min, alpha_max, NUM_ALPHAS)
    theta_dot_values = jnp.linspace(
        0.0,
        jnp.sqrt(MAX_FROUDE * params["gravity"] / params["length"]),
        NUM_THETA_DOTS,
    )
    theta_dot_grid, alpha_grid = jnp.meshgrid(
        theta_dot_values, alpha_values, indexing="xy"
    )

    next_theta_dots = jax.vmap(
        get_next_poincare_velocity,
        in_axes=(0, 0, None, None, None, None),
    )(
        theta_dot_grid.ravel(),
        alpha_grid.ravel(),
        params,
        large_timestep,
        small_timestep,
        max_sim_time,
    )

    return ReturnMapResult(
        alpha_values=alpha_values,
        theta_dot_values=theta_dot_values,
        next_theta_dot_grid=next_theta_dots.reshape((NUM_ALPHAS, NUM_THETA_DOTS)),
    )


def plot_return_map(result: ReturnMapResult):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    color_map = plt.get_cmap("viridis")
    color_norm = Normalize(
        vmin=np.rad2deg(result.alpha_values[0]),
        vmax=np.rad2deg(result.alpha_values[-1]),
    )

    theta_dot_values = np.rad2deg(result.theta_dot_values)
    for alpha, next_theta_dots in zip(result.alpha_values, result.next_theta_dot_grid):
        alpha_degrees = np.rad2deg(alpha)
        ax.plot(
            theta_dot_values,
            np.rad2deg(next_theta_dots),
            color=color_map(color_norm(alpha_degrees)),
            linewidth=1.5,
        )

    velocity_max = np.nanmax(
        [np.max(theta_dot_values), np.nanmax(np.rad2deg(result.next_theta_dot_grid))]
    )
    ax.plot([0.0, velocity_max], [0.0, velocity_max], "k--", label="Identity")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap=color_map),
        ax=ax,
        label=r"Angle of attack $\alpha$ (deg)",
    )
    ax.set_xlim(0.0, velocity_max)
    ax.set_ylim(0.0, velocity_max)
    ax.set_title(r"Walker Return Map on $\theta=0$")
    ax.set_xlabel(r"Current velocity $\dot{\theta}_k$ (deg/s)")
    ax.set_ylabel(r"Next velocity $\dot{\theta}_{k+1}$ (deg/s)")
    ax.grid(alpha=0.25)
    ax.legend()

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

        timestep = jnp.minimum(large_timestep, sim_time - time)
        next_state, collision_occurred = simulation_step(
            time, state, timestep, small_timestep, params
        )

        stable = stabilized(next_state)
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
    subparsers.add_parser("return-map", help="plot the theta=0 Poincare return map")
    args = parser.parse_args()

    params = {
        "gravity": 9.81,  # m/s^2
        "length": 1.0,  # m
        "mass": 1.0,  # kg
        "incline": 0.06,  # rad
        "angle_of_attack": np.pi / 8,  # rad
        "angle_of_attack_bounds": np.array([np.pi / 8, np.pi / 7]),  # rad
        "ankle_torque": 0.0,  # N m
        "ankle_torque_bounds": np.array([-0.1 * 9.81, 0.05 * 9.81]),  # N m
        "ankle_torque_damping": 5.0,
    }
    output = Path("output/assignment_2")
    output.mkdir(parents=True, exist_ok=True)

    if args.command == "trajectory":
        sim_time = 3.0
        timestep = 1e-3
        small_timestep = 1e-4
        num_timesteps = round(sim_time / timestep)
        initial_state = np.array([params["incline"], 2.0])

        start = time.perf_counter()
        time_traj, state_traj, impacts = simulate(
            initial_state, timestep, small_timestep, num_timesteps, params
        )
        jax.block_until_ready((time_traj, state_traj, impacts))
        end = time.perf_counter()
        print(f"Computed trajectory in {end - start}s")

        fps = 25
        animation = create_walker_animation(
            time_traj, state_traj, params, timestep, fps
        )
        animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))
        energy_fig = plot_energy(time_traj, state_traj, impacts, params)
        energy_fig.savefig(output / "energy.png")
        print(f"Saved {output / 'walker.gif'} and {output / 'energy.png'}.")
    elif args.command == "roa":
        sim_time = 10.0
        large_timestep = 1e-2
        small_timestep = 1e-4

        start = time.perf_counter()
        roa_result = find_upright_roa(params, large_timestep, small_timestep, sim_time)
        jax.block_until_ready(roa_result)
        end = time.perf_counter()
        print(f"Computed RoA in {end - start}s")

        roa_fig = plot_upright_roa(roa_result)
        roa_fig.savefig(output / "upright_roa.png")
        print(f"Saved {output / 'upright_roa.png'}.")
    elif args.command == "return-map":
        sim_time = 5.0
        large_timestep = 1e-2
        small_timestep = 1e-4

        start = time.perf_counter()
        return_map = compute_return_map(
            params,
            large_timestep,
            small_timestep,
            sim_time,
        )
        jax.block_until_ready(return_map)
        end = time.perf_counter()
        print(f"Computed return map in {end - start}s")

        return_map_fig = plot_return_map(return_map)
        return_map_fig.savefig(output / "return_map.png")
        print(f"Saved {output / 'return_map.png'}.")

    plt.show()


if __name__ == "__main__":
    main()
