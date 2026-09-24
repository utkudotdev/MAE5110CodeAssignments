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
    entered_roa_grid: jax.Array


@dataclass
class StepsToStabilityResult:
    theta_dot_values: np.ndarray
    steps: np.ndarray
    alpha_values: np.ndarray


@jax.tree_util.register_dataclass
@dataclass
class PolicyRolloutResult:
    theta_dot_values: jax.Array
    steps: jax.Array
    converged: jax.Array
    fell_backward: jax.Array
    first_alpha: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class InitialStateStepsResult:
    theta_values: jax.Array
    theta_dot_values: jax.Array
    steps_grid: jax.Array


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


def is_state_in_roa(state, roa_bounds):
    closest_index = jnp.argmin(jnp.abs(roa_bounds[:, 0] - state[0]))
    lower_theta_dot = roa_bounds[closest_index, 1]
    upper_theta_dot = roa_bounds[closest_index, 2]
    return (
        (state[0] >= roa_bounds[0, 0])
        & (state[0] <= roa_bounds[-1, 0])
        & (state[1] >= lower_theta_dot)
        & (state[1] <= upper_theta_dot)
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class StandingController:
    alpha: jax.Array

    def update(self, state, params):
        control = jnp.array([compute_ankle_torque(state, params), self.alpha])
        return self, control


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MinimumAlphaController:
    roa_bounds: jax.Array
    ankle_controller_enabled: jax.Array

    def update(self, state, params):
        ankle_controller_enabled = self.ankle_controller_enabled | is_state_in_roa(
            state, self.roa_bounds
        )
        ankle_torque = jnp.where(
            ankle_controller_enabled,
            compute_ankle_torque(state, params),
            0.0,
        )
        alpha = params["angle_of_attack_bounds"][0]
        next_controller = MinimumAlphaController(
            roa_bounds=self.roa_bounds,
            ankle_controller_enabled=ankle_controller_enabled,
        )
        return next_controller, jnp.array([ankle_torque, alpha])


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LookupTableController:
    policy_map: jax.Array
    roa_bounds: jax.Array
    previous_theta: jax.Array
    alpha: jax.Array
    ankle_controller_enabled: jax.Array
    initialized: jax.Array

    def update(self, state, params):
        starts_on_section = ~self.initialized & (state[0] == 0.0) & (state[1] >= 0.0)
        section_crossed = starts_on_section | (
            self.initialized & (self.previous_theta <= 0.0) & (state[0] > 0.0)
        )
        closest_index = jnp.argmin(jnp.abs(self.policy_map[:, 0] - state[1]))
        steps_remaining = self.policy_map[closest_index, 1]
        selected_alpha = self.policy_map[closest_index, 2]

        inside_roa = is_state_in_roa(state, self.roa_bounds)
        ankle_controller_enabled = self.ankle_controller_enabled | inside_roa

        walking_alpha = jnp.where(
            section_crossed & (steps_remaining > 0),
            selected_alpha,
            jnp.where(
                self.initialized, self.alpha, params["angle_of_attack_bounds"][0]
            ),
        )
        alpha = jnp.where(
            ankle_controller_enabled,
            params["angle_of_attack_bounds"][0],
            walking_alpha,
        )
        ankle_torque = jnp.where(
            ankle_controller_enabled,
            compute_ankle_torque(state, params),
            0.0,
        )

        next_controller = LookupTableController(
            policy_map=self.policy_map,
            roa_bounds=self.roa_bounds,
            previous_theta=state[0],
            alpha=alpha,
            ankle_controller_enabled=ankle_controller_enabled,
            initialized=jnp.asarray(True),
        )
        control = jnp.array([ankle_torque, alpha])
        return next_controller, control


def simulation_step(time, state, control, timestep, small_timestep, params):
    """Advance by ``timestep``, using ``small_timestep`` near an impact.

    ``control`` contains ankle torque and angle of attack, and is held constant
    for the entire outer timestep.

    The inexpensive full-size integration is retained when it does not cross an
    event guard. If it does, the step is replayed with `small_timestep` through
    the impact, event dynamics are applied, and the remainder of the original
    timestep is integrated. Thus the returned state is always at
    ``time + timestep``.

    At most one impact is supported within a timestep.
    """
    dynamics = ft.partial(model.dynamics, control=control, params=params)

    full_step_state = rk4(dynamics, time, state, timestep)
    full_step_collision = model.event_guard(state, full_step_state, control, params)

    def refine_collision():
        def searching_for_impact(carry):
            _, elapsed, collision_occurred = carry
            return (elapsed < timestep) & ~collision_occurred

        def take_small_step(carry):
            current_state, elapsed, _ = carry
            step_size = jnp.minimum(small_timestep, timestep - elapsed)
            next_state = rk4(dynamics, time + elapsed, current_state, step_size)
            collision_occurred = model.event_guard(
                current_state, next_state, control, params
            )
            next_state = jax.lax.cond(
                collision_occurred,
                lambda: model.event_dynamics(next_state, control, params),
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
def simulate(
    initial_state,
    timestep,
    small_timestep,
    num_timesteps,
    params,
    controller,
):
    """Simulates the systems starting from `initial_state` for `num_timesteps`
    steps of length `timestep`. Returns
    `(time_traj, state_traj, impacts, control_traj)`.

    `time_traj` is of length `num_timesteps + 1` and indicates the time at which
    each state in `state_traj` occurred. `state_traj` is of shape (2, `num_timesteps + 1`),
    one row for theta and another for theta dot. `impacts` is of length `num_timesteps + 1`
    and is true when a forward impact occurred during that timestep. When an
    impact occurs, the corresponding state is recorded at the end of the
    timestep, after event dynamics and the remaining integration.

    The immutable controller is updated at the start of each outer timestep.
    It returns its next state and the control ``[ankle_torque, alpha]`` to use
    during that timestep. Each returned control is paired with the state and
    time at which it was evaluated; the final control is evaluated without
    taking an additional simulation step.
    """

    def step(carry, _):
        t, state, current_controller = carry

        next_controller, control = current_controller.update(state, params)

        next_state, collision_occurred = simulation_step(
            t, state, control, timestep, small_timestep, params
        )

        next_t = t + timestep
        next_carry = (next_t, next_state, next_controller)
        return next_carry, (t, state, collision_occurred, control)

    (
        (final_time, final_state, final_controller),
        (
            time_traj,
            state_traj,
            step_impacts,
            control_traj,
        ),
    ) = jax.lax.scan(
        step,
        (0.0, initial_state, controller),
        length=num_timesteps,
    )
    _, final_control = final_controller.update(final_state, params)

    time_traj = jnp.concatenate([time_traj, final_time.reshape((1,))])
    state_traj = jnp.concatenate([state_traj, final_state.reshape((1, -1))]).T
    impacts = jnp.concatenate([jnp.array([False]), step_impacts])
    control_traj = jnp.concatenate(
        [control_traj, final_control.reshape((1, -1))], axis=0
    )

    return time_traj, state_traj, impacts, control_traj


def calculate_absolute_energy(state_traj, impacts, params, control_traj):
    """Return kinetic and potential energy in a fixed global reference frame."""
    gravity = params["gravity"]
    mass = params["mass"]
    length = params["length"]
    incline = params["incline"]
    # impacts[i] describes the integration step from state i-1 to state i.
    alpha_traj = control_traj[:, 1]
    impact_alphas = jnp.concatenate([alpha_traj[:1], alpha_traj[:-1]])

    kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
    step_height = 2 * length * jnp.sin(impact_alphas) * jnp.sin(incline)
    # A forward step places the new stance foot lower on the downhill slope.
    stance_height_changes = -(impacts * step_height)
    stance_height = jnp.cumsum(stance_height_changes)
    potential_energy = potential_energy + mass * gravity * stance_height

    return kinetic_energy, potential_energy


def plot_energy(time_traj, state_traj, impacts, params, control_traj):
    kinetic_energy, potential_energy = calculate_absolute_energy(
        state_traj, impacts, params, control_traj
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


def plot_state_space(state_traj, impacts, control_traj, params):
    theta_degrees = np.rad2deg(np.asarray(state_traj[0]))
    theta_dot_degrees = np.rad2deg(np.asarray(state_traj[1]))
    impacts = np.asarray(impacts, dtype=bool)
    gamma = float(params["incline"])
    alpha_values = np.unique(np.asarray(control_traj[:, 1]))

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(theta_degrees, theta_dot_degrees, color="#23699b", linewidth=1.8)
    ax.scatter(
        theta_degrees[impacts],
        theta_dot_degrees[impacts],
        color="#df8a25",
        s=24,
        zorder=3,
        label="Post-impact state",
    )
    ax.scatter(
        theta_degrees[0],
        theta_dot_degrees[0],
        color="#2a9d8f",
        s=45,
        zorder=4,
        label="Initial state",
    )
    ax.scatter(
        theta_degrees[-1],
        theta_dot_degrees[-1],
        color="#e63946",
        s=45,
        zorder=4,
        label="Final state",
    )

    for index, alpha in enumerate(alpha_values):
        ax.axvline(
            np.rad2deg(gamma + alpha),
            color="#df8a25",
            linestyle="--",
            linewidth=1.0,
            alpha=0.45,
            label=r"Touchdown: $\theta=\alpha+\gamma$" if index == 0 else None,
        )
        ax.axvline(
            np.rad2deg(gamma - alpha),
            color="#6f42c1",
            linestyle=":",
            linewidth=1.0,
            alpha=0.45,
            label=(
                r"Post-impact: $\theta=\gamma-\alpha=-(\alpha-\gamma)$"
                if index == 0
                else None
            ),
        )

    ax.set_title("Closed-Loop State-Space Trajectory")
    ax.set_xlabel(r"Angle $\theta$ (deg)")
    ax.set_ylabel(r"Angular velocity $\dot{\theta}$ (deg/s)")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    return fig


def can_stabilize(
    initial_state, params, large_timestep, small_timestep, sim_time
) -> jax.Array:
    controller = StandingController(params["angle_of_attack_bounds"][0])

    def wrap_angle(theta):
        return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

    def fell_backwards(state):
        theta, _ = state
        collide_angle = jnp.deg2rad(-90.0) + params["incline"]
        return wrap_angle(theta) < collide_angle - jnp.deg2rad(1.0)

    def stabilized(state):
        theta, theta_dot = state
        return (jnp.abs(wrap_angle(theta)) < np.deg2rad(1)) & (
            jnp.abs(theta_dot) < np.deg2rad(0.1)
        )

    def step(val):
        state, time, _, _ = val

        timestep = jnp.minimum(large_timestep, sim_time - time)
        _, control = controller.update(state, params)
        next_state, collision_occurred = simulation_step(
            time, state, control, timestep, small_timestep, params
        )

        stable = stabilized(next_state)
        next_time = time + timestep
        done = (
            fell_backwards(next_state)
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


@jax.jit
def find_upright_roa(
    params, theta_values, theta_dot_values, large_timestep, small_timestep, sim_time
):
    all_initial_states = jnp.stack(
        jnp.meshgrid(theta_values, theta_dot_values, indexing="ij"), axis=-1
    )
    flat_initial_states = all_initial_states.reshape((-1, 2))

    stable = jax.vmap(can_stabilize, in_axes=(0, None, None, None, None))(
        flat_initial_states,
        params,
        large_timestep,
        small_timestep,
        sim_time,
    )
    classification_grid = stable.reshape(
        (theta_values.size, theta_dot_values.size)
    ).astype(int)

    return RoAResult(
        theta_values=theta_values,
        theta_dot_values=theta_dot_values,
        classification_grid=classification_grid,
    )


def get_roa_bounds(result: RoAResult) -> np.ndarray:
    """Return [theta, minimum theta dot, maximum theta dot] for each angle.
    Assumes RoA is contiguous in theta dot at all theta, which it looks like it is.
    """
    theta_values = np.asarray(result.theta_values)
    theta_dot_values = np.asarray(result.theta_dot_values)
    stabilizable = (
        np.asarray(result.classification_grid) == RoAClassification.STABILIZABLE
    )
    has_stabilizable_state = np.any(stabilizable, axis=1)
    lower_indices = np.argmax(stabilizable, axis=1)
    upper_indices = stabilizable.shape[1] - 1 - np.argmax(stabilizable[:, ::-1], axis=1)
    lower_bounds = np.where(
        has_stabilizable_state, theta_dot_values[lower_indices], np.nan
    )
    upper_bounds = np.where(
        has_stabilizable_state, theta_dot_values[upper_indices], np.nan
    )

    return np.column_stack((theta_values, lower_bounds, upper_bounds))


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


def get_next_poincare_velocity_or_roa(
    theta_dot,
    alpha,
    params,
    roa_bounds,
    large_timestep,
    small_timestep,
    max_sim_time,
):
    """Return the next section velocity or report earlier entry into the RoA."""
    POINCARE_THETA = 0.0
    FALL_ANGLE = params["incline"] - jnp.pi / 2 - jnp.deg2rad(1.0)

    initial_state = jnp.array([POINCARE_THETA, theta_dot])
    control = jnp.array([0.0, alpha])
    dynamics = ft.partial(model.dynamics, control=control, params=params)

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
        _, time, _, section_crossed, fell_backward, entered_roa = carry
        return ~section_crossed & ~fell_backward & ~entered_roa & (time < max_sim_time)

    def take_step(carry):
        state, time, impact_occurred, _, _, _ = carry
        timestep = jnp.minimum(large_timestep, max_sim_time - time)
        next_state, impact_this_step = simulation_step(
            time,
            state,
            control,
            timestep,
            small_timestep,
            params,
        )

        section_crossed = (
            impact_occurred
            & (state[0] <= POINCARE_THETA)
            & (next_state[0] >= POINCARE_THETA)
            & (next_state[1] >= 0.0)
        )
        next_state = jax.lax.cond(
            section_crossed,
            lambda: refine_section_crossing(time, state, timestep),
            lambda: next_state,
        )
        fell_backward = next_state[0] < FALL_ANGLE
        entered_roa = is_state_in_roa(next_state, roa_bounds)

        return (
            next_state,
            time + timestep,
            impact_occurred | impact_this_step,
            section_crossed,
            fell_backward,
            entered_roa,
        )

    final_state, _, _, section_crossed, _, entered_roa = jax.lax.while_loop(
        continue_simulation,
        take_step,
        (
            initial_state,
            jnp.asarray(0.0),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(False),
            is_state_in_roa(initial_state, roa_bounds),
        ),
    )

    crossing_velocity = jnp.where(section_crossed, final_state[1], jnp.nan)
    return crossing_velocity, entered_roa


@jax.jit
def compute_return_map(
    params,
    theta_dot_values,
    alpha_values,
    large_timestep,
    small_timestep,
    max_sim_time,
    roa_bounds,
):
    """Sweep next-section velocities and RoA entries over theta dot and alpha."""
    theta_dot_grid, alpha_grid = jnp.meshgrid(
        theta_dot_values, alpha_values, indexing="xy"
    )

    next_theta_dots, entered_roa = jax.vmap(
        get_next_poincare_velocity_or_roa,
        in_axes=(0, 0, None, None, None, None, None),
    )(
        theta_dot_grid.ravel(),
        alpha_grid.ravel(),
        params,
        roa_bounds,
        large_timestep,
        small_timestep,
        max_sim_time,
    )

    return ReturnMapResult(
        alpha_values=alpha_values,
        theta_dot_values=theta_dot_values,
        next_theta_dot_grid=next_theta_dots.reshape(
            (alpha_values.size, theta_dot_values.size)
        ),
        entered_roa_grid=entered_roa.reshape(
            (alpha_values.size, theta_dot_values.size)
        ),
    )


def compute_steps_to_stability(
    roa_result: RoAResult,
    return_map_result: ReturnMapResult,
    state_match_tolerance,
) -> StepsToStabilityResult:
    """Find the minimum number of footsteps needed to enter the upright RoA."""
    theta_values = np.asarray(roa_result.theta_values)
    theta_dot_values = np.asarray(return_map_result.theta_dot_values)
    roa_theta_dot_values = np.asarray(roa_result.theta_dot_values)
    if not np.array_equal(theta_dot_values, roa_theta_dot_values):
        raise ValueError("RoA and return map must use the same theta-dot grid.")

    poincare_index = np.argmin(np.abs(theta_values))
    if not np.isclose(theta_values[poincare_index], 0.0):
        raise ValueError("The RoA theta grid must contain theta=0.")

    steps = np.full(theta_dot_values.shape, -1, dtype=int)
    alpha_values = np.full(theta_dot_values.shape, np.nan)
    in_upright_roa = (
        np.asarray(roa_result.classification_grid[poincare_index])
        == RoAClassification.STABILIZABLE
    )
    steps[in_upright_roa] = 0

    next_theta_dot_grid = np.asarray(return_map_result.next_theta_dot_grid)
    entered_roa_grid = np.asarray(return_map_result.entered_roa_grid)
    next_step_count = 1
    while True:
        unmarked_indices = np.flatnonzero(steps < 0)
        marked_velocities = theta_dot_values[steps >= 0]
        if not unmarked_indices.size or not marked_velocities.size:
            break

        candidate_next_states = next_theta_dot_grid[:, unmarked_indices]
        distances = np.abs(
            candidate_next_states[:, :, None] - marked_velocities[None, None, :]
        )
        distances = np.where(np.isfinite(distances), distances, np.inf)
        closest_marked_distance = np.min(distances, axis=2)
        successful_actions = entered_roa_grid[:, unmarked_indices] | (
            closest_marked_distance <= state_match_tolerance
        )
        successful_action_counts = np.sum(successful_actions, axis=0)
        median_success_ranks = np.maximum(successful_action_counts - 1, 0) // 2
        cumulative_successes = np.cumsum(successful_actions, axis=0)
        best_alpha_indices = np.argmax(
            cumulative_successes > median_success_ranks[None, :],
            axis=0,
        )
        newly_reachable = np.any(successful_actions, axis=0)
        if not np.any(newly_reachable):
            break

        newly_reachable_indices = unmarked_indices[newly_reachable]
        steps[newly_reachable_indices] = next_step_count
        alpha_values[newly_reachable_indices] = np.asarray(
            return_map_result.alpha_values
        )[best_alpha_indices[newly_reachable]]
        next_step_count += 1

    return StepsToStabilityResult(
        theta_dot_values=theta_dot_values,
        steps=steps,
        alpha_values=alpha_values,
    )


def rollout_policy_from_section(
    theta_dot,
    params,
    policy_map,
    roa_bounds,
    timestep,
    small_timestep,
    num_timesteps,
):
    initial_state = jnp.array([0.0, theta_dot])
    initial_alpha = params["angle_of_attack_bounds"][0]
    controller = LookupTableController(
        policy_map=policy_map,
        roa_bounds=roa_bounds,
        previous_theta=initial_state[0],
        alpha=initial_alpha,
        ankle_controller_enabled=jnp.asarray(False),
        initialized=jnp.asarray(False),
    )
    _, state_traj, impacts, control_traj = simulate(
        initial_state,
        timestep,
        small_timestep,
        num_timesteps,
        params,
        controller,
    )

    fall_angle = params["incline"] - jnp.pi / 2 - jnp.deg2rad(1.0)
    fell_backward = jnp.any(state_traj[0] < fall_angle)
    final_theta = (state_traj[0, -1] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    converged = (
        ~fell_backward
        & (jnp.abs(final_theta) < jnp.deg2rad(1.0))
        & (jnp.abs(state_traj[1, -1]) < jnp.deg2rad(0.1))
    )
    return jnp.count_nonzero(impacts), converged, fell_backward, control_traj[0, 1]


@ft.partial(jax.jit, static_argnames=["num_timesteps"])
def validate_policy_rollouts(
    params,
    theta_dot_values,
    policy_map,
    roa_bounds,
    timestep,
    small_timestep,
    num_timesteps,
):
    steps, converged, fell_backward, first_alpha = jax.vmap(
        rollout_policy_from_section,
        in_axes=(0, None, None, None, None, None, None),
    )(
        theta_dot_values,
        params,
        policy_map,
        roa_bounds,
        timestep,
        small_timestep,
        num_timesteps,
    )
    return PolicyRolloutResult(
        theta_dot_values=theta_dot_values,
        steps=steps,
        converged=converged,
        fell_backward=fell_backward,
        first_alpha=first_alpha,
    )


def plot_steps_to_stability(
    result: StepsToStabilityResult,
    rollout_result: PolicyRolloutResult | None = None,
):
    max_steps = max(0, np.max(result.steps))
    if rollout_result is not None:
        rollout_steps = np.asarray(rollout_result.steps)
        converged = np.asarray(rollout_result.converged)
        successful_rollout_steps = rollout_steps[converged]
        if successful_rollout_steps.size:
            max_steps = max(max_steps, int(np.max(successful_rollout_steps)))
    step_colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.95, max_steps + 1))
    color_map = ListedColormap(["#6c757d", *step_colors])
    color_norm = BoundaryNorm(
        np.arange(max_steps + 3) - 0.5,
        color_map.N,
    )

    plot_values = np.where(result.steps < 0, 0, result.steps + 1)
    velocity_limits = np.rad2deg(result.theta_dot_values[[0, -1]])
    velocity_edges = np.linspace(*velocity_limits, plot_values.size + 1)

    if rollout_result is None:
        fig, ax = plt.subplots(figsize=(8, 2.5), layout="constrained")
        axes = [ax]
    else:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(8, 8),
            sharex=True,
            layout="constrained",
        )
        ax = axes[0]
    ax.pcolormesh(
        velocity_edges,
        np.array([0.0, 1.0]),
        plot_values[None, :],
        cmap=color_map,
        norm=color_norm,
        shading="flat",
    )

    legend_handles = [Patch(color="#6c757d", label="Unreachable / did not converge")]
    legend_handles.extend(
        Patch(
            color=step_colors[steps],
            label=f"{steps} step" if steps == 1 else f"{steps} steps",
        )
        for steps in range(max_steps + 1)
        if np.any(result.steps == steps)
        or (rollout_result is not None and np.any(successful_rollout_steps == steps))
    )
    ax.legend(
        handles=legend_handles,
        title="Steps to stability",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    ax.set_yticks([])
    ax.set_title(r"Steps to Reach the Upright RoA from $\theta=0$")
    if rollout_result is None:
        ax.set_xlabel(r"Initial velocity $\dot{\theta}_k$ (deg/s)")
    else:
        rollout_ax = axes[1]
        rollout_steps = np.asarray(rollout_result.steps)
        converged = np.asarray(rollout_result.converged)
        rollout_plot_values = np.where(converged, rollout_steps + 1, 0)
        rollout_velocity_edges = np.linspace(
            *velocity_limits, rollout_plot_values.size + 1
        )
        rollout_ax.pcolormesh(
            rollout_velocity_edges,
            np.array([0.0, 1.0]),
            rollout_plot_values[None, :],
            cmap=color_map,
            norm=color_norm,
            shading="flat",
        )
        rollout_ax.set_title("High-Resolution Closed-Loop Rollouts")
        rollout_ax.set_yticks([])

        alpha_ax = axes[2]
        first_alpha_degrees = np.rad2deg(np.asarray(rollout_result.first_alpha))
        alpha_color_map = plt.get_cmap("plasma")
        alpha_color_norm = Normalize(
            vmin=np.min(first_alpha_degrees),
            vmax=np.max(first_alpha_degrees),
        )
        alpha_mesh = alpha_ax.pcolormesh(
            rollout_velocity_edges,
            np.array([0.0, 1.0]),
            first_alpha_degrees[None, :],
            cmap=alpha_color_map,
            norm=alpha_color_norm,
            shading="flat",
        )
        alpha_ax.set_title("First-Step Angle of Attack")
        alpha_ax.set_xlabel(r"Initial velocity $\dot{\theta}_k$ (deg/s)")
        alpha_ax.set_yticks([])
        fig.colorbar(
            alpha_mesh,
            ax=alpha_ax,
            label=r"Angle of attack $\alpha$ (deg)",
        )

    return fig


def simulate_and_count_steps(
    initial_state,
    params,
    policy_map,
    roa_bounds,
    timestep,
    small_timestep,
    num_timesteps,
):
    """Simulate the lookup controller and count impacts."""
    initial_alpha = params["angle_of_attack_bounds"][0]
    controller = LookupTableController(
        policy_map=policy_map,
        roa_bounds=roa_bounds,
        previous_theta=initial_state[0],
        alpha=initial_alpha,
        ankle_controller_enabled=jnp.asarray(False),
        initialized=jnp.asarray(False),
    )
    _, state_traj, impacts, _ = simulate(
        initial_state,
        timestep,
        small_timestep,
        num_timesteps,
        params,
        controller,
    )
    fall_angle = params["incline"] - jnp.pi / 2 - jnp.deg2rad(1.0)
    fell_backward = jnp.any(state_traj[0] < fall_angle)
    return jnp.where(fell_backward, -1, jnp.count_nonzero(impacts))


@ft.partial(jax.jit, static_argnames=["num_timesteps"])
def compute_initial_state_steps(
    params,
    theta_values,
    theta_dot_values,
    policy_map,
    roa_bounds,
    timestep,
    small_timestep,
    num_timesteps,
):
    """Simulate every initial state and count impacts over a fixed horizon."""
    initial_states = jnp.stack(
        jnp.meshgrid(theta_values, theta_dot_values, indexing="ij"), axis=-1
    )
    flat_initial_states = initial_states.reshape((-1, 2))
    total_steps = jax.vmap(
        simulate_and_count_steps,
        in_axes=(0, None, None, None, None, None, None),
    )(
        flat_initial_states,
        params,
        policy_map,
        roa_bounds,
        timestep,
        small_timestep,
        num_timesteps,
    )

    return InitialStateStepsResult(
        theta_values=theta_values,
        theta_dot_values=theta_dot_values,
        steps_grid=total_steps.reshape((theta_values.size, theta_dot_values.size)),
    )


def plot_initial_state_steps(result: InitialStateStepsResult):
    steps_grid = np.asarray(result.steps_grid)
    max_steps = max(0, np.max(steps_grid))
    step_colors = plt.get_cmap("tab10")(np.arange(max_steps + 1))
    color_map = ListedColormap(["#6c757d", *step_colors])
    color_norm = BoundaryNorm(
        np.arange(max_steps + 3) - 0.5,
        color_map.N,
    )
    plot_grid = np.where(steps_grid < 0, 0, steps_grid + 1)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.pcolormesh(
        np.rad2deg(result.theta_values),
        np.rad2deg(result.theta_dot_values),
        plot_grid.T,
        cmap=color_map,
        norm=color_norm,
        shading="nearest",
    )

    legend_handles = []
    if np.any(steps_grid < 0):
        legend_handles.append(Patch(color="#6c757d", label="Unreachable"))
    legend_handles.extend(
        Patch(
            color=step_colors[steps],
            label=f"{steps} step" if steps == 1 else f"{steps} steps",
        )
        for steps in range(max_steps + 1)
        if np.any(steps_grid == steps)
    )
    ax.legend(
        handles=legend_handles,
        title="Footsteps",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    ax.set_title("Footsteps During Fixed-Time Controller Simulation")
    ax.set_xlabel(r"Initial angle $\theta$ (deg)")
    ax.set_ylabel(r"Initial angular velocity $\dot{\theta}$ (deg/s)")
    ax.grid(alpha=0.25)

    return fig


def create_walker_animation(
    time_traj, state_traj, control_traj, params, timestep, fps=25
):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    def draw_frame(index):
        # The massless swing leg is repositioned instantaneously at each impact.
        model.visualize(state_traj[:, index], params, control_traj[index], ax=ax)
        ax.set_title(f"t = {time_traj[index]:.2f} s")

    # Simulate at a small timestep, but render only at the requested frame rate.
    frame_stride = round(1 / (fps * timestep))
    frame_indices = list(range(0, time_traj.size, frame_stride))
    if frame_indices[-1] != time_traj.size - 1:
        frame_indices.append(time_traj.size - 1)

    return FuncAnimation(
        fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
    )


def run_trajectory(args, params, output):
    sim_time = 3.0
    timestep = 1e-3
    small_timestep = 1e-4
    num_timesteps = round(sim_time / timestep)
    initial_state = np.array([-0.1, 3.0])

    start = time.perf_counter()
    initial_alpha = jnp.asarray(params["angle_of_attack_bounds"][0])
    controller_name = args.controller
    if controller_name == "standing":
        controller = StandingController(initial_alpha)
    elif controller_name == "minimum-alpha":
        roa_bounds = jnp.asarray(np.load(args.roa_bounds_file, allow_pickle=False))
        controller = MinimumAlphaController(
            roa_bounds=roa_bounds,
            ankle_controller_enabled=jnp.asarray(False),
        )
    else:
        policy_map = np.load(args.map_file, allow_pickle=False)
        roa_bounds = np.load(args.roa_bounds_file, allow_pickle=False)
        # The policy is defined on the theta=0 Poincare section.
        controller = LookupTableController(
            policy_map=jnp.asarray(policy_map),
            roa_bounds=jnp.asarray(roa_bounds),
            previous_theta=jnp.asarray(initial_state[0]),
            alpha=initial_alpha,
            ankle_controller_enabled=jnp.asarray(False),
            initialized=jnp.asarray(False),
        )
    print(f"Using {controller_name} controller.")
    time_traj, state_traj, impacts, control_traj = simulate(
        initial_state,
        timestep,
        small_timestep,
        num_timesteps,
        params,
        controller,
    )
    jax.block_until_ready((time_traj, state_traj, impacts, control_traj))
    end = time.perf_counter()
    print(f"Computed trajectory in {end - start}s")

    fps = 25
    animation = create_walker_animation(
        time_traj, state_traj, control_traj, params, timestep, fps
    )
    animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))
    energy_fig = plot_energy(time_traj, state_traj, impacts, params, control_traj)
    energy_fig.savefig(output / "energy.png")
    state_space_fig = plot_state_space(
        state_traj,
        impacts,
        control_traj,
        params,
    )
    state_space_fig.savefig(output / "state_space.png")
    print(
        f"Saved {output / 'walker.gif'}, {output / 'energy.png'}, and "
        f"{output / 'state_space.png'}."
    )


def run_roa(params, output):
    NUM_THETAS = 200
    NUM_THETA_DOTS = 200
    THETA_MIN = np.deg2rad(-90.0) + params["incline"]
    THETA_MAX = params["incline"] + params["angle_of_attack_bounds"][0]
    THETA_DOT_MIN = np.deg2rad(-100.0)
    THETA_DOT_MAX = np.deg2rad(300.0)
    sim_time = 10.0
    large_timestep = 1e-2
    small_timestep = 1e-4
    theta_values = jnp.linspace(THETA_MIN, THETA_MAX, NUM_THETAS)
    theta_dot_values = jnp.linspace(THETA_DOT_MIN, THETA_DOT_MAX, NUM_THETA_DOTS)

    start = time.perf_counter()
    roa_result = find_upright_roa(
        params,
        theta_values,
        theta_dot_values,
        large_timestep,
        small_timestep,
        sim_time,
    )
    jax.block_until_ready(roa_result)
    end = time.perf_counter()
    print(f"Computed RoA in {end - start}s")

    roa_fig = plot_upright_roa(roa_result)
    roa_path = output / "upright_roa.png"
    roa_fig.savefig(roa_path)
    roa_bounds_path = output / "roa_bounds.npy"
    np.save(roa_bounds_path, get_roa_bounds(roa_result))
    print(f"Saved {roa_path} and {roa_bounds_path}.")


def run_lookup_table(args, params, output):
    NUM_THETA_DOTS = 40
    NUM_ALPHAS = 4
    MAX_FROUDE = 2.0
    ROA_SIM_TIME = 10.0
    RETURN_MAP_SIM_TIME = 5.0
    LARGE_TIMESTEP = 1e-2
    SMALL_TIMESTEP = 1e-4
    VALIDATION_NUM_THETA_DOTS = 2000
    VALIDATION_TIMESTEP = 1e-4
    VALIDATION_SIM_TIME = 10.0
    VALIDATION_NUM_TIMESTEPS = round(VALIDATION_SIM_TIME / VALIDATION_TIMESTEP)

    theta_values = jnp.array([0.0])
    theta_dot_values = jnp.linspace(
        0.0,
        jnp.sqrt(MAX_FROUDE * params["gravity"] / params["length"]),
        NUM_THETA_DOTS,
    )
    alpha_values = jnp.linspace(*params["angle_of_attack_bounds"], NUM_ALPHAS)
    STATE_MATCH_TOLERANCE = float((theta_dot_values[1] - theta_dot_values[0]) / 2)
    roa_bounds = jnp.asarray(np.load(args.roa_bounds_file, allow_pickle=False))

    start = time.perf_counter()
    roa_result = find_upright_roa(
        params,
        theta_values,
        theta_dot_values,
        LARGE_TIMESTEP,
        SMALL_TIMESTEP,
        ROA_SIM_TIME,
    )
    return_map = compute_return_map(
        params,
        theta_dot_values,
        alpha_values,
        LARGE_TIMESTEP,
        SMALL_TIMESTEP,
        RETURN_MAP_SIM_TIME,
        roa_bounds,
    )
    jax.block_until_ready((roa_result, return_map))
    steps_result = compute_steps_to_stability(
        roa_result,
        return_map,
        STATE_MATCH_TOLERANCE,
    )
    end = time.perf_counter()
    print(f"Computed lookup table in {end - start}s")
    print(
        f"Reachable states: {np.count_nonzero(steps_result.steps >= 0)}/"
        f"{steps_result.steps.size}; maximum steps: {np.max(steps_result.steps)}"
    )

    policy_map = np.column_stack(
        (
            steps_result.theta_dot_values,
            steps_result.steps,
            steps_result.alpha_values,
        )
    )
    validation_theta_dot_values = jnp.linspace(
        0.0,
        jnp.sqrt(MAX_FROUDE * params["gravity"] / params["length"]),
        VALIDATION_NUM_THETA_DOTS,
    )
    validation_start = time.perf_counter()
    rollout_result = validate_policy_rollouts(
        params,
        validation_theta_dot_values,
        jnp.asarray(policy_map),
        roa_bounds,
        VALIDATION_TIMESTEP,
        SMALL_TIMESTEP,
        VALIDATION_NUM_TIMESTEPS,
    )
    jax.block_until_ready(rollout_result)
    validation_end = time.perf_counter()
    converged = np.asarray(rollout_result.converged)
    fell_backward = np.asarray(rollout_result.fell_backward)
    print(
        f"Validated {converged.size} rollouts in "
        f"{validation_end - validation_start}s: "
        f"{np.count_nonzero(converged)} converged, "
        f"{np.count_nonzero(fell_backward)} fell backward, "
        f"{np.count_nonzero(~converged & ~fell_backward)} timed out."
    )
    if np.any(~converged):
        failed_velocities = np.rad2deg(
            np.asarray(rollout_result.theta_dot_values)[~converged]
        )
        print(
            "Failed initial velocities (deg/s): "
            f"{np.array2string(failed_velocities, precision=3)}"
        )

    steps_fig = plot_steps_to_stability(steps_result, rollout_result)
    steps_path = output / "steps_to_stability.png"
    steps_fig.savefig(steps_path)
    policy_map_path = output / "policy_map.npy"
    np.save(policy_map_path, policy_map)
    print(f"Saved {steps_path} and {policy_map_path}.")


def run_initial_state_steps(args, params, output):
    NUM_THETAS = 200
    NUM_THETA_DOTS = 800
    MAX_FROUDE = 2.0
    SIM_TIME = 5.0
    TIMESTEP = 1e-2
    SMALL_TIMESTEP = 1e-4
    NUM_TIMESTEPS = round(SIM_TIME / TIMESTEP)

    initial_alpha = params["angle_of_attack_bounds"][0]
    theta_values = jnp.linspace(
        params["incline"] - jnp.pi / 2,
        params["incline"] + initial_alpha,
        NUM_THETAS,
    )
    theta_dot_values = jnp.linspace(
        jnp.deg2rad(-100.0),
        jnp.sqrt(MAX_FROUDE * params["gravity"] / params["length"]),
        NUM_THETA_DOTS,
    )
    policy_map = jnp.asarray(np.load(args.map_file, allow_pickle=False))
    roa_bounds = jnp.asarray(np.load(args.roa_bounds_file, allow_pickle=False))

    start = time.perf_counter()
    result = compute_initial_state_steps(
        params,
        theta_values,
        theta_dot_values,
        policy_map,
        roa_bounds,
        TIMESTEP,
        SMALL_TIMESTEP,
        NUM_TIMESTEPS,
    )
    jax.block_until_ready(result)
    end = time.perf_counter()
    print(f"Computed initial-state steps in {end - start}s")
    print(
        f"Simulated states: {result.steps_grid.size}; "
        f"maximum footsteps: {np.max(result.steps_grid)}"
    )

    steps_fig = plot_initial_state_steps(result)
    steps_path = output / "initial_state_steps.png"
    steps_fig.savefig(steps_path)
    print(f"Saved {steps_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate the inverted pendulum walker"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    trajectory_parser = subparsers.add_parser(
        "trajectory", help="animate a trajectory and plot its energy"
    )
    trajectory_parser.add_argument(
        "--map-file",
        type=Path,
        default=Path("output/assignment_2/policy_map.npy"),
        help="policy map generated by lookup-table",
    )
    trajectory_parser.add_argument(
        "--controller",
        choices=("standing", "lookup", "minimum-alpha"),
        required=True,
        help="controller to simulate",
    )
    trajectory_parser.add_argument(
        "--roa-bounds-file",
        type=Path,
        default=Path("output/assignment_2/roa_bounds.npy"),
        help="RoA bounds generated by the roa command",
    )
    subparsers.add_parser(
        "roa", help="plot the upright controller's region of attraction"
    )
    lookup_table_parser = subparsers.add_parser(
        "lookup-table",
        help="compute minimum steps to the ankle controller's RoA and save the optimal alphas",
    )
    lookup_table_parser.add_argument(
        "--roa-bounds-file",
        type=Path,
        default=Path("output/assignment_2/roa_bounds.npy"),
        help="RoA bounds generated by the roa command",
    )
    initial_state_steps_parser = subparsers.add_parser(
        "initial-state-steps",
        help="plot estimated steps to stability over the full initial-state grid",
    )
    initial_state_steps_parser.add_argument(
        "--map-file",
        type=Path,
        default=Path("output/assignment_2/policy_map.npy"),
        help="policy map generated by lookup-table",
    )
    initial_state_steps_parser.add_argument(
        "--roa-bounds-file",
        type=Path,
        default=Path("output/assignment_2/roa_bounds.npy"),
        help="RoA bounds generated by the roa command",
    )
    args = parser.parse_args()

    params = {
        "gravity": 9.81,  # m/s^2
        "length": 1.0,  # m
        "mass": 1.0,  # kg
        "incline": 0.06,  # rad
        "angle_of_attack_bounds": np.array([np.pi / 8, np.pi / 7]),  # rad
        "ankle_torque_bounds": np.array([-0.1 * 9.81, 0.05 * 9.81]),  # N m
        "ankle_torque_damping": 5.0,
    }
    output = Path("output/assignment_2")
    output.mkdir(parents=True, exist_ok=True)

    command_functions = {
        "trajectory": lambda: run_trajectory(args, params, output),
        "roa": lambda: run_roa(params, output),
        "lookup-table": lambda: run_lookup_table(args, params, output),
        "initial-state-steps": lambda: run_initial_state_steps(args, params, output),
    }
    command_functions[args.command]()

    plt.show()


if __name__ == "__main__":
    main()
