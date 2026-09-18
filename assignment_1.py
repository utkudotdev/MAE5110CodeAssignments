import argparse
import enum
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from tqdm import tqdm

from integrators import rk4 as integrator
from models import rimless_wheel as model


class Attractor(enum.Enum):
    ROLLING = 0
    STABLE = 1
    UNKNOWN = 2


@dataclass
class AttractorMapResult:
    theta_values: np.ndarray
    theta_dot_values: np.ndarray
    attractor_grid: np.ndarray


@dataclass
class ReturnMapResult:
    current_velocities: np.ndarray
    next_velocities: np.ndarray
    fixed_points: np.ndarray
    floquet_multipliers: np.ndarray


def main():
    parser = argparse.ArgumentParser(description="Simulate the rimless wheel")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("attractors", help="plot the attractor map")
    subparsers.add_parser("trajectory", help="plot a single simulation")
    subparsers.add_parser("return", help="plot the return map")
    subparsers.add_parser(
        "sweep-inclinations", help="sweep inclinations and save plots"
    )
    subparsers.add_parser("sweep-spokes", help="sweep spoke count and save plots")
    args = parser.parse_args()

    params = model.generate_params()
    large_timestep = 1e-2
    small_timestep = 1e-3

    if args.command == "attractors":
        result = compute_attractors(
            params, large_timestep, small_timestep, sim_time=5.0
        )
        plot_attractors(result)
        plt.show()
    elif args.command == "trajectory":
        initial_state = np.array([np.deg2rad(20.0), np.deg2rad(1.0), 0.0])
        time_traj, state_traj = simulate(
            initial_state,
            params,
            large_timestep,
            small_timestep,
            sim_time=5.0,
        )
        plot_traj(time_traj, state_traj, params)
        plt.show()
    elif args.command == "return":
        result = compute_return_map(
            params, large_timestep, small_timestep, sim_time=5.0
        )
        print_return_map_results(result)
        plot_return_map(result)
        plt.show()
    elif args.command == "sweep-inclinations":
        inclinations = [15.0, 25.0, 35.0, 45.0, 55.0]
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    compute_sweep_case,
                    params,
                    "gamma",
                    np.deg2rad(inclination),
                    large_timestep,
                    small_timestep,
                    7.0,
                ): inclination
                for inclination in inclinations
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                inclination = futures[future]
                return_map, attractors = future.result()
                floquet_multipliers = return_map.floquet_multipliers
                tqdm.write(f"{inclination=} {floquet_multipliers=}")

                return_fig = plot_return_map(return_map)
                return_fig.savefig(f"results/inclination_{int(inclination)}_return.png")
                plt.close(return_fig)
                attractor_fig = plot_attractors(attractors)
                attractor_fig.savefig(
                    f"results/inclination_{int(inclination)}_attractors.png"
                )
                plt.close(attractor_fig)
    elif args.command == "sweep-spokes":
        spoke_counts = [6, 7, 8, 9, 10, 11, 12]
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    compute_sweep_case,
                    params,
                    "alpha",
                    np.deg2rad(360.0 / spokes) / 2.0,
                    large_timestep,
                    small_timestep,
                    7.0,
                ): spokes
                for spokes in spoke_counts
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                spokes = futures[future]
                return_map, attractors = future.result()
                floquet_multipliers = return_map.floquet_multipliers
                tqdm.write(f"{spokes=} {floquet_multipliers=}")

                return_fig = plot_return_map(return_map)
                return_fig.savefig(f"results/spokes_{spokes}_return.png")
                plt.close(return_fig)
                attractor_fig = plot_attractors(attractors)
                attractor_fig.savefig(f"results/spokes_{spokes}_attractors.png")
                plt.close(attractor_fig)


def compute_sweep_case(
    params,
    parameter_name,
    parameter_value,
    large_timestep,
    small_timestep,
    sim_time,
):
    sweep_params = params.copy()
    sweep_params[parameter_name] = parameter_value
    return_map = compute_return_map(
        sweep_params,
        large_timestep,
        small_timestep,
        sim_time,
        show_progress=False,
    )
    attractors = compute_attractors(
        sweep_params,
        large_timestep,
        small_timestep,
        sim_time,
        show_progress=False,
    )
    return return_map, attractors


def compute_return_map(
    params, large_timestep, small_timestep, sim_time, show_progress=True
) -> ReturnMapResult:
    # Since we're most interested in collecting theta-dot transitions here,
    # we don't need to sample over theta. We can just start the wheel right before impact.
    THETA_DOT_MIN, THETA_DOT_MAX = np.deg2rad(-500.0), np.deg2rad(500.0)
    NUM_THETA_DOT = 300

    FLOQUET_PERTURBATION = np.deg2rad(1.0)
    NUM_FLOQUET_PERTURBATIONS = 5
    FLOQUET_DT = 5e-4  # use more accurate simulation for floquet estimation
    BISECTION_DT = 1e-3

    current_velocities = []
    next_velocities = []

    for theta_dot in tqdm(
        np.linspace(THETA_DOT_MIN, THETA_DOT_MAX, NUM_THETA_DOT),
        leave=False,
        disable=not show_progress,
    ):
        # get_next_pre_impact_velocity automatically places the wheel at impact depending on the direction
        next_theta_dot = get_next_pre_impact_velocity(
            theta_dot, params, large_timestep, small_timestep, sim_time
        )

        current_velocities.append(theta_dot)
        next_velocities.append(next_theta_dot)

    current_velocities = np.array(current_velocities)
    next_velocities = np.array(next_velocities)

    sort_args = np.argsort(current_velocities)
    current_velocities = current_velocities[sort_args]
    next_velocities = next_velocities[sort_args]

    deltas = next_velocities - current_velocities

    zero_crossings = np.where(np.diff(deltas >= 0))[0]
    after_crossing = zero_crossings + 1

    # Filter out discontinous branch changes. Not the most robust method, but good enough for our purposes.
    DISCONTINUITY_THRESH = np.deg2rad(10.0)
    continuous = (
        np.abs(next_velocities[after_crossing] - next_velocities[zero_crossings])
        < DISCONTINUITY_THRESH
    )
    zero_crossings = zero_crossings[continuous]
    after_crossing = after_crossing[continuous]

    def f(theta_dot):
        return (
            get_next_pre_impact_velocity(
                theta_dot, params, BISECTION_DT, BISECTION_DT, sim_time
            )
            - theta_dot
        )

    fixed_points = []
    for lower_idx, upper_idx in zip(zero_crossings, after_crossing):
        lower = current_velocities[lower_idx]
        lower_value = next_velocities[lower_idx] - lower
        upper = current_velocities[upper_idx]
        upper_value = next_velocities[upper_idx] - upper

        zero_point = bisection_method(f, lower, lower_value, upper, upper_value)
        fixed_points.append(zero_point)

    fixed_points = np.asarray(fixed_points)
    perturbation_multiples = np.arange(
        -NUM_FLOQUET_PERTURBATIONS, NUM_FLOQUET_PERTURBATIONS + 1
    )
    floquet_multipliers = []
    for fixed_point in fixed_points:
        perturbed_velocities = (
            fixed_point + perturbation_multiples * FLOQUET_PERTURBATION
        )
        perturbed_next_velocities = np.array(
            [
                get_next_pre_impact_velocity(
                    velocity, params, FLOQUET_DT, FLOQUET_DT, sim_time
                )
                for velocity in perturbed_velocities
            ]
        )
        floquet_multipliers.append(
            np.polyfit(perturbed_velocities, perturbed_next_velocities, deg=1)[0]
        )

    return ReturnMapResult(
        current_velocities=current_velocities,
        next_velocities=next_velocities,
        fixed_points=fixed_points,
        floquet_multipliers=np.asarray(floquet_multipliers),
    )


def print_return_map_results(result: ReturnMapResult):
    if not result.fixed_points.size:
        print("No fixed points found.")
        return

    fixed_points_deg = np.rad2deg(result.fixed_points)
    print(f"Fixed-point estimates: {fixed_points_deg} deg/s")
    print(f"Floquet multipliers: {result.floquet_multipliers}")


def plot_return_map(result: ReturnMapResult):
    fig, ax = plt.subplots()
    current_velocities_deg = np.rad2deg(result.current_velocities)
    next_velocities_deg = np.rad2deg(result.next_velocities)
    ax.scatter(
        current_velocities_deg,
        next_velocities_deg,
        color="#3a86ff",
        s=10,
        alpha=0.75,
    )

    velocity_min = min(min(current_velocities_deg), min(next_velocities_deg))
    velocity_max = max(max(current_velocities_deg), max(next_velocities_deg))

    velocity_padding = 0.05 * (velocity_max - velocity_min)
    plot_min = velocity_min - velocity_padding
    plot_max = velocity_max + velocity_padding

    ax.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        "k--",
        label="Identity",
    )
    if result.fixed_points.size:
        fixed_points_deg = np.rad2deg(result.fixed_points)
        ax.scatter(
            fixed_points_deg,
            fixed_points_deg,
            color="#e71d36",
            edgecolor="black",
            marker="X",
            s=120,
            label="Fixed points",
            zorder=3,
        )
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_title("Rimless Wheel Return Map")
    ax.set_xlabel(r"Pre-impact velocity $\dot{\theta}_k$ (deg/s)")
    ax.set_ylabel(r"Next pre-impact velocity $\dot{\theta}_{k+1}$ (deg/s)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def compute_attractors(
    params, large_timestep, small_timestep, sim_time, show_progress=True
) -> AttractorMapResult:
    alpha, gamma = params["alpha"], params["gamma"]

    THETA_MIN, THETA_MAX = gamma - alpha, alpha + gamma
    NUM_THETA = 40
    THETA_DOT_MIN, THETA_DOT_MAX = np.deg2rad(-500.0), np.deg2rad(100.0)
    NUM_THETA_DOT = 400

    theta_values = np.linspace(THETA_MIN, THETA_MAX, NUM_THETA)
    theta_dot_values = np.linspace(THETA_DOT_MIN, THETA_DOT_MAX, NUM_THETA_DOT)
    attractor_grid = np.empty((NUM_THETA_DOT, NUM_THETA), dtype=int)

    for theta_idx, theta in tqdm(
        enumerate(theta_values),
        total=NUM_THETA,
        leave=False,
        disable=not show_progress,
    ):
        for theta_dot_idx, theta_dot in tqdm(
            enumerate(theta_dot_values),
            total=NUM_THETA_DOT,
            leave=False,
            disable=not show_progress,
        ):
            initial_state = np.array([theta, theta_dot, 0.0])
            _, state_traj = simulate(
                initial_state, params, large_timestep, small_timestep, sim_time
            )
            attractor = classify_attractor(state_traj)
            attractor_grid[theta_dot_idx, theta_idx] = attractor.value

    return AttractorMapResult(
        theta_values=theta_values,
        theta_dot_values=theta_dot_values,
        attractor_grid=attractor_grid,
    )


def plot_attractors(result: AttractorMapResult):
    colors = {
        Attractor.ROLLING: "#e63946",
        Attractor.STABLE: "#2a9d8f",
        Attractor.UNKNOWN: "#8338ec",
    }

    fig, ax = plt.subplots()
    color_map = ListedColormap([colors[attractor] for attractor in Attractor])
    color_norm = BoundaryNorm(np.arange(len(Attractor) + 1) - 0.5, color_map.N)
    ax.pcolormesh(
        np.rad2deg(result.theta_values),
        np.rad2deg(result.theta_dot_values),
        result.attractor_grid,
        cmap=color_map,
        norm=color_norm,
        shading="nearest",
    )

    legend_handles = [
        Patch(color=colors[attractor], label=attractor.name.title())
        for attractor in Attractor
        if np.any(result.attractor_grid == attractor.value)
    ]
    ax.legend(
        handles=legend_handles,
        title="Attractor",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    ax.set_title("Rimless Wheel Attractors")
    ax.set_xlabel(r"Initial angle $\theta$ (deg)")
    ax.set_ylabel(r"Initial angular velocity $\dot{\theta}$ (deg/s)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def simulate(initial_state, params, large_timestep, small_timestep, sim_time):
    theta = initial_state[0]
    alpha, gamma = params["alpha"], params["gamma"]
    assert (gamma - alpha) <= theta <= (alpha + gamma)

    times = [0.0]
    states = [np.array(initial_state, copy=True)]

    while times[-1] < sim_time:
        adaptive_timestep = get_timestep_for_state(
            states[-1], params, large_timestep, small_timestep
        )
        adaptive_timestep = min(adaptive_timestep, sim_time - times[-1])
        next_state = model.discrete_dynamics(
            states[-1], integrator, adaptive_timestep, params
        )
        times.append(times[-1] + adaptive_timestep)
        states.append(next_state)

    return np.asarray(times), np.column_stack(states)


def get_timestep_for_state(state, params, large_timestep, small_timestep):
    alpha, gamma = params["alpha"], params["gamma"]
    upper_limit = alpha + gamma
    lower_limit = gamma - alpha
    total_range = upper_limit - lower_limit
    assert total_range > 0

    theta, theta_dot, _ = state

    if theta_dot >= 0:
        distance = upper_limit - theta
    else:
        distance = theta - lower_limit

    assert distance >= 0
    progress = (total_range - distance) / total_range
    return progress * small_timestep + (1 - progress) * large_timestep


def classify_attractor(state_traj) -> Attractor:
    STABLE_THRESH = 1e-4
    ROLLING_THRESH = 0.05

    angular_velocity = state_traj[1]

    if np.abs(angular_velocity[-1]) < STABLE_THRESH:
        return Attractor.STABLE
    else:
        pre_impact_steps = get_pre_impact_steps(state_traj)
        pointcare_velocities = angular_velocity[pre_impact_steps].flatten()
        diff = np.diff(pointcare_velocities)
        if np.abs(diff[-1]) < ROLLING_THRESH:
            return Attractor.ROLLING
        else:
            return Attractor.UNKNOWN


IMPACT_DETECTION_THRESH = 1e-6


def get_pre_impact_steps(state_traj):
    global_height = state_traj[2]
    height_diff = np.diff(global_height)
    pre_impact_steps = np.argwhere(np.abs(height_diff) > IMPACT_DETECTION_THRESH)
    return pre_impact_steps


def get_next_pre_impact_velocity(
    pre_impact_velocity,
    params,
    large_timestep,
    small_timestep,
    max_sim_time,
):
    alpha, gamma = params["alpha"], params["gamma"]
    theta = alpha + gamma if pre_impact_velocity >= 0.0 else gamma - alpha
    initial_state = np.array([theta, pre_impact_velocity, 0.0])

    # We will rewrite the simulation loop here, because we only have to simulate until the next impact
    time = 0.0
    state = initial_state
    impact_counter = 0

    while time < max_sim_time:
        adaptive_timestep = get_timestep_for_state(
            state, params, large_timestep, small_timestep
        )
        adaptive_timestep = min(adaptive_timestep, max_sim_time - time)
        next_state = model.discrete_dynamics(
            state, integrator, adaptive_timestep, params
        )

        height = state[2]
        next_height = next_state[2]

        if abs(next_height - height) > IMPACT_DETECTION_THRESH:
            impact_counter += 1
            if impact_counter >= 2:
                break

        state = next_state
        time += adaptive_timestep

    return state[1]


def bisection_method(f, lower, lower_value, upper, upper_value, epsilon=1e-5):
    previous_value = None
    while True:
        middle = (lower + upper) / 2
        middle_value = f(middle)
        if previous_value is not None and abs(middle_value - previous_value) < epsilon:
            return middle

        lower_sign = math.copysign(1.0, lower_value)
        middle_sign = math.copysign(1.0, middle_value)
        upper_sign = math.copysign(1.0, upper_value)

        if middle_sign != lower_sign:
            upper = middle
            upper_value = middle_value
        elif middle_sign != upper_sign:
            lower = middle
            lower_value = middle_value
        else:
            assert False, "what"

        previous_value = middle_value


def plot_traj(time_traj, state_traj, params):
    angle = state_traj[0]
    angular_momentum = model.calculate_angular_momentum(state_traj, params)
    kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)

    ax1.plot(time_traj, angle, label="Angle")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle (rad)")

    ax2.plot(time_traj, angular_momentum, label="Angular momentum")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angular momentum")

    ax3.plot(time_traj, kinetic_energy, label="Kinetic energy")
    ax3.plot(time_traj, potential_energy, label="Potential energy ")
    ax3.plot(time_traj, kinetic_energy + potential_energy, label="Total energy")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Energy")

    fig.legend()
    return fig


if __name__ == "__main__":
    main()
