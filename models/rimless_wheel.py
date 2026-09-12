import numpy as np
import numpy.typing as npt

from integrators import Integrator


def discrete_dynamics(
    state: npt.NDArray,
    integrator: Integrator,
    dt: float,
    params,
) -> npt.NDArray:
    alpha = params["alpha"]
    gamma = params["gamma"]
    length = params["length"]

    theta, theta_dot, global_height = state
    new_state = integrator(
        lambda _, x: swing_dynamics(x, params), 0.0, np.array([theta, theta_dot]), dt
    )
    theta_new, theta_dot_new = new_state
    global_height_new = global_height

    triangle_base = 2 * length * np.sin(alpha)

    if theta_new > alpha + gamma:
        theta_new = gamma - alpha
        theta_dot_new = theta_dot * np.cos(2 * alpha)
        global_height_new -= triangle_base * np.sin(gamma)
    elif theta_new < gamma - alpha:
        theta_new = alpha + gamma
        theta_dot_new = theta_dot * np.cos(2 * alpha)
        global_height_new += triangle_base * np.sin(gamma)

    return np.array([theta_new, theta_dot_new, global_height_new])


def swing_dynamics(state: npt.NDArray, params):
    gravity = params["gravity"]
    length = params["length"]

    theta, theta_dot = state
    theta_double_dot = gravity * np.sin(theta) / length

    return np.array([theta_dot, theta_double_dot])


def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "length": 1,  # rod length (m)
        "mass": 1,  # point mass at end of rod (kg)
        "alpha": np.deg2rad(360.0 / 12) / 2.0,
        "gamma": np.deg2rad(20.0),
    }
    return params


def calculate_angular_momentum(states: npt.NDArray, params):
    mass = params["mass"]
    length = params["length"]
    return mass * length**2 * states[1]


def calculate_energy(states: npt.NDArray, params):
    gravity = params["gravity"]
    mass = params["mass"]
    length = params["length"]

    theta, theta_dot, global_height = states

    inertia = mass * length**2
    kinetic_energy = 0.5 * inertia * theta_dot**2
    potential_energy = mass * gravity * (global_height + length * np.cos(theta))

    return kinetic_energy, potential_energy
