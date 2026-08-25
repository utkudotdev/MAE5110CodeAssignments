import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    mass = params["mass"]
    spring_coeff = params["spring_coeff"]
    damping_coeff = params["damping_coeff"]

    y = state[0]
    velocity = state[1]

    if y > 0.0:
        acceleration = -gravity
    else:
        spring_force = -spring_coeff * y
        damper_force = -damping_coeff * velocity
        gravity_force = mass * -gravity
        force = spring_force + damper_force + gravity_force
        acceleration = force / mass

    state_derivative = np.array([velocity, acceleration])

    return state_derivative


def generate_params():
    params = {
        "gravity": 9.81,  # gravity (m/s^2)
        "mass": 1.0,  # point mass at end of rod (kg)
        "spring_coeff": 500.0,  # N/m
        "damping_coeff": 1.0,  # N/(m/s)
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    mass = params["mass"]
    spring_coeff = params["spring_coeff"]

    y = state[0]  # indexes entire row "vectorized" if state is (2, N)
    velocity = state[1]

    kinetic_energy = 0.5 * mass * velocity**2
    gravity_energy = mass * gravity * y
    spring_energy = np.where(y < 0.0, 0.5 * spring_coeff * y**2, 0.0)
    potential_energy = gravity_energy + spring_energy

    return kinetic_energy, potential_energy
