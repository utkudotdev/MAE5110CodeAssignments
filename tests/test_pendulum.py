import numpy as np
 
from integrators import rk4
from models import pendulum
 
DT = 0.01
STEPS = 100
 
 
def simulate(params, state):
    """Step the pendulum forward with RK4 and return every state, shape (STEPS + 1, 2)."""
    states = [state]
    for step in range(STEPS):
        state = rk4(pendulum.dynamics, step * DT, state, DT, params)
        states.append(state)
    return np.array(states)
 
 
def total_energy(states, params):
    kinetic, potential = pendulum.calculate_energy(states.T, params)
    return kinetic + potential
 
 
def test_energy_conserved_without_damping_or_torque():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0
    state = np.array([0.5, 0.0])  # not an equilibrium, so the pendulum moves
 
    energy = total_energy(simulate(params, state), params)
 
    assert np.all(np.isclose(np.diff(energy), 0.0, atol=1e-6))
 
 
def test_damping_removes_energy():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.5
    params["torque"] = 0.0
    state = np.array([0.5, 0.0])
 
    energy = total_energy(simulate(params, state), params)
 
    # Damping can only remove energy: it never goes up, and overall it drops.
    assert np.all(np.diff(energy) <= 1e-9)
    assert energy[-1] < energy[0]
 
 
def test_torque_does_work():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 2.0
    state = np.array([0.5, 0.0])
 
    states = simulate(params, state)
    energy = total_energy(states, params)
 
    # A constant torque adds energy equal to the work it does: torque * change in angle.
    work = params["torque"] * (states[:, 0] - states[0, 0])
    assert np.all(np.isclose(energy - energy[0], work, atol=1e-6))