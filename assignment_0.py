import numpy as np
import matplotlib.pyplot as plt
import time

# from models import pendulum as model
from models import ball as model

# from integrators import explicit_euler as integrator
from integrators import rk4 as integrator
from models.ball import generate_params

# Basic simulation of the pendulum
# jk we're doing a spring now

params = generate_params()


# some set-up
initial_state = np.array([10.0, 0.0])

timestep = 1e-2
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

# simulation loop
start = time.perf_counter()
for step, t in enumerate(time_traj[:-1]):
    state_traj[:, step + 1] = integrator(
        lambda ta, s: model.dynamics(ta, s, params), t, state_traj[:, step], timestep
    )
end = time.perf_counter()

print(f"Integrator: {integrator.__name__}")
print(f"dt: {timestep}s")
print(f"Time: {end - start}s")

# sanity check the energies: since there is no actuation, and no damping, total energy should stay
# constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
# a stand-still.

kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

total_energy = potential_energy + kinetic_energy
print(f"Energy difference: {abs(total_energy[-1] - total_energy[0])}")

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Pendulum energy")
plt.legend()
plt.tight_layout()
plt.show()

# TODO: make a phase portrait plot
