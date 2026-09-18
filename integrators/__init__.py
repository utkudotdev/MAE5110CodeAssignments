from typing import Callable
import numpy.typing as npt
import numpy as np


ContinuousDynamics = Callable[[float, npt.NDArray], npt.NDArray]
Integrator = Callable[[ContinuousDynamics, float, npt.NDArray, float], npt.NDArray]


def explicit_euler(
    f: ContinuousDynamics, t: float, state: npt.NDArray, dt: float
) -> npt.NDArray:
    return state + dt * f(t, state)


def rk4(f: ContinuousDynamics, t: float, state: npt.NDArray, dt: float) -> npt.NDArray:
    k1 = f(t, state)
    k2 = f(t + dt / 2, state + k1 * dt / 2)
    k3 = f(t + dt / 2, state + k2 * dt / 2)
    k4 = f(t + dt, state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
