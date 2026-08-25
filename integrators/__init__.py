from typing import Callable
import numpy.typing as npt


def explicit_euler(
    f: Callable[[float, float], float], t: float, state: float, dt: float
) -> float:
    return state + dt * f(t, state)


def rk4(f: Callable[[float, float], float], t: float, state: float, dt: float) -> float:
    k1 = f(t, state)
    k2 = f(t + dt / 2, state + k1 * dt / 2)
    k3 = f(t + dt / 2, state + k2 * dt / 2)
    k4 = f(t + dt, state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
