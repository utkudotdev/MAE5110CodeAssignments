from collections.abc import Callable
from typing import Protocol, Self

import numpy.typing as npt

type ContinuousDynamics[TArray] = Callable[[float, TArray, dict], TArray]
type Integrator[TArray] = Callable[
    [ContinuousDynamics[TArray], float, npt.NDArray, float, dict], TArray
]


class ArrayLike(Protocol):
    def __add__(self, other: Self, /) -> Self: ...
    def __mul__(self, other: float, /) -> Self: ...
    def __rmul__(self, other: float, /) -> Self: ...
    def __truediv__(self, other: float, /) -> Self: ...


def explicit_euler[TArray: ArrayLike](
    f: ContinuousDynamics[TArray], t: float, state: TArray, dt: float, params: dict
) -> TArray:
    return state + dt * f(t, state, params)


def rk4[TArray: ArrayLike](
    f: ContinuousDynamics[TArray], t: float, state: TArray, dt: float, params: dict
) -> TArray:
    k1 = f(t, state, params)
    k2 = f(t + dt / 2, state + k1 * dt / 2, params)
    k3 = f(t + dt / 2, state + k2 * dt / 2, params)
    k4 = f(t + dt, state + dt * k3, params)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
