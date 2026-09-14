from collections.abc import Callable
from typing import Protocol, Self

import numpy.typing as npt

type ContinuousDynamics[TArray] = Callable[[float, TArray], TArray]
type Integrator[TArray] = Callable[
    [ContinuousDynamics[TArray], float, npt.NDArray, float], TArray
]


class ArrayLike(Protocol):
    def __add__(self, other: Self, /) -> Self: ...
    def __mul__(self, other: float, /) -> Self: ...
    def __rmul__(self, other: float, /) -> Self: ...
    def __truediv__(self, other: float, /) -> Self: ...


def explicit_euler[TArray: ArrayLike](
    f: ContinuousDynamics[TArray], t: float, state: TArray, dt: float
) -> TArray:
    return state + dt * f(t, state)


def rk4[TArray: ArrayLike](
    f: ContinuousDynamics[TArray], t: float, state: TArray, dt: float
) -> TArray:
    k1 = f(t, state)
    k2 = f(t + dt / 2, state + k1 * dt / 2)
    k3 = f(t + dt / 2, state + k2 * dt / 2)
    k4 = f(t + dt, state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
