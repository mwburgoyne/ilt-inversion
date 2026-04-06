"""
Fixed Talbot method for numerical inverse Laplace transforms.

Delegates to mpmath.invertlaplace with method='talbot'.  This is a good
choice for well-behaved (non-oscillatory) transforms where double precision
or moderate mpmath precision suffices.
"""

from typing import Any, Callable

from mpmath import invertlaplace


def talbot_single(fn: Callable[..., Any], time: float, degree: int = 32) -> Any:
    """Invert fn at a single time point using Fixed Talbot."""
    return invertlaplace(fn, time, method='talbot', degree=degree)
