"""
Optimizer registry — import all algorithm classes and expose a factory function.
"""

from .abc import ABCOptimizer
from .kh  import KHOptimizer
from .mbo import MBOOptimizer
from .ewa import EWAOptimizer
from .eho import EHOOptimizer
from .ms  import MSOptimizer
from .sma import SMAOptimizer
from .hho import HHOOptimizer

REGISTRY = {
    "ABC": ABCOptimizer,
    "KH":  KHOptimizer,
    "MBO": MBOOptimizer,
    "EWA": EWAOptimizer,
    "EHO": EHOOptimizer,
    "MS":  MSOptimizer,
    "SMA": SMAOptimizer,
    "HHO": HHOOptimizer,
}


def get_optimizer(name: str, **kwargs):
    """Return an instantiated optimizer by algorithm name (case-insensitive)."""
    key = name.upper()
    if key not in REGISTRY:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[key](**kwargs)


__all__ = list(REGISTRY.keys()) + ["REGISTRY", "get_optimizer"]
