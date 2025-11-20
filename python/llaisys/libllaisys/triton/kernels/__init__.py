"""Package for Triton kernels.

This package contains individual kernel modules (e.g. `add.py`, `linear.py`).
An explicit `__init__` ensures the package is importable when installed.
"""

# Expose common kernels when imported as `from ...triton.kernels import add`
# (modules are imported by `setup_kernels.py` via `from .kernels import add`)
__all__ = []
