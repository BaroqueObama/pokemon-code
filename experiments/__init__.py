"""Pokemon Theorem experiments package.

Import torch early to avoid BLAS/OpenMP conflicts with sklearn on macOS.
"""

import torch as _torch  # noqa: F401 -- must be imported before sklearn
