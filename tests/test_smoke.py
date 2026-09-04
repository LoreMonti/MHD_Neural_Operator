"""Smoke tests: the package imports and exposes a version.

Real tests (solver linear-growth validation, Laplacian invertibility, div(B)=0)
come in Phase 1.
"""

import mhd_fno


def test_version():
    assert isinstance(mhd_fno.__version__, str)


def test_subpackages_import():
    import mhd_fno.data
    import mhd_fno.evaluation
    import mhd_fno.models
    import mhd_fno.solver
    import mhd_fno.training
    import mhd_fno.utils  # noqa: F401
