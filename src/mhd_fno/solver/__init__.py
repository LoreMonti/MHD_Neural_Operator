"""Pseudo-spectral 2D incompressible MHD solver (ground truth data generator).

Evolves vorticity omega = -laplacian(psi) and magnetic flux potential A in a doubly
periodic box using an FFT-based pseudo-spectral method with 2/3 dealiasing. Primitive
fields (v, B) are reconstructed by spectral Laplacian inversion.

Planned modules:
    spectral.py    FFT wavenumber grids, Laplacian / inverse-Laplacian, dealiasing
    mhd2d.py       the MHD2DSolver class (state, RHS, time stepping)
    initial.py     Kelvin-Helmholtz initial conditions (shear layer + perturbation)
"""
