"""mhd_fno — Fourier Neural Operator surrogate for 2D incompressible MHD instabilities.

Subpackages:
    solver      pseudo-spectral 2D incompressible MHD solver (ground truth)
    data        dataset generation, storage, and PyTorch data loading
    models      FNO and baseline architectures
    training    training loops, losses, checkpointing
    evaluation  physics diagnostics (growth rate, spectra, conservation, threshold)
    utils       spectral helpers, configuration, logging
"""

__version__ = "0.0.1"
