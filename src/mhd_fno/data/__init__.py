"""Dataset generation, storage, and PyTorch data loading.

Planned modules:
    sweep.py       Latin-Hypercube parameter sampling (M_A, Re) with the test hole
    generate.py    run the solver over the sweep, save (omega, A) trajectories to HDF5
    dataset.py     torch Dataset / DataLoader yielding (field(t) -> field(t+dt)) pairs
"""
