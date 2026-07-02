"""Tests for the parameter sweep design."""

from mhd_fno.data.sweep import generate_sweep


def test_counts_and_splits():
    specs = generate_sweep(n_train=50, n_test=10)
    assert len(specs) == 60
    assert sum(s.split == "train" for s in specs) == 50
    assert sum(s.split == "test" for s in specs) == 10


def test_train_avoids_hole_and_test_is_inside():
    specs = generate_sweep(n_train=200, n_test=40, M_A_hole=(1.5, 2.5))
    for s in specs:
        if s.split == "train":
            assert not (1.5 < s.M_A < 2.5)         # training avoids the hole
        else:
            assert 1.5 <= s.M_A <= 2.5             # test lives in the hole
        assert 0.5 <= s.M_A <= 6.0
        assert 500.0 <= s.Re <= 5000.0


def test_reproducible():
    a = generate_sweep(n_train=20, n_test=5, seed=3)
    b = generate_sweep(n_train=20, n_test=5, seed=3)
    assert [s.as_dict() for s in a] == [s.as_dict() for s in b]


def test_unique_ids_and_seeds():
    specs = generate_sweep(n_train=100, n_test=20)
    assert len({s.run_id for s in specs}) == len(specs)
    assert len({s.seed for s in specs}) == len(specs)
