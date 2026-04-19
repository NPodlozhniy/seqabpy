import numpy as np
import pytest

from seqabpy.gavi import AlwaysValidInference, sequential_p_value


def test_sequential_p_value():
    assert sequential_p_value([100, 101], [0.5, 0.5]) == 1
    assert sequential_p_value([100, 201], [0.5, 0.5]) < 1e-5


def test_sequential_p_value_custom_dirichlet():
    # With custom dirichlet_alpha, equal counts still yield p=1
    assert sequential_p_value([100, 100], [0.5, 0.5], dirichlet_alpha=[50, 50]) == 1
    assert sequential_p_value([100, 100], [0.5, 0.5], dirichlet_alpha=[1, 1]) == 1


def test_sequential_p_value_three_groups():
    # Equal counts across 3 groups should not be significant
    assert sequential_p_value([100, 100, 100], [1/3, 1/3, 1/3]) == 1


def test_avi():
    avi = AlwaysValidInference(np.arange(10, 100, 10), 1, 1)
    assert (
        avi.GAVI(50)
        == np.array([False, True, True, True, True, True, True, True, True])
    ).all()
    assert (
        avi.mSPRT(0.08)
        == np.array([False, False, True, True, True, True, True, True, True])
    ).all()
    assert (
        avi.StatSig_SPRT()
        == np.array([False, True, True, True, True, True, True, True, True])
    ).all()
    assert (
        avi.statsig_alpha_corrected_v1(100)
        == np.array([False, False, False, True, True, True, True, True, True])
    ).all()


def test_avi_two_sided():
    avi = AlwaysValidInference(np.arange(10, 100, 10), 1, 1, sides=2)
    # Two-sided should be less powerful (wider CI) than one-sided
    assert (
        avi.GAVI(50)
        == np.array([False, True, True, True, True, True, True, True, True])
    ).all()
    assert (
        avi.mSPRT(0.08)
        == np.array([False, False, True, True, True, True, True, True, True])
    ).all()
    assert (
        avi.StatSig_SPRT()
        == np.array([False, True, True, True, True, True, True, True, True])
    ).all()
    assert (
        avi.statsig_alpha_corrected_v1(100)
        == np.array([False, False, False, False, True, True, True, True, True])
    ).all()


def test_avi_default_phi():
    avi = AlwaysValidInference(np.arange(10, 100, 10), 1, 1)
    # Default phi (uses max sample size for GAVI, computed from estimate for mSPRT)
    assert (
        avi.GAVI()
        == np.array([False, True, True, True, True, True, True, True, True])
    ).all()
    assert (
        avi.mSPRT()
        == np.array([False, True, True, True, True, True, True, True, True])
    ).all()


def test_avi_default_N():
    avi = AlwaysValidInference(np.arange(10, 100, 10), 1, 1)
    # Default N uses max sample size
    assert (
        avi.statsig_alpha_corrected_v1()
        == np.array([False, False, False, True, True, True, True, True, True])
    ).all()


def test_avi_invalid_sides():
    with pytest.raises(ValueError, match="Sides must be 1 or 2"):
        AlwaysValidInference(np.arange(10, 100, 10), 1, 1, sides=3)


def test_avi_no_effect():
    # With zero effect, nothing should be significant
    avi = AlwaysValidInference(np.arange(10, 100, 10), 1, 0)
    assert not avi.GAVI(50).any()
    assert not avi.mSPRT(0.08).any()
    assert not avi.StatSig_SPRT().any()
    assert not avi.statsig_alpha_corrected_v1(100).any()
