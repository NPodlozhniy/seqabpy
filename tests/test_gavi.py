import numpy as np

from seqabpy.gavi import *


def test_sequential_p_value():
    assert sequential_p_value([100, 101], [0.5, 0.5]) == 1
    assert sequential_p_value([100, 201], [0.5, 0.5]) < 1e-5


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
