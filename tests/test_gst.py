import numpy as np

from seqabpy.gatsby import *

def test_calculate_sequential_bounds():
    expected_bounds = (
        np.array([-3.021, -1.415, -0.596, -0.057, 0.351, 0.682, 0.964, 1.212, 1.434, 1.795]),
        np.array([6.088, 4.229, 3.396, 2.906, 2.579, 2.342, 2.16, 2.015, 1.895, 1.795]),
    )
    calculated_bounds = calculate_sequential_bounds(
        np.linspace(1 / 10, 1, 10), alpha=0.05, beta=0.2
    )
    for row in range(2):
        assert (calculated_bounds[row].round(3) == expected_bounds[row]).all()


def test_ldBounds(alpha=0.025):
    expected_bounds = np.array(
        [
            [3.929, 2.67, 1.981],
        ]
    )
    result = ldBounds(t=np.array([0.3, 0.6, 1.0]), alpha=alpha)
    assert abs(result["overall.alpha"] - alpha) < 1e-5
    assert (result["upper.bounds"].round(3) == expected_bounds).all()


def test_gst(alpha=0.025):
    expected_oversampling = np.array(
        [
            [3.929, 2.67, 1.989],
        ]
    )
    expected_undersampling = np.array(
        [
            [3.929, 2.67, 1.969],
        ]
    )
    assert (
        GST(
            actual=np.array([0.3, 0.6, 1.2]),
            expected=np.array([0.3, 0.6, 1]),
            alpha=alpha,
        ).round(3)
        == expected_oversampling
    ).all()
    assert (
        GST(
            actual=np.array([0.3, 0.6, 0.8]),
            expected=np.array([0.3, 0.6, 1]),
            alpha=alpha,
        ).round(3)
        == expected_undersampling
    ).all()
