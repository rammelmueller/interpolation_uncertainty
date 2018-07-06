""" ---------------------------------------------------------------------------

    test_spline.py - LR, July 2018

--------------------------------------------------------------------------- """
import numpy as np
from scipy.interpolate import CubicSpline
from cs_uncert import CubicSplineUncertainty


def _dummy_set():
    """ provides a test-set for all calculations.
    """
    x = np.array([-1, 0, 1, 2, 7, 9, 12.5, 16, 20, 24, 25, 28, 31])
    y = np.array([0.5, 0, 3, 5, 1, 6, 3, 3, 2, 1, 4, 5, 8])
    sig = np.array([0.4, 0.45, .375, 0.55, 0.39, 0.51, 0.35, 0.91, 0.35, 0.55, 0.39, 0.51, 0.35])
    return x, y, sig


def test_spline():
    """ checks whether the interpolated function agrees with scipy.
    """
    n_points = 100
    x, y, sig = _dummy_set()
    csu = CubicSplineUncertainty(x, y, sig)
    x_spline = np.linspace(np.min(x), np.max(x), n_points)

    # version to test.
    y_spline, _ = csu.interpolate(x_spline, order=0)

    # scipy.
    cs = CubicSpline(x, y, bc_type='natural')
    y_scipy = cs(x_spline)

    assert(np.allclose(y_spline, y_scipy))


def test_spline_derivative():
    """ check whether the first derivative agrees with scipy.
    """
    n_points = 100
    x, y, sig = _dummy_set()
    csu = CubicSplineUncertainty(x, y, sig)
    x_spline = np.linspace(np.min(x), np.max(x), n_points)

    # version to test.
    yp_spline, _ = csu.interpolate(x_spline, order=1)

    # scipy.
    cs = CubicSpline(x, y, bc_type='natural')
    yp_scipy = cs(x_spline, 1)

    assert(np.allclose(yp_spline, yp_scipy))


def test_uncertainties():
    """ checks whether the uncertainties at the input x-values are correct (i.e.
        the input uncertainties).
    """
    x, y, sig = _dummy_set()
    csu = CubicSplineUncertainty(x, y, sig)

    # version to test.
    _, sigp_spline = csu.interpolate(x, order=0)

    assert(np.allclose(sigp_spline, sig))
