"""-----------------------------------------------------------------------------

    cs_uncert.py - LR, July 2018

    Class that implements a cubic spline interpolation with propagation of
    uncertainty of input values according to [1, 2].


    input:  x, y, sig
    Takes (assumedly uncorrelated) input data with known uncertainties
    (standard devation).

    ouput:  y, sig
    Produces interpolated values y along with the propagated uncertainty
    (standard deviation).


    [1] W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery.
        Numerical Recipes 3rd Edition: The Art of Scientific Computing (2007).

    [2] J.L Gardner.
        J. Res. Natl. Inst. Stand. Technol. 108:69-78 (2003).

-----------------------------------------------------------------------------"""
import numpy as np
import scipy.linalg as la


class CubicSplineUncertainty(object):
    """ Interpolates given data and provides uncertainty bands.

        Notes:
            -   assumes statistically independent input data, i.e. the
                covariance matrix of the input values is diagonal.

    """
    def __init__(self, x, y, sig):
        """ Calculates spline and covariance of input data.
        """
        self.x = x
        self.y = y
        self.sig = sig

        self.bounds = (np.min(x), np.max(x))

        # Solve the spline.
        self._calculate_coefficients()
        self._calculate_covariance()



    def interpolate(self, x_grid):
        """ Takes in a 1D array x and returns interpolated y-values along with
            uncertainties.
        """
        # check if the requested grid is within bounds.
        if np.max(x_grid) > self.bounds[1]:
            raise ValueError('upper bound exceeded - extrapolation not possible.')
        if np.min(x_grid) < self.bounds[0]:
            raise ValueError('lower bound exceeded - extrapolation not possible.')

        # interpolated function.
        y_grid = np.zeros_like(x_grid)
        for k, x in enumerate(x_grid):

            # find left end of current box.
            i = max(len(self.x[x>self.x])-1, 0)

            # calculate coefficients.
            h = self.x[i+1] - self.x[i]
            A = (self.x[i+1] - x) / h
            B = 1. - A
            C = (A**3 - A) * h**2 / 6.
            D = (B**3 - B) * h**2 / 6.

            # produce the interpolated value.
            y_grid[k] = A*self.y[i] + B*self.y[i+1] + C*self.s[i] + D*self.s[i+1]


        # uncertainty band.
        cov = np.zeros(shape=(len(x_grid),len(x_grid)))
        # TODO.


        self.cov = cov
        return y_grid, np.sqrt(np.diag(self.cov))


    def _calculate_coefficients(self):
        """ Solves the spline.
        """

        # construct the system of equations.
        self.h = np.diff(self.x)
        b = np.diff(np.diff(self.y)/self.h)
        a = np.zeros(shape=(len(self.y),len(self.y)))

        for k in range(1, len(self.y)-1):
            a[k,k] = (self.x[k+1] - self.x[k-1]) / 3.
            if k>1:
                a[k,k-1] = (self.x[k] - self.x[k-1]) / 6.
            if k<len(self.x)-2:
                a[k,k+1] = (self.x[k+1] - self.x[k]) / 6.

        # solve for the second derivatives (tridiagonal matrix).
        self.s = np.zeros(len(self.y))

        # invert the matrix and multiply.
        # self.s[1:-1] = np.linalg.inv(a[1:-1,1:-1]).dot(b)

        # alternatively, solve the system of equations a*x = b for x
        self.s[1:-1] = la.solve(a[1:-1,1:-1], b)


    def _calculate_covariance(self):
        """ Calculates the covariance matrices to propagate the uncertainty.
        """
        pass
