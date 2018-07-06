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
        self.h = np.diff(self.x)
        self.y = y
        self.var = sig**2
        self.N = len(x)

        self.bounds = (np.min(x), np.max(x))

        # Solve the spline.
        self._solve_spline()
        self._calculate_jacobian()



    def interpolate(self, x_grid, order=0):
        """ Takes in a 1D array x and returns interpolated y-values along with
            uncertainties.
        """
        # check if the requested grid is within bounds.
        if np.max(x_grid) > self.bounds[1]:
            raise ValueError('upper bound exceeded - extrapolation not possible.')
        if np.min(x_grid) < self.bounds[0]:
            raise ValueError('lower bound exceeded - extrapolation not possible.')

        if order > 1:
            raise NotImplementedError('order of derivative not implemented.')


        # bins & coefficients.
        b = np.zeros_like(x_grid, dtype=int)
        c = np.zeros(shape=(len(x_grid),4))
        for n, x in enumerate(x_grid):

            # calculate the appropriate bin.
            b[n] = max(len(self.x[x>self.x])-1, 0)
            i = b[n]
            h = self.x[i+1] - self.x[i]

            # coefficients for the spline.
            c[n,0] = (self.x[i+1] - x) / h
            c[n,1] = 1. - c[n,0]
            c[n,2] = (c[n,0]**3 - c[n,0]) * h**2 / 6.
            c[n,3] = (c[n,1]**3 - c[n,1]) * h**2 / 6.


        # coefficients for the derivative.
        if order == 1:
            cp = np.zeros(shape=(len(x_grid),4))

            for n, x in enumerate(x_grid):
                i = b[n]
                h = self.x[i+1] - self.x[i]

                cp[n,0] = -1./h
                cp[n,1] = 1./h
                cp[n,2] = h/6. * (1. - 3*c[n,0]**2)
                cp[n,3] = h/6. * (3*c[n,1]**2 - 1.)


        # interpolated function.
        if order == 0:
            y_grid = np.zeros_like(x_grid)
            for n in range(len(x_grid)):
                i = b[n]
                y_grid[n] = np.dot(c[n,:], [self.y[i], self.y[i+1], self.s[i], self.s[i+1]])


            # uncertainty band.
            cov = np.zeros(shape=(len(x_grid),len(x_grid)))
            for m in range(len(x_grid)):
                i = b[m]
                for n in range(len(x_grid)):
                    j = b[n]

                    for k in range(self.N):
                        l = np.dot([i==k, i==(k-1), self.jac[i,k], self.jac[i+1,k]], c[m,:])
                        r = np.dot([j==k, j==(k-1), self.jac[j,k], self.jac[j+1,k]], c[n,:])
                        cov[m,n] += l * r * self.var[k]


        # derivative of interpolated function.
        elif order == 1:
            y_grid = np.zeros_like(x_grid)
            for n in range(len(x_grid)):
                i = b[n]
                y_grid[n] = np.dot(cp[n,:], [self.y[i], self.y[i+1], self.s[i], self.s[i+1]])

            # uncertainty band.
            cov = np.zeros(shape=(len(x_grid),len(x_grid)))
            for m in range(len(x_grid)):
                i = b[m]
                for n in range(len(x_grid)):
                    j = b[n]

                    for k in range(self.N):
                        l = np.dot([i==k, i==(k-1), self.jac[i,k], self.jac[i+1,k]], cp[m,:])
                        r = np.dot([j==k, j==(k-1), self.jac[j,k], self.jac[j+1,k]], cp[n,:])
                        cov[m,n] += l * r * self.var[k]

        self.cov = cov
        return y_grid, np.sqrt(np.diag(self.cov))


    def _solve_spline(self):
        """ Solves the spline.
        """

        # construct the system of equations.
        b = np.diff(np.diff(self.y)/self.h)
        a = np.zeros(shape=(len(self.y),len(self.y)))

        for k in range(1, self.N-1):
            a[k,k] = (self.x[k+1] - self.x[k-1]) / 3.
            if k>1:
                a[k,k-1] = (self.x[k] - self.x[k-1]) / 6.
            if k<len(self.x)-2:
                a[k,k+1] = (self.x[k+1] - self.x[k]) / 6.

        # solve for the second derivatives (tridiagonal matrix).
        self.s = np.zeros(self.N)

        # invert the matrix and multiply.
        self.eta = np.linalg.inv(a[1:-1,1:-1])
        self.s[1:-1] = self.eta.dot(b)


    def _calculate_jacobian(self):
        """ Calculates the jacobian for the second derivatives.
        """
        jac = np.zeros(shape=(len(self.x),len(self.x)))
        for i in range(0, self.N-2):
            for j in range(0, self.N):
                if j >= 2:
                    jac[i+1,j] += self.eta[i,j-2]/self.h[j-1]
                if j >= 1 and j < self.N-1:
                    jac[i+1,j] += self.eta[i,j-1] * (1./self.h[j] - 1./self.h[j-1])
                if j < self.N-2:
                    jac[i+1,j] += self.eta[i,j]/self.h[j]
        self.jac = jac
