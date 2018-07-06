# interpolation_uncertainty

Interpolates data known up to uncertainties and provides uncertainty bands for the interpolated function as well as its first derivative.

Notes:
 - for now limited to a *cubic spline with natural boundary conditions*.
 - assumes statistically independent input data.
 - only interpolation, extrapolation omitted


Disclaimer: the implementation works but is only tested for small datasets and a medium number of interpolation points (tests up to ~200). The implementation should work above that but nothing is known about the performance. 
