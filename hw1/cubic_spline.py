import numpy as np
from matplotlib import pyplot
from scipy.linalg.lapack import dgels
from bsplines import bsplines
from spline_data import t, y

N = len(t)
A = np.zeros((N,13))
for k in range(N):
    g, gp, gpp = bsplines(t[k])
    A[k,:] = g
lqr, x, info = dgels(A, y)
xls = x[:13]

# plot solution
npts = 1000
pts = np.linspace(0, 10, npts)
f = np.zeros(npts)
for k in range(npts):
    g, gp, gpp = bsplines(pts[k])
    f[k] = np.vecdot(g, xls)
pyplot.plot(pts, f, 'b')
pyplot.plot(t, y, 'og')
pyplot.show()