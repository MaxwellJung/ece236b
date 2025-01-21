import numpy as np
import cvxpy as cp
from matplotlib import pyplot
from bsplines import bsplines
from spline_data import t, y

N = len(t)
A = np.zeros((N,13))
G = np.zeros((N,13))
for k in range(N):
    g, gp, gpp = bsplines(t[k])
    A[k,:] = g
    G[k,:] = gpp
x = cp.Variable(13)
objective = cp.Minimize(cp.norm(A@x - y, 2))
constraints = [-G@x <= 0]
cp.Problem(objective, constraints).solve()
xls = x.value

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