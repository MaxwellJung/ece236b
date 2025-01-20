import numpy as np
import cvxpy as cp
import math

import illumdata
from illumdata import A
m, n = A.shape
b = np.ones((m,1))

# Saturated LS
from scipy.linalg.lapack import dgels
lqr, x, info = dgels(A,b)
x = x[:n]
x1 = np.maximum(0.0, np.minimum(x,1.0))
print("Solution 1: %.4f" %(np.max(np.abs(np.log(A@x1)))))

# Weighted LS
rho = 0.2190
AA = np.vstack([A, math.sqrt(rho) * np.identity(n)])
bb = np.vstack([b, 0.5 * math.sqrt(rho) * np.ones((n,1))])
lqr, x, info = dgels(AA,bb)
x2 = x[:n]
print("Solution 2: %.4f" %(np.max(np.abs(np.log(A@x2)))))

# Chebyshev
x = cp.Variable(n)
objective = cp.Minimize(cp.norm(A@x - b, "inf"))
constraints = [0 <= x, x <= 1]
cp.Problem(objective, constraints).solve()
x3 = x.value
print("Solution 3: %.4f" %(np.max(np.abs(np.log(A@x3)))))

# Exact
x = cp.Variable(n)
objective = cp.Minimize(cp.maximum(cp.max(A@x), cp.max(cp.inv_pos(A@x))))
constraints = [0 <= x, x <= 1]
cp.Problem(objective, constraints).solve()
x4 = x.value
print("Solution 4: %.4f" %(np.max(np.abs(np.log(A@x4)))))