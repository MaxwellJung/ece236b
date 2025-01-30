import numpy as np
import cvxpy as cp
from matplotlib import pyplot

# setup vars
N = 30
A = np.matrix(
    [[-1, 0.4, 0.8],
    [1, 0, 0],
    [0, 1, 0]]
)
B = np.array([1, 0, 0.3])
X = np.squeeze(np.array([np.linalg.matrix_power(A, i)@B for i in range(N-1, -1, -1)])).T
x_des = np.array([7, 2, -6])
u = cp.Variable(N)
f = cp.maximum(cp.abs(u), 2*cp.abs(u)-1)

objective = cp.Minimize(cp.sum(f))
constraints = [X@u == x_des]
cp.Problem(objective, constraints).solve()
xls = u.value
print(xls)

# plot solution
pyplot.step(np.arange(N), u.value)
pyplot.xlabel('time t')
pyplot.ylabel('actual signal u(t)')
pyplot.show()