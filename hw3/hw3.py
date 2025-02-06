import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt

# setup vars
n = 40
theta_tar = 15
delta = 15
N = 400
x = 30 * np.random.rand(n)
y = 30 * np.random.rand(n)
theta = np.linspace(-180, 180, N)
theta_sidelobe = theta[np.abs(theta - theta_tar) >= delta]
gamma_sidelobe = np.stack([np.cos(theta_sidelobe/180*np.pi), np.sin(theta_sidelobe/180*np.pi)], axis=1) @ np.stack([x,y], axis=1).T
A = np.exp(gamma_sidelobe*1j)
gamma_tar = np.stack([np.cos(theta_tar/180*np.pi), np.sin(theta_tar/180*np.pi)]) @ np.stack([x,y], axis=1).T
gamma_all = np.stack([np.cos(theta/180*np.pi), np.sin(theta/180*np.pi)], axis=1) @ np.stack([x,y], axis=1).T

w = cp.Variable(n, complex=True)
objective = cp.Minimize(cp.max(cp.abs(A@w)))
constraints = [np.exp(gamma_tar*1j)@w == 1]
cp.Problem(objective, constraints).solve()
print(w.value)

abs_g_theta = np.abs(np.exp(gamma_all*1j)@w.value)

# plot solution
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta/180*np.pi, abs_g_theta)
plt.show()

plt.plot(theta, abs_g_theta)
plt.show()