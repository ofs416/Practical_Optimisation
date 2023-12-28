import numpy as np
from numpy.linalg import inv

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize': (14, 8.27)}, font_scale=2)
sns.set_palette('colorblind')

def f(A, b, x_til, lam, t):
    n =256
    x, u = x_til[:n], x_til[n:]
    if (u-x < 0).any() or (u+x < 0).any():
        return 'infeasible'
    residual = A @ x - b
    return t*residual.T @ residual + t*lam*np.sum(u) - np.sum(np.log(u-x) + np.log(u+x))

## Gradient

def grad_f_x(A, b, x_til, t):
    n =256
    x, u = x_til[:n], x_til[n:]
    gradient = 2*t*(A.T @ A @ x - A.T @ b) + 1/(u - x) - 1/(u + x)
    return gradient

def grad_f_u(x_til, t, lam):
    n =256
    x, u = x_til[:n], x_til[n:]
    gradient = t*lam - 1/(u - x) - 1/(u + x)
    return gradient

def grad_f(A, b, x_til, t, lam):
    return np.concatenate((grad_f_x(A, b, x_til, t), grad_f_u(x_til, t, lam)))

## Hessian

def grad_f_xx(A, x_til, t):
    n =256
    x, u = x_til[:n], x_til[n:]
    gradient = 2*t*A.T @ A + np.diag(1/np.square(u + x) + 1/np.square(u - x))
    return gradient

def grad_f_xu(x_til):
    n =256
    x, u = x_til[:n], x_til[n:]
    gradient = np.diag(1/np.square(u + x) - 1/np.square(u - x))
    return gradient

def grad_f_uu(x_til):
    n =256
    x, u = x_til[:n], x_til[n:]
    gradient = np.diag(1/np.square(u + x) + 1/np.square(u - x))
    return gradient

def hessian_f(A, x_til, t):
    h_11 = grad_f_xx(A, x_til, t)
    h_12 = grad_f_xu(x_til)
    h_22 = grad_f_uu(x_til)
    return np.block([[h_11, h_12],[h_12, h_22]])

A = np.genfromtxt('2023-data/A.csv', delimiter=",")
x_0 = np.genfromtxt('2023-data/x.csv', delimiter=",")

b = A @ x_0 + np.random.uniform(-0.005, 0.005, A.shape[0])

plt.plot(x_0)
plt.show()

def N_decrement(A, b, x_til, t, lam):
    jacobian = grad_f(A, b, x_til, t, lam)
    return (jacobian.T @ np.linalg.inv(hessian_f(A, x_til, t)) @ jacobian)/2
#
def Newtons_method(x_til_start, epsilon, t, lam_factor):
    lam_max = np.linalg.norm(2 * A.T @ b, ord=np.inf)
    lam = lam_factor * lam_max
    beta = 0.8
    iterations = 0
    x_til = x_til_start
    func = f(A, b, x_til, lam, t)
    costs = [func]
    grad = grad_f(A, b, x_til, t, lam)
    hessian = hessian_f(A, x_til, t)
    step = - np.linalg.inv(hessian) @ grad

    print(f't: {t}, Iteration: {iterations}, func: {func}')

    while (grad.T @ - step)/2 >= epsilon:
        t_b = 1
        while f(A, b, x_til + t_b * step, lam, t) == 'infeasible':
            t_b = t_b * beta
        x_til = x_til + t_b * step
        iterations += 1
        grad = grad_f(A, b, x_til, t, lam)
        hessian = hessian_f(A, x_til, t)
        step = - np.linalg.inv(hessian) @ grad
        func = f(A, b, x_til, lam, t)
        costs.append(func)
        print(f't: {t}, Iteration: {iterations}, func: {func}')
    return x_til, costs

x_til = np.concatenate((np.zeros(256), np.ones(256)))
ep = 0.1
for lam_factor in [0.01]:
    costs = []
    for t in [1, 10, 100, 1000]:
        x_til, cost = Newtons_method(x_til, ep, t, lam_factor)
        costs = costs + cost

plt.plot(x_til_opt[:256])
plt.show()

x_til_opt = np.concatenate((np.zeros(256), np.ones(256)))
ep = 0.1
x_opt = []
for lam_factor in [10, 1, 0.5, 0.1, 0.05,  0.01, 0.005]:
    for t in [1, 10]:
        x_til_opt, cost = Newtons_method(x_til_opt, ep, t, lam_factor)
    x_opt.append(x_til_opt[:256])

plt.plot(x_opt[4])
plt.show()