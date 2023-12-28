import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize': (14, 8.27)}, font_scale=2)
sns.set_palette('colorblind')

A = np.genfromtxt('2023-data/A3.csv', delimiter=",")
b = np.genfromtxt('2023-data/b3.csv', delimiter=",")

m, n = A.shape
b = b.reshape((m, 1))
c_til = np.zeros((n + m, 1))
c_til[-m:] = 1
b_til = np.vstack((b, -b))
I = np.identity(m)
A_til = np.vstack((np.hstack((A, -I)), np.hstack((-A, -I))))

def f(A, b, c, x, t):
    if (A @ x - b <= 0).all():
        function = t * c.T @ x - np.sum(np.log(b - A @ x))
    else:
        function = 'infeasible'
    return function

def grad_f(x, A, b, c, t):
    grad = t * c + A.T @ (1 / (b - A @ x))
    return grad

def backtracking(x, A, b, c, grad, t, alpha = 0.2, beta= 0.3):
    t_b = 1
    while f(A, b, c, x - t_b * grad, t) == 'infeasible':
        t_b = beta*t_b
    while f(A, b, c, x - t_b * grad, t) > f(A, b, c, x, t) - alpha * t_b * grad.T @ grad:
        t_b = beta * t_b
    x = x - t_b * grad
    return x

opt_x_t = []
x = c_til
costs = []
for t, epsilon in [(1, 0.001), (10, 0.1), (100, 1)]:
    iteration = 0
    grad = grad_f(x, A_til, b_til, c_til, t)
    while grad.T @ grad >= epsilon:
        x = backtracking(x, A_til, b_til, c_til, grad, t)
        grad = grad_f(x, A_til, b_til, c_til, t)
        func = f(A_til, b_til, c_til, x, t)
        costs.append(func)
        iteration += 1
        if iteration % 500 == 0:
            print(f'Iteration:{iteration},  func: {func}')
        if iteration % 2000 == 0:
            residual = A @ x[:n] - b
            print(np.sum(np.sqrt(residual*residual)))
            opt_x_t.append(x)

residual = A @ opt_x_t[-1][:n] - b

costs = np.asarray(costs)
costs = costs.reshape((costs.shape[0]))

plt.plot(costs)
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.xlim(-5, 17000)
plt.savefig('Figures/ex2_costViterations.png', format="png", dpi=800, bbox_inches="tight")
plt.show()
plt.close()