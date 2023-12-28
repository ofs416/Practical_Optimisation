import numpy as np
from numpy.linalg import inv
import timeit
import cvxpy as cp

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(14,8.27)}, font_scale=2)
sns.set_palette('colorblind')

Ab = []
for i in range(5):
    A = np.genfromtxt('2023-data/A' + str(i) + '.csv', delimiter=",")
    b = np.genfromtxt('2023-data/b' + str(i) + '.csv', delimiter=",")
    Ab.append((A, b))

def lp(data, norm, method, iterations):
    times = np.zeros(5)
    residuals = [None] * 5
    opt = np.zeros(5)
    low_bound = np.zeros(5)
    for iter in range(iterations):
        for i, (A,b) in enumerate(data):
            m, n = A.shape
            if norm == 'l1':
                x = cp.Variable(n+m)
                c_til = np.zeros((n + m,))
                c_til[-m:] = 1
                b_til = np.hstack((b, -b))
                I = np.identity(m)
                A_til = np.vstack((np.hstack((A, -I)), np.hstack((-A, -I))))
            elif norm == 'linf':
                x = cp.Variable(n+1)
                c_til = np.zeros((n + 1,))
                c_til[-1:] = 1
                b_til = np.hstack((b, -b))
                A_til = np.vstack((np.hstack((A, -np.ones((m, 1)))), np.hstack((-A, -np.ones((m, 1))))))
            else:
                print('Error: incorrect norm')
                return

            start_time = timeit.default_timer()
            prob = cp.Problem(cp.Minimize(c_til.T@x),
                             [A_til @ x <= b_til])
            prob.solve(solver=cp.SCIPY, scipy_options={"method": method})
            time = timeit.default_timer() - start_time

            lam = cp.Variable(2*m)
            dual_prob = cp.Problem(cp.Maximize(-b_til.T@lam),
                             [A_til.T @ lam == -c_til,  lam >= np.zeros((2*m,))])
            dual_prob.solve(solver=cp.SCIPY, scipy_options={"method": method})

            opt[i] = prob.value
            low_bound[i] = dual_prob.value
            residuals[i] = A @ x.value[:n] - b
            times[i] += time / iterations

    return opt, times, residuals, low_bound

## $l_1$ norm

### i) Simplex Method

l1_smp_opt, l1_smp_times, l1_smp_residuals, l1_smp_low = lp(Ab, 'l1', 'highs-ds', 1)

print(l1_smp_opt)
print(l1_smp_low)
print(l1_smp_times)

### ii) Interior Point Method

l1_ipm_opt, l1_ipm_times, l1_ipm_residuals, l1_ipm_low = lp(Ab, 'l1', 'highs-ipm', 1)

print(l1_ipm_opt)
print(l1_ipm_low)
print(l1_ipm_times)

plt.hist(l1_ipm_residuals[-1], bins=40)
plt.xlabel('Residual')
plt.xlim(-2.5,2.5)
plt.savefig('Figures/l1residual.png', format="png", dpi=800, bbox_inches="tight")
plt.show()
plt.close()

## $l_\infty$ norm

### i) Simplex Method

linf_smp_opt, linf_smp_times, linf_smp_residuals, linf_smp_low = lp(Ab, 'linf', 'highs-ds', 1)

print(linf_smp_opt)
print(linf_smp_low)
print(linf_smp_times)

### ii) Interior Point LP

linf_ipm_opt, linf_ipm_times, linf_ipm_residuals, linf_ipm_low = lp(Ab, 'linf', 'highs-ipm', 1)

print(linf_ipm_opt)
print(linf_ipm_low)
print(linf_ipm_times)

plt.hist(linf_ipm_residuals[-1], bins=40)
plt.xlabel('Residual')
plt.savefig('Figures/linfresidual.png', format="png", dpi=800, bbox_inches="tight")
plt.show()
plt.close()

## $l_2$ norm

l2_lp_times = [0] * 5
l2_lp_opt = [None] * 5
l2_lp_residuals = [None] * 5
iterations = 10
for n in range(iterations):
    for i, (A,b) in enumerate(Ab):

        start_time = timeit.default_timer()
        x = inv(A.T @ A) @ A.T @ b
        time = timeit.default_timer() - start_time

        l2_lp_residuals[i] = A @ x - b

        l2_lp_opt[i] = np.sqrt(l2_lp_residuals[i].T @ l2_lp_residuals[i])
        l2_lp_times[i] += time / iterations