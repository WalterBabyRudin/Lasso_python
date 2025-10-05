import cvxpy as cp
import numpy as np
import time
def my_CVXPY_solver(A, b, mu):
    # Goal: Solve the optimization problem
    #   minimize ||x||_1 + (1/(2μ))||Ax-b||^2
    #   subject to x >= 0
    # Input: 
    #   A: m x n matrix
    #   b: m-dimensional vector
    #   mu: positive scalar
    # Output:
    #   x: n-dimensional vector

    # Define variable
    x = cp.Variable(n, nonneg=True)
    # Define objective
    objective = cp.norm1(x) + (1/(2*mu)) * cp.sum_squares(A @ x - b)
    # Form and solve problem
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve()
    return x.value

def my_dual_ADMM_solver(A, b, mu, beta=1, gamma=1, max_iter=10, tol=1e-4, silent=False):
    # Goal: Solve the optimization problem
    #   minimize ||x||_1 + (1/(2μ))||Ax-b||^2
    #   subject to x >= 0
    # Iteratively,
    # z^{k+1} = Proj_{F}(A.T @ y^k + (1/beta) * x^k), whre F = {z | z <= 1}
    # y^{k+1} = y^k - alpha^k * g^k, 
    # where g^k = mu*y^k + A@x^k - b + beta*A@(A.T@y^k - z^{k+1})
    # alpha^k = (g^k.T @ g^k) / (g^k.T @ (mu*I + beta*A@A.T) @ g^k)
    # x^{k+1} = x^k - beta * gamma * (z^{k+1} - A.T @ y^{k+1})
    # Input: 
    #   A: m x n matrix
    #   b: m-dimensional vector
    #   mu: positive scalar
    #   beta: penalty parameter for ADMM
    #   gamma: over-relaxation parameter for ADMM
    #   max_iter: maximum number of iterations
    #   tol: tolerance for stopping criterion
    # Output:
    #   x: n-dimensional vector

    m, _ = A.shape
    x = A.T @ b  # Initialize x
    y = np.zeros(m)
    ATy = A.T @ y
    beta_gamma = beta * gamma

    for k in range(max_iter):
        
        # z-update
        z = ATy + (1/beta) * x
        z = np.minimum(z, 1)  # Projection onto F = {z | z <= 1}

        # y-update
        g = mu * y + A @ x - b + beta * A @ (ATy - z)
        gnorm2 = np.sum(g**2)
        ATg = A.T @ g
        alpha = gnorm2 / (gnorm2 * mu + beta * np.sum(ATg**2))
        y = y - alpha * g
        ATy = A.T @ y

        # x-update
        x_old = x.copy()
        x = x - beta_gamma * (z - ATy)

        # Check convergence
        r_norm = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + 1e-6)
        if not silent:
            if k % 100 == 0 or k == max_iter - 1:
                objval = np.linalg.norm(x, 1) + (1/(2*mu)) * np.linalg.norm(A @ x - b)**2
                print(f'Iteration {k}, Residual norm: {r_norm:.2e}, Objective value: {objval:.2e}')
            if r_norm < tol:
                print(f'Converged in {k} iterations.')
                break

    return x


if __name__ == "__main__":
    # Example usage and testing
    np.random.seed(0)

    n = 8192
    m = int(n/5)

    for trials in range(5):
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        mu = 1.0  # parameter
        
        time_start = time.time()
        x = my_CVXPY_solver(A, b, mu)
        time_end = time.time()
        print(f"Trial {trials+1}: CVXPY solver time: {time_end - time_start:.2f} seconds")
        objective_value = np.linalg.norm(x, 1) + (1/(2*mu)) * np.linalg.norm(A @ x - b)**2
        print("Optimal objective value:", objective_value)



        beta = np.sum(np.abs(b)) / m
        time_start = time.time()
        x_admm = my_dual_ADMM_solver(A, b, mu, beta=beta, gamma=1.168, max_iter=10000, tol=1e-3,silent=False)
        objective_value_admm = np.linalg.norm(x_admm, 1) + (1/(2*mu)) * np.linalg.norm(A @ x_admm - b)**2
        print("ADMM Optimal objective value:", objective_value_admm)
        time_end = time.time()
        print(f"Trial {trials+1}: ADMM solver time: {time_end - time_start:.2f} seconds")
