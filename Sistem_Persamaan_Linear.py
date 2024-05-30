import numpy as np
from scipy.linalg import lu


def solve_by_matrix_inverse(A, b):
    """
    Solves the system of linear equations Ax = b using the matrix inverse method.

    Parameters:
    A (numpy.ndarray): Coefficient matrix
    b (numpy.ndarray): Constant vector

    Returns:
    numpy.ndarray: Solution vector x
    """
    A_inv = np.linalg.inv(A)
    x = np.dot(A_inv, b)
    return x


def solve_by_lu_decomposition(A, b):
    """
    Solves the system of linear equations Ax = b using LU decomposition method.

    Parameters:
    A (numpy.ndarray): Coefficient matrix
    b (numpy.ndarray): Constant vector

    Returns:
    numpy.ndarray: Solution vector x
    """
    P, L, U = lu(A)
    y = np.linalg.solve(L, np.dot(P, b))
    x = np.linalg.solve(U, y)
    return x


def crout_decomposition(A):
    """
    Performs Crout's LU decomposition of matrix A.

    Parameters:
    A (numpy.ndarray): Coefficient matrix

    Returns:
    tuple: Lower and Upper triangular matrices, L and U
    """
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)

    for j in range(n):
        L[j, j] = 1
        for i in range(j + 1):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for i in range(j, n):
            L[i, j] = (A[i, j] - sum(L[i, k] * U[k, j] for k in range(j))) / U[j, j]

    return L, U


def solve_by_crout_decomposition(A, b):
    """
    Solves the system of linear equations Ax = b using Crout's decomposition method.

    Parameters:
    A (numpy.ndarray): Coefficient matrix
    b (numpy.ndarray): Constant vector

    Returns:
    numpy.ndarray: Solution vector x
    """
    L, U = crout_decomposition(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)
    return x


if __name__ == "__main__":
    # Testing
    A = np.array([[3, 2], [1, 2]])
    b = np.array([5, 5])

    # Matrix Inverse Method
    x_inv = solve_by_matrix_inverse(A, b)
    print(f"Solution using matrix inverse method: {x_inv}")

    # LU Decomposition Method
    x_lu = solve_by_lu_decomposition(A, b)
    print(f"Solution using LU decomposition method: {x_lu}")

    # Crout Decomposition Method
    x_crout = solve_by_crout_decomposition(A, b)
    print(f"Solution using Crout decomposition method: {x_crout}")
