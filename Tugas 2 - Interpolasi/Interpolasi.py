import numpy as np
import matplotlib.pyplot as plt

# Data yang diberikan
x = np.array([5, 10, 15, 20, 25, 30, 35, 40])
y = np.array([40, 30, 25, 40, 18, 20, 22, 15])


# Fungsi untuk interpolasi polinomial Lagrange
def lagrange_interpolation(x_points, y_points, x):
    def L(k, x):
        term = [
            (x - x_points[i]) / (x_points[k] - x_points[i])
            for i in range(len(x_points))
            if i != k
        ]
        return np.prod(term, axis=0)

    P = np.sum([y_points[k] * L(k, x) for k in range(len(x_points))], axis=0)
    return P


# Fungsi untuk interpolasi polinomial Newton
def newton_interpolation(x_points, y_points, x):
    n = len(x_points)
    divided_diff = np.zeros((n, n))
    divided_diff[:, 0] = y_points

    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (
                divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]
            ) / (x_points[i + j] - x_points[i])

    def N(k, x):
        term = [(x - x_points[i]) for i in range(k)]
        return np.prod(term, axis=0)

    P = divided_diff[0, 0] + np.sum(
        [divided_diff[0, k] * N(k, x) for k in range(1, n)], axis=0
    )
    return P


# Rentang x untuk plot
x_values = np.linspace(5, 40, 400)

# Interpolasi dengan metode Lagrange dan Newton
y_lagrange = lagrange_interpolation(x, y, x_values)
y_newton = newton_interpolation(x, y, x_values)

# Plot hasil interpolasi
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_lagrange, label="Lagrange Interpolation", color="blue")
plt.plot(x_values, y_newton, label="Newton Interpolation", color="green")
plt.scatter(x, y, color="red", label="Data Points")
plt.xlabel("Tegangan (kg/mm^2)")
plt.ylabel("Waktu Patah (jam)")
plt.title("Interpolasi Polinomial Menggunakan Metode Lagrange dan Newton")
plt.legend()
plt.grid(True)
plt.show()
