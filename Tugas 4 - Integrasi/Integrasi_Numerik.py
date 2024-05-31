import numpy as np
import time


# Fungsi untuk menghitung nilai integral dengan metode trapezoid
def trapezoid_integration(f, a, b, N):
    x = np.linspace(a, b, N + 1)
    y = f(x)
    h = (b - a) / N
    integral = h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))
    return integral


# Fungsi yang akan diintegrasikan
def f(x):
    return 4 / (1 + x**2)


# Nilai referensi pi
pi_ref = 3.14159265358979323846

# Variasi nilai N
N_values = [10, 100, 1000, 10000]

# Untuk menyimpan hasil
results = []

# Pengujian untuk setiap nilai N
for N in N_values:
    start_time = time.time()
    pi_approx = trapezoid_integration(f, 0, 1, N)
    end_time = time.time()
    execution_time = end_time - start_time
    error_rms = np.sqrt((pi_approx - pi_ref) ** 2)
    results.append((N, pi_approx, error_rms, execution_time))

# Output hasil
for result in results:
    N, pi_approx, error_rms, execution_time = result
    print(
        f"N = {N}, Pi Approx = {pi_approx}, Error RMS = {error_rms}, Execution Time = {execution_time} seconds"
    )
