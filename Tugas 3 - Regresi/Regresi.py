import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Membaca data dari dataset
data_path = "Tugas 3 - Regresi/Student_Performance.csv"
df = pd.read_csv(data_path)

# Pilih kolom yang relevan
df = df[["Sample Question Papers Practiced", "Performance Index"]]
df.columns = ["NL", "NT"]

# Definisikan variabel independen dan dependen
X = df["NL"].values.reshape(-1, 1)
y = df["NT"].values

# Inisialisasi model linear
linear_model = LinearRegression()

# Fit model
linear_model.fit(X, y)

# Prediksi nilai NT
y_pred_linear = linear_model.predict(X)

# Hitung galat RMS untuk Model Linear
rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
print(f"RMSE untuk Model Linear: {rmse_linear}")


# Definisikan fungsi eksponensial
def exp_func(x, a, b):
    return a * np.exp(b * x)


# Fit model eksponensial
popt, _ = curve_fit(exp_func, df["NL"], df["NT"])

# Prediksi nilai NT
y_pred_exp = exp_func(df["NL"], *popt)

# Hitung galat RMS untuk Model Eksponensial
rmse_exp = np.sqrt(mean_squared_error(y, y_pred_exp))
print(f"RMSE untuk Model Eksponensial: {rmse_exp}")

# Plot hasil regresi linear dan eksponensial
plt.scatter(df["NL"], df["NT"], label="Data Asli", color="blue")
plt.plot(df["NL"], y_pred_linear, label="Regresi Linear", color="red")
plt.plot(df["NL"], y_pred_exp, label="Regresi Eksponensial", color="green")
plt.xlabel("Jumlah Latihan Soal (NL)")
plt.ylabel("Nilai Ujian (NT)")
plt.legend()
plt.title("Regresi Linear dan Eksponensial terhadap Nilai Ujian")
plt.show()
