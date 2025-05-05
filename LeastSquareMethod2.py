import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Linear Regression/advertising.csv")

print(df.head(3))
print(df.info())

print(df.isnull().sum(),"\n")

x = df.drop(columns="Sales",axis=1).values
y = df[["Sales"]].values

x = np.c_[np.ones((x.shape[0],1)),x]

#theta = (xᵀx)⁻¹xᵀy

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def matmul(A, B):
    result = [[0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def identity_matrix(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def inverse(matrix):
    n = len(matrix)
    AM = [row[:] for row in matrix]
    I = identity_matrix(n)

    for fd in range(n): 
        if AM[fd][fd] == 0:
            raise ValueError("Matris terslenemez (sıfır determinant)")

        fd_scalar = AM[fd][fd]
        for j in range(n):
            AM[fd][j] /= fd_scalar
            I[fd][j] /= fd_scalar

        for i in range(n):
            if i == fd:
                continue
            factor = AM[i][fd]
            for j in range(n):
                AM[i][j] -= factor * AM[fd][j]
                I[i][j] -= factor * I[fd][j]

    return I


x_T = transpose(x)
x_T_x = matmul(x_T, x)
x_T_y = matmul(x_T, y)
x_T_x_inv = inverse(x_T_x)
theta = matmul(x_T_x_inv, x_T_y)

y_pred = matmul(x, theta)

print("Theta (Ağırlıklar):")
for row in theta:
    print(row)

print("\nY_Tahmin:")
for row in y_pred:
    print(row)


features = df.drop(columns="Sales").columns

for i, feature in enumerate(features):
    plt.figure(figsize=(8, 5))
    plt.scatter(df[feature], y, color="green", label="Gerçek Satış")
    plt.scatter(df[feature], y_pred, color="magenta", label="Tahmin", alpha=0.5)
    plt.xlabel(f"{feature} Reklam Harcaması (bin $)")
    plt.ylabel("Satışlar (bin adet)")
    plt.title(f"{feature} Harcamasına Göre Satış Tahmini")
    plt.legend()
    plt.grid(True)
    plt.show()