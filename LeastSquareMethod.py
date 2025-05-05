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

#theta = (XᵀX)⁻¹Xᵀy

theta = np.linalg.inv(x.T @ x) @ (x.T) @ (y)

y_pred = x @ (theta)

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