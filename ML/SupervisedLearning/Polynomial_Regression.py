

import numpy as np 
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


X = 4 * np.random.rand(100, 1)
y = 2 + 3*X ** 2 + 0.5 * np.random.rand(100, 1) # y = 2 + 3x^2

# plt.scatter(X, y)

poly_feat = PolynomialFeatures(degree = 2)
X_poly = poly_feat.fit_transform(X)


poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)


plt.scatter(X, y, color = "blue", label = "actual data")

X_test = np.linspace(0, 4, 100).reshape(-1, 1)
X_test_poly = poly_feat.transform(X_test)
y_pred = poly_reg.predict(X_test_poly)




plt.plot(X_test, y_pred, color = "red", label = "model output")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Regression")

plt.show()
plt.legend()











