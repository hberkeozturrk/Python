

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np


oli = fetch_olivetti_faces()

# 2D Image --> 1D 


# plt.figure()
# for i in range(2):
#     plt.subplot(1, 2, i + 1)
#     plt.imshow(oli.images[i + 320], cmap = "gray")
#     plt.axis("off")
# plt.show()


X = oli.data
y = oli.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)


rf_clf = RandomForestClassifier(n_estimators= 100, random_state= 42)
rf_clf.fit(X_train, y_train)


y_pred = rf_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")


est_count = np.arange(10, 110, 10) 
accuracies = []
for i in range(len(est_count)):
    rf_clf = RandomForestClassifier(n_estimators= est_count[i], random_state= 42)
    rf_clf.fit(X_train, y_train)


    y_pred = rf_clf.predict(X_test)

    accuracies.append(accuracy_score(y_test, y_pred))


plt.figure()
plt.plot(est_count, accuracies, marker = "o")
plt.xticks(est_count)
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy")
plt.legend()


# %% 

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import numpy as np 


california_housing = fetch_california_housing()

X = california_housing.data
y = california_housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)
rf_reg = RandomForestRegressor(random_state= 42)
rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("rmse:", rmse)











