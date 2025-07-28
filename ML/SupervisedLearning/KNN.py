# sklearn: ML Lib
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib.pyplot as plt

""" #1- Dataset Analysis """

# Loading the dataset into the program
cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns= cancer.feature_names)

# Add the target variable column to the df
df["target"] = cancer.target

X = cancer.data
y = cancer.target

""" #2- Train Test Split """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 42)


""" #3- Training the Model """
# Call KNN
knn = KNeighborsClassifier(n_neighbors=3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# fit method trains the ML model by using the features and target variables
knn.fit(X_train, y_train)

""" #4- Prediction and Evaluating Results """
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(conf_mat)


""" #5- Hyperparameter Tuning """


accuracy_val = []
k_values = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors= k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_val.append(accuracy)
    k_values.append(k)


# Visualizing the k parameters' affects to the model performance
plt.figure()
plt.plot(k_values, accuracy_val, marker = "o", linestyle = "-")
plt.title("Accuracy Subject to K Values")
plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)


# %%
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor


X = np.sort(5 * np.random.rand(40, 1), axis = 0) # uniform
y = np.sin(X).ravel()

plt.scatter(X, y)

# Add noise
y[::5] += 1 * (0.5 - np.random.rand(8))

T = np.linspace(0, 5, 500)[:, np.newaxis]


for i, weight in enumerate(["uniform", "distance"]):    
    knn =  KNeighborsRegressor(n_neighbors= 5, weights= weight)
    y_pred = knn.fit(X, y).predict(T)
    
    
    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color = "green", label = "data")
    plt.plot(T, y_pred, color = "blue", label = "prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN Regressor Weights {}".format(weight))    

plt.tight_layout()
plt.show()

























