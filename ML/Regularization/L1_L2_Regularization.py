

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error


diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42)



# Ridge Regression
ridge = Ridge()
ridge_param_grid = {"alpha": [0.1, 1, 10, 100]}


ridge_grid_search = GridSearchCV(ridge, ridge_param_grid, cv = 5)
ridge_grid_search.fit(X_train, y_train)


print("Ridge best parameters: ", ridge_grid_search.best_params_)
print("Ridge best score: ", ridge_grid_search.best_score_)


best_ridge_model = ridge_grid_search.best_estimator_
y_pred = best_ridge_model.predict(X_test)

ridge_mse = mean_squared_error(y_test, y_pred)
print("Ridge mse: ", ridge_mse)


# Lasso
lasso = Lasso()
lasso_param_grid = {"alpha": [0.1, 1, 10, 100]}

lasso_grid_search = GridSearchCV(lasso, lasso_param_grid, cv = 5)
lasso_grid_search.fit(X_train, y_train)

print("Lasso best parameters: ", lasso_grid_search.best_params_)
print("Lasso best score: ", lasso_grid_search.best_score_)


best_lasso_model = lasso_grid_search.best_estimator_
y_pred_lasso = mean_squared_error(y_test, y_pred)
lasso_mse = mean_squared_error(y_test, y_pred)
print("Lasso mse: ", lasso_mse)




































