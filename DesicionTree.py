import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
apple = pd.read_csv(r"C:\Users\ASUS\Desktop\Final-Project\AAPL.csv")
print(apple.head())

apple = apple[["Close"]]
futureDays = 120
apple["Prediction"] = apple[["Close"]].shift(-60)

x = np.array(apple.drop(["Prediction"], axis=1))[:-futureDays]
y = np.array(apple["Prediction"])[:-futureDays]

# Future data for prediction
xfuture = apple.drop(["Prediction"], axis=1)[-futureDays:]
xfuture = np.array(xfuture)

# Creating a DataFrame for plotting
valid = apple[-futureDays:].copy()

# Decision Tree Regressor with GridSearchCV
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)
tree = DecisionTreeRegressor()

# Adjusting the grid search with more conservative parameters
tree_param_grid = {
    'max_depth': [3, 5, 7],  # More conservative depth
    'min_samples_split': [20, 40],  # Higher values to reduce overfitting
    'min_samples_leaf': [10, 20]  # Higher values to ensure more generalization
}
tree_grid_search = GridSearchCV(tree, tree_param_grid, cv=5, scoring='neg_mean_squared_error')
tree_grid_search.fit(xtrain, ytrain)
tree_best = tree_grid_search.best_estimator_

tree_prediction = tree_best.predict(xfuture)  # Making predictions

# Plotting predictions for Decision Tree Regressor
valid['Tree Prediction'] = tree_prediction
plt.figure(figsize=(16, 8))
plt.title("Apple's Stock Price Prediction (Decision Tree Regressor)")
plt.xlabel("Days")
plt.ylabel("Close Price USD ($)")
plt.plot(apple["Close"])
plt.plot(valid[["Close", "Tree Prediction"]])
plt.legend(["Original", "Valid", "Tree Prediction"])
plt.show()

# Linear Regression Model (Unchanged)
linear = LinearRegression().fit(x, y)  # Training the model
linear_prediction = linear.predict(xfuture)  # Making predictions

# Plotting predictions for Linear Regression
valid['Linear Prediction'] = linear_prediction
plt.figure(figsize=(16, 8))
plt.title("Apple's Stock Price Prediction (Linear Regression)")
plt.xlabel("Days")
plt.ylabel("Close Price USD ($)")
plt.plot(apple["Close"])
plt.plot(valid[["Close", "Linear Prediction"]])
plt.legend(["Original", "Valid", "Linear Prediction"])
plt.show()
