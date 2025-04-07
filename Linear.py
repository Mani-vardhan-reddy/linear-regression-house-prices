import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

data = pd.read_csv(r"C:\Users\maniv\PycharmProjects\pythonProject\Linear Regression\house_data_large.csv")

# checking for null values

print(data.isna().sum()) # no null values in the data

# Checking for duplicates

print(data.duplicated().sum()) # no duplicates

# Checking for data types

print(data.dtypes) # data types are in correct format

# finding the correlation between features and dependent variable

print(data.corr())

sns.heatmap(data.corr(),annot=True)
plt.show()

# dividing the data into dependent variable and independent variables

X = data.drop("Price",axis=1)
y = data["Price"]

print(X)
print(y)

# Splitting data into training set and testing sets

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# loading model

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

score = r2_score(y_test,y_pred)

mse = mean_squared_error(y_test,y_pred)

mae = mean_absolute_error(y_test,y_pred)

print(f"r2_score = {score}")
print(f"mean_squared_error = {mse}")
print(f"mean_absolute_error = {mae}")

# visualizations of actual and predicted points

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()

# coefficient and intercepts

print(model.coef_)
print(model.intercept_)