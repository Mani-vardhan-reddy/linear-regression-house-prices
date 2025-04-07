# House Price Prediction - Linear Regression

This project uses **Linear Regression** to predict house prices based on multiple features using a synthetic dataset of 1000 rows.

---

## Problem Statement

Predict the selling price of a house based on features like area, number of bedrooms, bathrooms, etc. using a multiple linear regression model.

---

## Dataset

- `house_data_large.csv`
- 1000 rows, 6 columns
- Cleaned: no missing values, no duplicates

---

## Features

- Area (in sq ft)
- Number of Bedrooms
- Number of Bathrooms
- Age of the Property
- Is property having a garage and how many

---

## Steps Followed

1. Loaded and cleaned the dataset
2. Built a Linear Regression model
3. Evaluated the model using:
   - R² Score
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
4. Plotted a graph between actual and predicted values

---

##  Model Performance

| Metric        | Value         |
|---------------|---------------|
| R² Score      | 0.9958        |
| MSE           | 99,862,352.69 |
| MAE           | 8,100.45      |

 Model performs exceptionally well on this dataset.

---

## Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- Matplotlib
- PyCharm

---


