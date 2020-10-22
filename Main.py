import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#load data

df = pd.read_csv("AC.TO.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df = df.dropna()
print(df.tail(25))
print("*" * 100)
df = df[["Adj Close"]]


# Create prediction variable
days_in_future = 30
df["Prediction"] = df[["Adj Close"]].shift(-days_in_future)


#Create feature data set
x = np.array(df.drop(["Prediction"], 1))[:-days_in_future]

# Creat target data set
y = np.array(df["Prediction"])[:-days_in_future]

# Split data 75/25
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.15)

# Create models
tree = DecisionTreeRegressor().fit(x_train, y_train)

# Linear Regression model
lr = LinearRegression().fit(x_train, y_train)

x_future = df.drop(["Prediction"], 1)[:-days_in_future]
x_future = x_future.tail(days_in_future)
x_future = np.array(x_future)

# Model tree
tree_prediction = tree.predict(x_future)
lr_prediction = lr.predict(x_future)

#Visualize TREE PREDICTION
predictions = tree_prediction
valid = df[x.shape[0]:]
valid["Predictions"] = predictions
plt.figure(figsize=(16,8))
plt.title("Linear Regression Model")
plt.xlabel("Days")
plt.ylabel("Adjusted close CAD ($)")
plt.plot(df["Adj Close"])
plt.plot(valid[["Adj Close", "Predictions"]])
plt.legend(["Default", "Actual", "Predicted"])
plt.show()

# Visualize Linear Regression
# predictions = tree_prediction
# valid = df[x.shape[0]:]
# valid["Predictions"] = predictions
# plt.figure(figsize=(8,4))
# plt.title("Linear Regression Model")
# plt.xlabel("Days")
# plt.ylabel("Adjusted close CAD ($)")
# plt.plot(df["Adj Close"])
# plt.plot(valid[["Adj Close", "Predictions"]])
# plt.legend(["Default", "Actual", "Predicted"])
# plt.show()