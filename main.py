# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data split and data processing
datas = pd.read_csv("goldsilver_1791-2021.csv")
r = datas.iloc[:, 2:4]
l = datas.iloc[:, 0]
y = datas.iloc[:, 1]
R = pd.DataFrame(data=r.values, index=range(170), columns=["silver", "ratio"])
L = pd.DataFrame(data=l.values, index=range(170), columns=["year"])
x = pd.concat([L, R], axis=1)
X = x.values
Y = y.values
print(x)
print("***********************************************************")
print(y)

# Linear regression predict the gold price
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, Y)
predict = reg.predict(X)
# data visualization
plt.scatter(X[:, 0], Y)
plt.plot(X[:, 0], predict)
plt.show()

# polynomial Features predict the gold price by the degree = 2
from sklearn.preprocessing import PolynomialFeatures
polly = PolynomialFeatures(degree=2)
X_polly = polly.fit_transform(X, Y)
reg2 = LinearRegression()
reg2.fit(X_polly, Y)
predict = reg2.predict(polly.fit_transform(X))
# data visualization
plt.scatter(X[:, 0], Y)
plt.plot(X[:, 0], predict)
plt.show()

# polynomial Features predict the gold price by the degree = 4
from sklearn.preprocessing import PolynomialFeatures
polly2 = PolynomialFeatures(degree=4)
X_polly = polly2.fit_transform(X, Y)
reg3 = LinearRegression()
reg3.fit(X_polly, Y)
predict = reg3.predict(polly2.fit_transform(X))
# data visualization
plt.scatter(X[:, 0], Y)
plt.plot(X[:, 0], predict)
plt.show()

# the ML predict the 2020 gold price
print("*********************************************************************")
a = [[2020, 20.54, 86.14]]
print(a)
# linear Regression predict 2020 gold price
predict = reg.predict(a)
print("predicted by linear Linear Regression : ", predict)
# polynomial regression predict 2020 gold price by the degree = 2
predict = reg2.predict(polly.fit_transform(a))
print("predicted by Polynomial Regression degree=2 : ", predict)
# polynomial regression predict 2020 gold price by the degree = 4
predict = reg3.predict(polly2.fit_transform(a))
print("predicted by Polynomial Regression degree=4 : ", predict)

# the ML predict the 2021 gold price but 2021 gold price is not at the data
print("*********************************************************************")
a = [[2021, 22.30, 79.15]] # gold price = 1765.13
print(a)
# linear regression predict 2021 gold price
predict = reg.predict(a)
print("predicted by linear Linear Regression : ", predict)
# polynomial regression predict 2021 gold price by the degree = 2
predict = reg2.predict(polly.fit_transform(a))
print("predicted by Polynomial Regression degree=2 : ", predict)
# polynomial regression predict 2021 gold price by the degree = 4
predict = reg3.predict(polly2.fit_transform(a))
print("predicted by Polynomial Regression degree=4 : ", predict)
