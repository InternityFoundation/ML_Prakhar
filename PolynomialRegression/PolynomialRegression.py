# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values  #matrix of features
y = dataset.iloc[:, 2].values


from sklearn.linear_model import LinearRegression
reg1=LinearRegression()
reg1.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X)

reg2=LinearRegression()
reg2.fit(X_poly,y)



#Polynomial Regression with degree 2
plt.scatter(X,y,color="red")
plt.plot(X,reg1.predict(X),color="blue")
plt.title("Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

plt.scatter(X,y,color="red")
plt.plot(X,reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.title("Polynomial Regression with degree 2")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


#Polynomial Regression with degree 3
poly_reg=PolynomialFeatures(degree=3)
X_poly=poly_reg.fit_transform(X)

reg2=LinearRegression()
reg2.fit(X_poly,y)

plt.scatter(X,y,color="red")
plt.plot(X,reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.title("Polynomial Regression with degree 3")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


#Polynomial Regression with degree 4
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)

reg2=LinearRegression()
reg2.fit(X_poly,y)

plt.scatter(X,y,color="red")
plt.plot(X,reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.title("Polynomial Regression with degree 4")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#To get the more continous & continuous curve
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color="red")
plt.plot(X_grid,reg2.predict(poly_reg.fit_transform(X_grid)),color="blue")
plt.title("Polynomial Regression with degree 4 and more continuos curve")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
