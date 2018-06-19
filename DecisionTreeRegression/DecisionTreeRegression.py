# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values  #matrix of features
y = dataset.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor(random_state=0)
reg.fit(X,y)

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color="red")
plt.plot(X_grid,reg.predict(X_grid),color="blue")
plt.title("Decision Tree Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()