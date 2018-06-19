import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

data=pd.read_csv("Salary_Data.csv")
X=data.iloc[:,:1].values
y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()

lm.fit(X_train,y_train)
y_pred=lm.predict(X_test)

plt.scatter(X_test,y_test,color="red")
plt.plot(X_test,y_pred,color="blue")

score=lm.score(X_test,y_test)
print "Score:",score
