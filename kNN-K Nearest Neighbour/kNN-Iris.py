import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:,5].values

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
reg=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2) #p=2 for the euclidian_distance
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


plt.scatter(X_test[:,0],y_pred,color="red")
plt.ylabel("Species(0-Iris-setosa, 1-Iris-versicolor, 2-Iris-virginica)")
plt.xlabel("Sepal Length(cm)")

plt.scatter(X_test[:,1],y_pred,color="blue")
plt.ylabel("Species(0-Iris-setosa, 1-Iris-versicolor, 2-Iris-virginica)")
plt.xlabel("Sepal Width(cm)")

plt.scatter(X_test[:,2],y_pred,color="green")
plt.ylabel("Species(0-Iris-setosa, 1-Iris-versicolor, 2-Iris-virginica)")
plt.xlabel("Petal Length(cm)")

plt.scatter(X_test[:,3],y_pred,color="yellow")
plt.ylabel("Species(0-Iris-setosa, 1-Iris-versicolor, 2-Iris-virginica)")
plt.xlabel("Petal Width(cm)")
