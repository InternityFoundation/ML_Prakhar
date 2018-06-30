import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("spambase.data")
X = dataset.iloc[:,:57].values  #matrix of features
y = dataset.iloc[:, -1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print "\nResults with Logistic Regression "
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
TN=cm[0][0]
FP=cm[0][1]
TP=cm[1][1]
FN=cm[1][0]
print "FP:",FP,"   FN:",FN
print "TP:",TP,"   TN:",TN
accuracy =(float) (TP + TN) / (TP + TN + FP + FN)
print "Accuracy:",accuracy
precision =(float)(TP) / (TP + FP)
print "Precision:",precision
recall = (float)(TP) / (TP + FN)
print "Recall:",recall
f1_Score =(float) (2 * precision * recall) / (precision + recall)
print "F1 Score:",f1_Score

#With the Naive Bayes Classifier
print "\nResults with Naive Bayes Classifier "
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
TN=cm[0][0]
FP=cm[0][1]
TP=cm[1][1]
FN=cm[1][0]
print "FP:",FP,"   FN:",FN
print "TP:",TP,"   TN:",TN
accuracy =(float) (TP + TN)/ (TP + TN + FP + FN)
print "Accuracy:",accuracy
precision =(float)(TP) / (TP + FP)
print "Precision:",precision
recall = (float)(TP) / (TP + FN)
print "Recall:",recall
f1_Score =(float) (2 * precision * recall) / (precision + recall)
print "F1 Score:",f1_Score

#With the Decision Tree Classifier
print "\nResults with Decision Tree Classifier"
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(random_state=0)
classifier.fit(X,y)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
TN=cm[0][0]
FP=cm[0][1]
TP=cm[1][1]
FN=cm[1][0]
print "FP:",FP,"   FN:",FN
print "TP:",TP,"   TN:",TN
accuracy = (float) (TP + TN) / (TP + TN + FP + FN)
print "Accuracy:",accuracy
precision =(float)(TP) / (TP + FP)
print "Precision:",precision
recall = (float)(TP) / (TP + FN)
print "Recall:",recall
f1_Score =(float) (2 * precision * recall) / (precision + recall)
print "F1 Score:",f1_Score

#With the K-NN Classifier
print "\nResults with KNN Classifier"
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2) #p=2 for the euclidian_distance
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
TN=cm[0][0]
FP=cm[0][1]
TP=cm[1][1]
FN=cm[1][0]
print "FP:",FP,"   FN:",FN
print "TP:",TP,"   TN:",TN
accuracy = (float) (TP + TN) / (TP + TN + FP + FN)
print "Accuracy:",accuracy
precision =(float)(TP) / (TP + FP)
print "Precision:",precision
recall = (float)(TP) / (TP + FN)
print "Recall:",recall
f1_Score =(float) (2 * precision * recall) / (precision + recall)
print "F1 Score:",f1_Score

