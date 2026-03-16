#Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Load the dataset

iris=load_iris()
x=iris.data[:,:2]
y=iris.target

#split dataset

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2, random_state=42
)

#Train the model

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

#Predictions

y_pred=knn.predict(x_test)

#Accuracy

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

#Visualizaton 
plt.scatter(x[:,0], x[:,1], c=y)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("KNN Classification")

plt.show()