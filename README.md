# 📌 K-Nearest Neighbors (KNN) Algorithm Implementation

## 📖 Overview

This project demonstrates the practical implementation of the **K-Nearest Neighbors (KNN)** algorithm using Python and the **scikit-learn** library.

KNN is a **supervised machine learning algorithm** used for both **classification** and **regression** problems. It works by identifying the **K nearest data points** to a new input and predicting the output based on the majority class among those neighbors.

In this project, we implement KNN using the **Iris dataset**, a widely used dataset for machine learning practice.

---

## 🧠 What is KNN?

K-Nearest Neighbors is a **lazy learning algorithm** that stores the entire training dataset and makes predictions based on the **distance between data points**.

The algorithm works in the following steps:

1. Choose the number of neighbors **K**
2. Calculate the distance between the new data point and all training data
3. Select the **K closest neighbors**
4. Perform **majority voting** for classification

---

## 🎯 Why Use KNN?

KNN is useful because:

* Simple and easy to understand
* No training phase required
* Works well with small datasets
* Effective for pattern recognition problems

---

## 🛠 Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/knn-algorithm.git
cd knn-algorithm
```

Install required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Or install using a requirements file:

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
KNN-Algorithm/
│
├── knn.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Implementation Steps

### 1️⃣ Import Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```

---

### 2️⃣ Load the Dataset

```python
iris = load_iris()

X = iris.data[:, :2]
y = iris.target
```

---

### 3️⃣ Split the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

### 4️⃣ Train the KNN Model

```python
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)
```

---

### 5️⃣ Make Predictions

```python
y_pred = knn.predict(X_test)
```

---

### 6️⃣ Evaluate the Model

```python
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

---

### 7️⃣ Visualize the Data

```python
plt.scatter(X[:,0], X[:,1], c=y)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("KNN Classification")

plt.show()
```

---

## 📊 Distance Formula Used

KNN commonly uses the **Euclidean Distance**:

[
d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
]

This formula measures the distance between two points in a 2D space.

---

## 📈 Output Example

```
Accuracy: 0.93
```

This means the model correctly predicted **93% of the test data**.

---

## 🚀 Applications of KNN

KNN is used in many real-world applications:

* Recommendation Systems
* Image Recognition
* Medical Diagnosis
* Credit Risk Detection
* Pattern Recognition

---

## ⚠️ Limitations

* Slow for large datasets
* Sensitive to noisy data
* Requires choosing an optimal **K value**

---

## 📌 Future Improvements

* Implement **KNN from scratch**
* Add **decision boundary visualization**
* Optimize **K value using cross-validation**

---

## 👨‍💻 Author

**Prakash K**

Machine Learning Enthusiast | AI Developer

---
