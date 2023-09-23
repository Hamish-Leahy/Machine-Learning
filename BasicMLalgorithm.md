# Import libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load data
iris_dataset = load_iris()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=1) 

# Train the model on training data
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# Evaluate predictions
print(knn.score(X_test, y_test))
