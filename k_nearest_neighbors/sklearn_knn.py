import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Sample dataset
X_train = np.array([[1, 2], [2, 3], [3, 1], [5, 4], [6, 7]])
y_train_class = np.array([0, 0, 1, 1, 1])
y_train_reg = np.array([2.5, 3.0, 1.5, 4.5, 6.0])

X_test = np.array([[3, 2], [5, 5]])

# Classification Example
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train_class)
print("Classification Predictions:", knn_clf.predict(X_test))

# Regression Example
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train, y_train_reg)
print("Regression Predictions:", knn_reg.predict(X_test))
