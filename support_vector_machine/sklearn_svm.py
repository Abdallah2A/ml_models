import numpy as np
from sklearn.svm import SVC

# Generate simple dataset
X_train = np.array([[2, 3], [1, 1], [2, 1], [3, 3], [3, 2], [1, 2]])
y_train = np.array([1, -1, -1, 1, 1, -1])

# Initialize and train the SVM model using sklearn
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

# Test prediction
X_test = np.array([[2, 2], [3, 1]])
predictions = svm.predict(X_test)
print("Predictions:", predictions)
