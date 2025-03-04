import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3, task='classification'):
        self.k = k
        self.task = task  # 'classification' or 'regression'
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Stores the training data."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """Predicts the class or value for each instance in X."""
        X = np.array(X)
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """Predict a single instance."""
        # Compute distances
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get k nearest labels
        k_nearest_labels = self.y_train[k_indices]

        if self.task == 'classification':
            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        elif self.task == 'regression':
            # Average of k nearest labels
            return np.mean(k_nearest_labels)
        else:
            raise ValueError("Task must be 'classification' or 'regression'")


# Example Usage
if __name__ == "__main__":
    # Sample dataset
    X_train = np.array([[1, 2], [2, 3], [3, 1], [5, 4], [6, 7]])
    y_train_class = np.array([0, 0, 1, 1, 1])
    y_train_reg = np.array([2.5, 3.0, 1.5, 4.5, 6.0])

    X_test = np.array([[3, 2], [5, 5]])

    # Classification Example
    knn_clf = KNN(k=3, task='classification')
    knn_clf.fit(X_train, y_train_class)
    print("Classification Predictions:", knn_clf.predict(X_test))

    # Regression Example
    knn_reg = KNN(k=3, task='regression')
    knn_reg.fit(X_train, y_train_reg)
    print("Regression Predictions:", knn_reg.predict(X_test))
