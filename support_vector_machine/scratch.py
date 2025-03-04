import numpy as np


class SVM:
    """
    Support Vector Machine (SVM) classifier implemented from scratch using Stochastic Gradient Descent (SGD).
    This implementation supports linear classification.
    """

    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, n_iters: int = 1000):
        """
        Initializes the SVM model with hyperparameters.

        Parameters:
        learning_rate (float): The step size for gradient descent optimization.
        lambda_param (float): Regularization parameter to control margin softness.
        n_iters (int): Number of iterations for training.
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the SVM model using the provided dataset.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target vector of shape (n_samples,), with values either -1 or 1.

        Returns:
        None
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Ensure labels are in {-1, 1}
        y = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y[idx]))
                    self.bias -= self.learning_rate * y[idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class labels for input samples.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
        np.ndarray: Predicted class labels of shape (n_samples,).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)


# Generate simple dataset
X_train = np.array([[2, 3], [1, 1], [2, 1], [3, 3], [3, 2], [1, 2]])
y_train = np.array([1, -1, -1, 1, 1, -1])

# Initialize and train the SVM model
svm = SVM(learning_rate=0.01, lambda_param=0.1, n_iters=1000)
svm.fit(X_train, y_train)

# Test prediction
X_test = np.array([[2, 2], [3, 1]])
predictions = svm.predict(X_test)
print("Predictions:", predictions)
