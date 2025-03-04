import numpy as np
from collections import Counter
from sklearn.utils import resample


class DecisionTree:
    def __init__(self, max_depth: int = None):
        """
        Initialize the Decision Tree classifier.

        Parameters:
        max_depth (int, optional): The maximum depth of the tree. Default is None (no limit).
        """
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, y: np.ndarray) -> float:
        """
        Compute the entropy of the labels.

        Parameters:
        y (np.ndarray): Array of labels.

        Returns:
        float: Entropy value.
        """
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Find the best feature and threshold to split the dataset.

        Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.

        Returns:
        tuple: (best feature index, best threshold value)
        """
        best_gain = 0
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_entropy = self.entropy(y[left_mask])
                right_entropy = self.entropy(y[right_mask])
                total_entropy = (len(y[left_mask]) * left_entropy + len(y[right_mask]) * right_entropy) / len(y)

                gain = self.entropy(y) - total_entropy

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """
        Recursively build the decision tree.

        Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        depth (int): Current depth of the tree.

        Returns:
        dict: Decision tree structure.
        """
        if len(set(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return Counter(y).most_common(1)[0][0]

        feature, threshold = self.best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self.build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self.build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the decision tree classifier.

        Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        """
        self.tree = self.build_tree(X, y)

    def predict_sample(self, sample: np.ndarray, tree):
        """
        Predict the class label for a single sample.

        Parameters:
        sample (np.ndarray): A single feature vector.
        tree (dict or int): Decision tree structure or leaf node.

        Returns:
        int: Predicted class label.
        """
        if not isinstance(tree, dict):  # If it's a leaf node, return its value
            return tree

        if sample[tree["feature"]] <= tree["threshold"]:
            return self.predict_sample(sample, tree["left"])
        else:
            return self.predict_sample(sample, tree["right"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for a dataset.

        Parameters:
        X (np.ndarray): Feature matrix.

        Returns:
        np.ndarray: Array of predicted labels.
        """
        return np.array([self.predict_sample(sample, self.tree) for sample in X])


class RandomForest:
    def __init__(self, n_trees: int = 10, max_depth: int = None, max_features: int = None):
        """
        Initialize the Random Forest classifier.

        Parameters:
        n_trees (int): Number of decision trees.
        max_depth (int, optional): Maximum depth of each tree.
        max_features (int, optional): Maximum number of features per tree.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the random forest classifier.

        Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        """
        n_samples, n_features = X.shape
        self.max_features = self.max_features or int(np.sqrt(n_features))  # Default to sqrt(features)

        for _ in range(self.n_trees):
            # Bootstrap sampling
            X_sample, y_sample = resample(X, y, replace=True)

            # Select random subset of features
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_sample = X_sample[:, feature_indices]

            # Train a decision tree
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)

            # Store tree along with feature indices
            self.trees.append((tree, feature_indices))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for a dataset.

        Parameters:
        X (np.ndarray): Feature matrix.

        Returns:
        np.ndarray: Array of predicted labels.
        """
        tree_predictions = np.array([
            tree.predict(X[:, features]) for tree, features in self.trees
        ])

        # Majority voting
        final_predictions = [Counter(tree_predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(final_predictions)
