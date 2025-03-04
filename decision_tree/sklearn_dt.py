from sklearn.tree import DecisionTreeClassifier
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # Example feature matrix
y = np.array([0, 1, 0, 1])  # Example labels

model = DecisionTreeClassifier(max_depth=None)
model.fit(X, y)
predictions = model.predict(X)

print("Predictions:", predictions)
print("score:", model.score(X, y))
