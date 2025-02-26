from sklearn.linear_model import LogisticRegression
import numpy as np


X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

lr_model = LogisticRegression()
lr_model.fit(X, y)

y_prediction = lr_model.predict(X)

print("Prediction on training set:", y_prediction)
print("Accuracy on training set:", lr_model.score(X, y))
