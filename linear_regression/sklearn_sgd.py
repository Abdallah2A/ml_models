import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_house_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the dataset from the file
    :return:
    x (np.ndarray): train set
    y (np.ndarray): label of train set
    """
    data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
    x = data[:, :4]
    y = data[:, 4]
    return x, y


X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

sgdr = SGDRegressor()
sgdr.fit(X_norm, y_train)

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters: w: {w_norm}, b:{b_norm}")

y_predict = sgdr.predict(X_norm)
print(f"Prediction on training set:\n{y_predict[:4]}")
print(f"Target values \n{y_train[:4]}")

# plot predictions and targets vs original features
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, label='target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:, i], y_predict, color='r', label='predict')
ax[0].set_ylabel("Price")
ax[0].legend()
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
