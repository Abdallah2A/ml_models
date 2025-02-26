import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


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

linear_model = LinearRegression()
linear_model.fit(X_norm, y_train)

b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")

y_predict = linear_model.predict(X_norm)
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
