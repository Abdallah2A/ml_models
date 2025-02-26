import numpy as np


def calculate_sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
    """
    g = 1 / (1 + np.exp(-z))

    return g


def compute_cost_logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    Computes cost

    Args:
      x (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(w, x[i]) + b
        g_z = calculate_sigmoid(z_i)
        cost += -y[i] * np.log(g_z) - (1 - y[i]) * np.log(1 - g_z)
    cost /= m

    return cost


def compute_gradient_logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float,
                              lambda_: float = 100) -> tuple[np.ndarray, float]:
    """
    Computes the gradient for linear regression

    Args:
      x (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b.
    """
    m, n = x.shape
    dj_dw, dj_db = np.zeros((n, )), 0.0
    for i in range(m):
        g_z = calculate_sigmoid(np.dot(w, x[i]) + b)
        err_i = g_z - y[i]
        for j in range(n):
            dj_dw[j] += err_i * x[i, j]
        dj_db += err_i
    for j in range(n):
        dj_dw[j] += (lambda_/m) * w[j]
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(x: np.ndarray, y: np.ndarray, w_in: np.ndarray, b_in: float, alpha: float,
                     num_iters: int) -> tuple[np.ndarray, float]:
    """
    Performs batch gradient descent

    Args:
      x (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter
    """
    w, b = w_in, b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(x, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

    return w, b


def predict_logistic(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    Predicts labels using logistic regression model parameters

    Args:
      x (ndarray (m,n)): Data, m examples with n features
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      predict (ndarray (m,)): Predicted labels (0 or 1)
    """
    m = x.shape[0]
    predict = np.zeros((m,))
    for i in range(m):
        z_i = np.dot(w, x[i]) + b
        g_z = calculate_sigmoid(z_i)
        predict[i] = 1 if g_z >= 0.5 else 0

    return predict


X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

w_tmp = np.zeros_like(X_train[0])
b_tmp = 0.
alph = 0.1
iters = 10000

w_out, b_out = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters)
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

# Calculate predictions and accuracy
predictions = predict_logistic(X_train, w_out, b_out)
accuracy = np.mean(predictions == y_train)
print(f"Accuracy: {accuracy * 100:.2f}%")
