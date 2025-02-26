import numpy as np
from typing import Callable, Tuple


def compute_cost(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    Computes the cost function for linear regression.

    Args:
      x (ndarray (m, n)): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,))    : model parameters
      b (scalar)    : model parameters

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0]

    total_cost = 0
    for i in range(m):
        f_wb = np.dot(x[i], w) + b
        total_cost += (f_wb - y[i]) ** 2
    total_cost /= 2 * m

    return total_cost


def compute_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float,
                     lambda_: float = 1000) -> Tuple[np.ndarray, float]:
    """
    Computes the gradient for linear regression
    Args:
      x (ndarray (m, n)): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,))    : model parameters
      b (scalar)    : model parameters
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b
     """
    m, n = x.shape
    dj_dw, dj_db = np.zeros((n,)), 0
    for i in range(m):
        err = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i, j]
        dj_db += err
    for j in range(n):
        dj_dw[j] += (lambda_/m) * w[j]

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(x: np.ndarray, y: np.ndarray, w_in: np.ndarray, b_in: float, alpha: float, num_iters: int,
                     cost_function: Callable, gradient_function: Callable) -> Tuple[np.ndarray, float]:
    """
    Performs gradient descent to fit w,b. Updates w,b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m, n))  : Data, m examples
      y (ndarray (m,))  : target values
      w_in (ndarray (n,)): initial values of model parameters
      b_in (scalar): initial values of model parameters
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient

    Returns:
      w (ndarray (n,)): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      """
    w, b = w_in, b_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

    return w, b


# Load our data set
x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# initialize parameters
b_init = 0.
w_init = np.zeros(4)

# some gradient descent settings
iterations = 1000
alpha = 5.0e-7

# run gradient descent
w_final, b_final = gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations, compute_cost, compute_gradient)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

m,_ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
