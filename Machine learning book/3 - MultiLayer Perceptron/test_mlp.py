import numpy as np
from mlp import mlp

and_data = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
xor_data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

X_train = and_data[:, :2]
y_train = and_data[:, 2:]

eta =  0.25
n_iterations = 1001

p = mlp(X_train, y_train, 2)
p.mlptrain(X_train, y_train, eta, n_iterations)
p.confmat(X_train, y_train)
