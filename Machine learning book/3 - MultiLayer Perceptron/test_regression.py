from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from mlp import mlp

x = ones((1, 40)) * linspace(0, 1, 40)
t = sin(2 * pi * x) + cos(2 * pi * x) + np.random.randn(40) * 0.2
x = transpose(x)
t = transpose(t)

n_hidden = 3
eta = 0.25
n_iterations = 101

plt.plot(x, t, '.')
plt.show()

train = x[0::2, :]
test = x[1::4, :]
valid = x[3::4, :]

train_targets = t[0::2, :]
test_targets = t[1::4, :]
valid_targets = t[3::4, :]

net = mlp(train, train_targets, n_hidden, out_type='linear')
net.mlptrain(train, train_targets, eta, n_iterations)

best_err = net.earlystopping(train, train_targets, valid, valid_targets, eta)
