from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import rbf

iris = load_iris()

X = iris.data
y = iris.target

n_rbf = 5
sigma = 1
use_kmeans = 1

eta = 0.25
n_iterations = 2000

order = range(X.shape[0])
np.random.shuffle(order)

X = X[order]
y = y[order]

lb = LabelBinarizer()
y = lb.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

net = rbf.rbf(X_train, y_train, n_rbf, sigma, use_kmeans)

net.rbf_train(X_train, y_train, eta, n_iterations)
net.confmat(X_test, y_test)
