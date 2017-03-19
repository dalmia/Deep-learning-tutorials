from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from mlp import mlp

iris = load_iris()

X = iris.data
y = iris.target

n_hidden = 5
out_type = 'softmax'
eta = 0.1

order = range(X.shape[0])
np.random.shuffle(order)

X = X[order]
y = y[order]

lb = LabelBinarizer()
y = lb.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                      test_size=0.25)

net = mlp(X_train, y_train, n_hidden, out_type=out_type)
net.earlystopping(X_train, y_train, X_valid, y_valid, eta)
net.confmat(X_test, y_test)
