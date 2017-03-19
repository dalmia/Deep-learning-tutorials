import numpy as np

def linreg(self, inputs, targets):
    inputs = np.concatenate((-ones((np.shape(inputs)[0], 1)), inputs), axis=1)

    beta = dot(dot(linalg.inv(dot(np.transpose(inputs), inputs)), transpose(inputs)), targets)
    outputs = np.dot(inputs, beta)
    return beta

def train_test(self, trainin, traintgt, testin, testtgt):
    beta = self.linreg(trainin, traintgt)

    testin = np.concatenate((-ones(np.shape(testin)[0], 1), testin), axis=1)
    testout = dot(testin, beta)

    error = np.sum((testout - testtgt) ** 2)
    return error
