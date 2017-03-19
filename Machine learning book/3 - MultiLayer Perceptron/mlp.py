import numpy as np


class mlp:
    """ A Multi-Layer Perceptron"""

    def __init__(self, inputs, targets, n_hidden, beta=1, momentum=0.9,
                 out_type='sigmoid'):
        """ Constructor """
        # Set up network size
        self.n_in = np.shape(inputs)[1]
        self.n_out = np.shape(targets)[1]
        self.n_data = np.shape(inputs)[0]
        self.n_hidden = n_hidden

        self.beta = beta
        self.momentum = momentum

        if out_type not in ['linear', 'sigmoid', 'softmax']:
            raise ValueError("'out_type' should be one of 'linear', 'sigmoid' "
                             " or 'softmax.'")
        self.out_type = out_type

        # Initialise network
        self.weights1 = ((np.random.rand(self.n_in + 1, self.n_hidden) - 0.5) *
                         2 / np.sqrt(self.n_in))
        self.weights2 = ((np.random.rand(self.n_hidden+1, self.n_out) - 0.5) *
                         2 / np.sqrt(self.n_hidden))

    def earlystopping(self, inputs, targets, valid, valid_targets, eta,
                      n_iterations=100):
        valid = np.concatenate((valid, -np.ones((np.shape(valid)[0], 1))),
                               axis=1)
        old_err_1 = 100002
        old_err_2 = 100001
        new_err = 10000

        count = 0
        while (old_err_1 - new_err) > 0.001 or (old_err_2 - old_err_1) > 0.001:
            count += 1
            print count
            self.mlptrain(inputs, targets, eta, n_iterations)
            old_err_2 = old_err_1
            old_err_1 = new_err
            valid_out = self.mlpfwd(valid)
            new_err = 0.5 * np.sum((valid_targets - valid_out) ** 2)

        print("Stopped ", new_err, old_err_1, old_err_2)
        return new_err

    def mlptrain(self, inputs, targets, eta, n_iterations):
        inputs = np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))),
                                axis=1)

        update_w1 = np.zeros(np.shape(self.weights1))
        update_w2 = np.zeros(np.shape(self.weights2))

        for n in range(n_iterations):
            self.outputs = self.mlpfwd(inputs)

            error = 0.5 * np.sum((self.outputs - targets) ** 2)
            if np.mod(n, 100) == 0:
                print("Iteration: ", n, "Error: ", error)

            # Different output neurons
            if self.out_type == 'linear':
                delta_o = (self.outputs - targets) / self.n_data
            elif self.out_type == 'sigmoid':
                delta_o = (self.beta * (self.outputs - targets) *
                           self.outputs * (1.0 - self.outputs))
            else:
                delta_o = ((self.outputs - targets) * self.outputs *
                           (1.0 - self.outputs) / self.n_data)

            delta_h = (self.hidden * self.beta * (1.0 - self.hidden) *
                       np.dot(delta_o, np.transpose(self.weights2)))

            update_w1 = (eta * np.dot(np.transpose(inputs), delta_h[:, :-1]) +
                         self.momentum * update_w1)
            update_w2 = (eta * np.dot(np.transpose(self.hidden), delta_o) +
                         self.momentum * update_w2)
            self.weights1 -= update_w1
            self.weights2 -= update_w2

    def mlpfwd(self, inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs, self.weights1)
        self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))
        self.hidden = np.concatenate((self.hidden,
                                      -np.ones((np.shape(inputs)[0], 1))),
                                     axis=1)

        outputs = np.dot(self.hidden, self.weights2)

        # Different output neurons
        if self.out_type == 'linear':
            return outputs
        elif self.out_type == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-self.beta * outputs))
        else:
            normalizers = (np.sum(np.exp(outputs), axis=1) *
                           np.ones((1, np.shape(outputs)[0])))
            return np.transpose(np.transpose(np.exp(outputs)) / normalizers)

    def confmat(self, inputs, targets):
        """ Confusion Matrix """

        inputs = np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))),
                                axis=1)
        outputs = self.mlpfwd(inputs)

        n_classes = np.shape(outputs)[1]

        if n_classes == 1:
            n_classes = 2
            outputs = np.where(outputs > 0.5, 1, 0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                cm[i, j] = (np.sum(np.where(outputs == i, 1, 0) *
                            np.where(targets == j, 1, 0)))

        print("Confusion Matrix is: ")
        print(cm)
        print("Percentage Correct: ", np.trace(cm) / np.sum(cm) * 100)
