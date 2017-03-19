import numpy as np
import pcn
import kmeans

class rbf:
    """The Radial Basis Function Network"""

    def __init__(self, inputs, targets, n_rbf, sigma=0, use_kmeans=0,
                 normalise=0):
        self.n_in = np.shape(inputs)[1]
        self.n_out = np.shape(targets)[1]
        self.n_data = np.shape(inputs)[0]
        self.n_rbf = n_rbf
        self.use_kmeans = use_kmeans
        self.normalise = normalise

        if use_kmeans:
            self.kmeans_net = kmeans.kmeans(self.n_rbf, inputs)

        self.hidden = np.zeros((self.n_data, self.n_rbf + 1))

        if sigma == 0:
            # Set the width of Gaussians
            d = (inputs.max(axis=0) - inputs.min(axis=0)).max()
            self.sigma = d / np.sqrt(2 * n_rbf)
        else:
            self.sigma = sigma

        self.perceptron = pcn.pcn(self.hidden[:, :-1], targets)

        # Initialize the network
        self.weights1 = np.zeros((self.n_in, self.n_rbf))

    def rbf_train(self, inputs, targets, eta=0.25, n_iterations=100):

        if self.use_kmeans == 0:
            # Version 1: Set RBFs to be datapoints
            indices = range(self.n_data)
            np.random.shuffle(indices)
            for i in range(self.n_rbf):
                self.weights1[: i] = inputs[indices[i], :]

        else:
            # Version 2: use k-means
            self.weights1 = np.transpose(self.kmeans_net.kmeanstrain(inputs))

        for i in range(self.n_rbf):
            self.hidden[:,i] = np.exp(-np.sum((inputs - np.ones((1,self.n_in))*self.weights1[:,i])**2,axis=1)/(2*self.sigma**2))

        if self.normalise:
            self.hidden[:, :-1] /= self.hidden[:, :-1].sum(axis=1).reshape(-1, 1)

        # Call Perceptron without bias node
        self.perceptron.pcntrain(self.hidden[:, :-1], targets, eta, n_iterations)

    def rbf_fwd(self, inputs):

        hidden = np.zeros((np.shape(inputs)[0], self.n_rbf + 1))

        for i in range(self.n_rbf):
            hidden[:,i] = np.exp(-np.sum((inputs - np.ones((1,self.n_in))*self.weights1[:,i])**2,axis=1)/(2*self.sigma**2))

        if self.normalise:
            hidden[:,:-1] /= hidden[:,:-1].sum(axis=1).reshape(-1, 1)

        # Add the bias
        hidden[:,-1] = -1

        outputs = self.perceptron.pcnfwd(hidden)
        return outputs

    def confmat(self, inputs, targets):
        """Confusion Matrix"""

        outputs = self.rbf_fwd(inputs)
        n_classes = np.shape(targets)[1]

        if n_classes == 1:
            n_classes = 2
            outputs = np.where(outputs > 0, 1, 0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                cm[i, j] = np.sum(np.where(outputs==i, 1, 0) * np.where(targets==j, 1, 0))

        print(cm)
        print(np.trace(cm) / np.sum(cm))
