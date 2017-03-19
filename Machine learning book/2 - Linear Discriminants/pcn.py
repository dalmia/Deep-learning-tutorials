import numpy as np

class pcn:
    """ A basic perceptron """

    def __init__(self, inputs, targets):
        """ Constructor """
        # Setup network
        if np.ndim(inputs) > 1:
            self.nIn = np.shape(inputs)[1]
        else:
            self.nIn = 1

        if np.ndim(targets) > 1:
            self.nOut = np.shape(targets)[1]
        else:
            self.nOut = 1

        self.nData = np.shape(inputs)[0]

        #Initialize network
        self.weights = np.random.rand(self.nIn+1, self.nOut)*0.1 - 0.05

    def pcntrain(self, inputs, targets, eta, nIterations):
        """ Train the perceptron """
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)
        # Training
        change = range(self.nData)

        for n in range(nIterations):
            self.activations = self.pcnfwd(inputs)
            self.weights -= eta*np.dot(np.transpose(inputs),
                                       self.activations - targets)

            print "Iteration: ", n
            print self.weights

            activations = self.pcnfwd(inputs)
            print "Final outputs are:"
            print activations

        # return self.weights

    def pcnfwd(self, inputs):
        """ Run the network forward """
        # Compute activations
        activations = np.dot(inputs, self.weights)

        # Threshold the activations
        return np.where(activations > 0, 1, 0)

    def confmat(self, inputs, targets):
        """ Confusion matrix """

        # Add the inputs that match the bias node
        inputs = np.concantenate((inputs, -np.ones((self.nData, 1))), axis=1)

        outputs = np.dot(inputs, self.weights)

        nClasses = np.shape(targets)[1]

        if nClasses == 1:
            nClasses = 2
            outputs = np.where(outputs > 0, 1, 0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        # Confusion matrix
        cm = np.zeros((nClasses, nClasses))
        for i in range(nClasses):
            for  j in range(nClasses):
                cm[i, j] = np.sum(np.where(outputs==i, 1, 0) * np.where(targets==j, 1, 0))

        print cm
        print np.trace(cm) / np.sum(cm)

def logic():
    import pcn
    """ Run XOR and AND functions """

    b = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
    a = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])

    p = pcn.pcn(a[:,0:2],a[:,2:])
    p.pcntrain(a[:,0:2],a[:,2:],0.25,10)
    p.confmat(a[:,0:2],a[:,2:])

    q = pcn.pcn(b[:,0:2],b[:,2:])
    q.pcntrain(b[:,0:2],b[:,2:],0.25,10)
    q.confmat(b[:,0:2],b[:,2:])
