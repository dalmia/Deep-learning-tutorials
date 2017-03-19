import numpy as np
import dtree

class bagger:
    def __init__(self):
        self.tree = dtree.dtree()

    def bag(self, data, targets, features, n_samples):
        n_points = np.shape(data)[0]
        n_dim = np.shape(data)[1]
        self.n_samples = n_samples

        # Compute the bootstrap samples
        sample_points = np.random.randint(0, n_points, (n_points, n_samples))
        classifiers = []

        for i in range(n_samples):
            sample = []
            sample_target = []
            for j in range(n_points):
                sample.append(data[sample_points[j, i]])
                sample_target.append(targets[sample_points[j, i]])

            classifiers.append(self.tree.make_tree(sample, sample_target,
                                                   features, 1))
        return classifiers

    def bag_class(self, classifiers, data):

        decision = []
        for j in range(len(data)):
            outputs = []
            for i in range(self.n_samples):
                out = self.tree.classify(classifiers[i], data[j])
                if out is not None:
                    outputs.append(out)

            # List the possible outputs
            out = []
            for each in outputs:
                if each not in out:
                    out.append(each)
            frequency = np.zeros(len(out))

            index = 0
            if len(out) > 0:
                for each in out:
                    frequency[index] = outputs.count(each)
                    index += 1
                decision.append(out[frequency.argmax()])
            else:
                decision.append(None)
        return decision
