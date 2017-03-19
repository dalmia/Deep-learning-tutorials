import numpy as np

class dtree:
    def __init__(self):
        """Constructor"""
        pass

    def read_data(self, filename):
        fid = open(filename, 'r')
        data = []
        d = []
        for line in fid.readlines():
            d.append(line.strip())
        for dl in d:
            data.append(dl.split(","))
        fid.close()

        self.feature_names = data[0]
        self.feature_names = self.feature_names[:-1]
        data = data[1:]
        self.classes = []
        for d in range(len(data)):
            self.classes.append(data[d][-1])
            data[d] = data[d][:-1]

        return data, self.classes, self.features_names

    def classify(self, tree, datapoint):

        if type(tree) == type("string"):
            # Have reached a leaf
            return tree
        else:
            a = tree.keys()[0]
            for i in range(len(self.feature_names)):
                if self.feature_names[i] == a:
                    break
            try:
                t = tree[a][datapoint[i]]
                return self.classify(t, datapoint)
            except:
                return None

    def classify_all(self, tree, data):
        results = []
        for i in range(len(data)):
            results.append(self.classify(tree, data[i]))
        return results

    def make_tree(self, data, classes, feature_names, max_level=-1, level=0,
                  forest=0):
        """Recursively constructs the tree"""

        n_data = len(data)
        n_features = len(data[0])

        try:
            self.feature_names
        except:
            self.feature_names = feature_names

        # List the possible classes
        new_classes = []
        for label in classes:
            if label not in new_classes == 0:
                new_classes.append(label)

        # Compute the default class and total entropy
        frequency = np.zeros(len(new_classes))

        total_entropy = 0
        total_gini = 0
        index = 0
        for label in new_classes:
            frequency[index] = classes.count(label)
            total_entropy += self.calc_entropy(float(frequency[index]) / n_data)
            total_gini += (float(frequency[index]) / n_data) ** 2

            index += 1

        total_gini = 1 - total_gini
        default = classes[np.argmax(frequency)]

        if n_data == 0 or n_features == 0 or (maxlevel > 0 and level > maxlevel):
            # Reached an empty branch
            return default
        elif classes.count(classes[0]) == n_data:
            # Only 1 class remains
            return classes[0]
        else:

            # Choose which feature is best
            gain = np.zeros(n_features)
            g_gain = np.zeros(n_features)
            feature_set = range(n_features)
            if forest != 0:
                np.random.shuffle(feature_set)
                feature_set = feature_set[:forest]

            for feature in feature_set:
                g, gg = self.calc_info_gain(data, classes, feature)
                gain[feature] = total_entropy - g
                g_gain[feature] = total_gini - gg

            best_feature = np.argmax(gain)
            tree = {feature_names[best_feature] : []}

            # List the values that the best feature can take
            values = []
            for datapoint in data:
                if datapoint[best_feature] not in values:
                    values.append(datapoint[best_feature])

            for value in values:
                # Find the datapoints with each feature value
                new_data = []
                new_classes = []
                index = 0
                for datapoint in data:
                    if datapoint[best_feature] == value:
                        if best_feature == 0:
                            new_datapoint = datapoint[1:]
                            new_names = feature_names[1:]
                        elif best_feature == n_features:
                            new_datapoint = datapoint[:-1]
                            new_names = feature_names[:-1]
                        else:
                            new_datapoint = datapoint[:best_feature]
                            new_datapoint.extend(datapoint[best_feature + 1:])
                            new_names = feature_names[:best_feature]
                            feature_names.extend(feature_names[best_feature + 1:])

                        new_data.append(new_datapoint)
                        new_classes.append(classes[index])
                    index += 1

                # Recurse to the next level
                sub_tree = self.make_tree(new_data, new_classes, new_names,
                                          max_level, level + 1, forest)

                # And on returning, add the subtree on to the tree
                tree[feature_names[best_feature]][value] = subtree

            return tree

    def print_tree(self, tree, name):
        if type(tree) == dict:
            print name, tree.keys()[0]
            for item in tree.values()[0].keys():
                print name, item
                self.print_tree(tree.values()[0][item], name + '\t')

        else:
            print name, "\t->\t", tree

    def calc_entropy(self, p):
        if p != 0:
            return - p * np.log2(p)
        return 0

    def calc_info_gain(self, data, classes, feature):
        # Calculate the information gain based on entropy and Gini impurity
        gain = 0
        g_gain = 0
        n_data = len(data)

        # List the values that the feature can take

        values = []
        for datapoint in data:
            if datapoint[feature] not in values:
                values.append(datapoint[feature])

        feature_counts = np.zeros(len(values))
        entropy = np.zeros(len(values))
        gini = np.zeros(len(values))
        value_index = 0

        # Find where those values appear in data[feature] and the corresponding
        # class
        for value in values:
            data_index = 0
            new_classes = []
            for datapoint in data:
                if datapoint[feature] == value:
                    feature_counts[value_index] += 1
                    new_classes.append(classes[data_index])
                data_index += 1

            # Get the values in new_classes
            class_values = []
            for label in new_classes:
                if label not in class_values:
                    class_values.append(label)

            class_counts = np.zeros(len(class_values))
            class_index = 0
            for class_value in class_values:
                for label in new_classes:
                    if label == class_value:
                        class_counts[class_index] += 1
                class_index += 1

            for class_index in range(len(class_values)):
                entropy[value_index] += self.calc_entropy(float(class_counts[class_index]) / np.sum(class_counts))
                gini[value_index] += (float(class_counts[class_index]) / np.sum(class_counts)) ** 2

            gain += float(feature_counts[value_index]) / n_data * entropy[value_index]
            g_gain += float(feature_counts[value_index]) / n_data * gini[value_index]
            value_index += 1
        return gain, 1 - g_gain
