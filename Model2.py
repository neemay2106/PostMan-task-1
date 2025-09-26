from ensurepip import bootstrap

import numpy

class RandomForestClassifier:
    def __init__(self, df):
        self.df = df


    def bootstrap_data(self):
        indices = numpy.random.choice(len(self.df), len(self.df), replace = True)
        return self.df[indices]

    def decision_tree(self):
        dataset = self.bootstrap_data()