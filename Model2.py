
from collections import Counter
import numpy as np
import random
from decision_tree import DecisionTree

SEED = 21
np.random.seed(SEED)
random.seed(SEED)

class RandomForestClassifier:
    def __init__(self,n_trees = 10 , max_depth= 10 , min_samples = 2 , features = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples =min_samples
        self.features = features
        self.trees = []

    def fit(self,x,y):

        self.trees = []

        for n in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth , min_samples = self.min_samples, no_features = None)
            x_sam , y_sam = self.bootstrap_data(x,y)

            tree.fit(x_sam,y_sam)
            self.trees.append(tree)

    def bootstrap_data(self,x,y):
        if type(x) != np.ndarray:
            x = x.to_numpy()
        if type(y) != np.ndarray:
            y = y.to_numpy()
        idx = np.random.choice(len(x), len(x), replace=True)
        return x[idx], y[idx]

    def most_common(self,y):
        if len(y) == 0:
            return None
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    def predict(self,x):
        predictions =np.array([tree.predict(x) for tree  in self.trees])
        preds = np.swapaxes(predictions,0,1)
        final_preds = np.array([self.most_common(p) for p in preds])
        return final_preds