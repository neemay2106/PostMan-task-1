import numpy as np
from collections import Counter
import random

SEED = 21
np.random.seed(SEED)
random.seed(SEED)


class Node:
    def __init__(self, feature = None , right = None , left = None , threshold = None,*,value = None):
        self.feature = feature
        self.right = right
        self.left = left
        self.value = value
        self.threshold = threshold

    def is_leaf(self):
        return self.value is not None





class DecisionTree:
    def __init__(self,max_depth=100 , min_samples = 3, no_features = None):
        self.max_depth =max_depth
        self.min_samples=min_samples
        self.no_features =no_features
        self.root_node = None





    def fit(self,x,y):
        x = np.array(x)  # force numpy
        y = np.array(y)

        self.no_features = x.shape[1] if not self.no_features else min(x.shape[1], self.no_features)
        self.root_node =  self.grow_tree(x,y)



    def grow_tree(self, x,y, depth = 0 ):

        no_samples , no_feats = x.shape
        no_labels = len(np.unique(y))

        if depth>= self.max_depth or no_samples <= self.min_samples or no_labels == 1 :
            return Node(value=self.most_common(y))

        feat_indexes = np.random.choice(no_feats , self.no_features , replace = False)

        best_feature, best_threshold = self.best_split(x,y,feat_indexes)
        if best_threshold is None:
            leaf_value = self.most_common(y)
            return Node(value=leaf_value)


        l_indxs , r_indxs = self.branch(x[: , best_feature], best_threshold)
        left_child = self.grow_tree(x[l_indxs], y[l_indxs], depth + 1)
        right_child = self.grow_tree(x[r_indxs], y[r_indxs], depth + 1)
        return Node(feature = best_feature, threshold = best_threshold, left = left_child,  right = right_child)





    def branch(self,x_col,threshold):
        left = np.argwhere(x_col <= threshold).flatten()
        right = np.argwhere(x_col > threshold).flatten()
        return left, right

    def best_split(self, x, y, feat_indexes):

        best_gain = -1
        best_feature = None
        best_threshold = None
        parent_entropy = self.entropy(y)

        for feature in feat_indexes:
            x_column = x[:, feature]  # ensure this is 1D and matches y
            sorted_idx = np.argsort(x_column)
            x_sorted = x_column[sorted_idx]
            y_sorted = y[sorted_idx].ravel()  # same length as x_sorted

            # Initialize counts for sweep
            classes = np.unique(y)
            left_counts = {c: 0 for c in classes}
            right_counts = {c: np.sum(y_sorted == c) for c in classes}

            # Sweep through sorted values
            for i in range(len(y_sorted) - 1):
                label = y_sorted[i]
                left_counts[label] += 1
                right_counts[label] -= 1

                # Only consider threshold where label changes
                if y_sorted[i] != y_sorted[i + 1]:
                    threshold = (x_sorted[i] + x_sorted[i + 1]) / 2

                    left_total = i + 1
                    right_total = len(y_sorted) - left_total

                    # Compute entropy efficiently using counts
                    left_entropy = -sum(
                        (count / left_total) * np.log2(count / left_total)
                        for count in left_counts.values() if count > 0
                    )
                    right_entropy = -sum(
                        (count / right_total) * np.log2(count / right_total)
                        for count in right_counts.values() if count > 0
                    )

                    weighted_entropy = (left_total / len(y_sorted)) * left_entropy + \
                                       (right_total / len(y_sorted)) * right_entropy

                    info_gain = parent_entropy - weighted_entropy

                    if info_gain > best_gain:
                        best_gain = info_gain
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold

    def entropy(self, y):
        y = y.astype(int)
        vals = np.bincount(y)
        p = vals / len(y)
        entropy_value = -np.sum([pi * np.log2(pi) for pi in p if pi > 0])
        return entropy_value

    def information_gain(self,x_col,y,t):
        parent_entropy = self.entropy(y)

        left_idx , right_idx = self.branch(x_col,t)
        left_e = self.entropy(y[left_idx])
        right_e =self.entropy(y[right_idx])

        weighted_mean = (len(left_idx)/len(y))*left_e+ (len(right_idx)/len(y))*right_e

        info_gain = parent_entropy - weighted_mean
        return info_gain






    def most_common(self,y):
        if len(y) == 0:
            return None
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value




    def predict(self, x):
        x = np.array(x)  # force numpy
        return np.array([self._traverse_tree(b, self.root_node) for b in x])


    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

