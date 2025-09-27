import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature = None , right = None , left = None , threshold = None,*,value = None):
        self.feature = feature
        self.right = right
        self.left = left
        self.value = value
        self.threshold = threshold

    def is_leaf(self):
        return self.value is None




class DecisionTree:
    def __init__(self,max_depth=100 , min_samples = 3, no_features = None):
        self.max_depth =max_depth
        self.min_samples=min_samples
        self.no_features =no_features
        self.root_node = None

    def fit(self,x,y):
        self.no_features = x.shape[1] if not self.no_features else min(x.shape[1], self.no_features)
        self.root_node =  self.grow_tree(x,y)



    def grow_tree(self, x,y, depth = 0 ):
        no_samples , no_feats = x.shape
        no_labels = len(np.unique(y))

        #stopping crieria
        if depth>= self.max_depth or no_feats <= self.min_samples or no_labels == 1 :
            leaf_value = self.most_common(y)
            return leaf_value

        #splitting the tree
        feat_indexes = np.random.choice(no_feats , self.no_features , replace = False)

        best_feature, best_threshold = self.best_split(x,y,feat_indexes)

        #creating the child node
        l_indxs , r_indxs = self.branch(best_feature, best_threshold)
        left_child = self.grow_tree(x[l_indxs], y[l_indxs], depth + 1)
        right_child = self.grow_tree(x[r_indxs], y[r_indxs], depth + 1)
        return Node(feature = best_feature, threshold = best_threshold, left = left_child,  right = right_child)

    def branch(self,x_col,threshold):
        left = np.argwhere(x_col <= threshold).flatten()
        right = np.argwhere(x_col > threshold).flatten()
        return left, right

    def best_split(self, x,y , feat):
        best_gain = -1
        split_index , split_threshold = None, None

        for i in feat:
            x_column = x[:,i]
            threshold = np.unique(x_column)

            for t in threshold:
                gain = self.information_gain(y,x_column,t)
                if gain > best_gain:
                    best_gain = gain
                    split_index = i
                    split_threshold = t
        return split_index, split_threshold


    def entropy(self,y):
        vals = np.bincount(y)
        p = vals/len(y)
        entropy_value = np.sum(p*np.log2(p))
        return entropy_value

    def information_gain(self,x_col,y,t):
        parent_entropy = self.entropy(y)

        left_idx , right_idx = self.branch(x_col,t)
        left_child = self.entropy(y[left_idx])
        right_child =self.entropy(y[right_idx])

        weighted_mean = (len(left_idx)/len(y)*(self.entropy(left_child))+ len(right_idx)/len(y)*(self.entropy(right_child)))

        info_gain = parent_entropy - weighted_mean
        return info_gain

    def most_common(self,y):
        common = Counter(y)
        value = common.most_common(1)[0][0]
        return value

    def predict(self, x):
        return np.array([self._traverse_tree(b, self.root_node) for b in x])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

