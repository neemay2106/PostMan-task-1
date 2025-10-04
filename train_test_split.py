import pandas as pd
import math
import numpy as np
import random

SEED = 21
np.random.seed(SEED)
random.seed(SEED)


class TrainTestSplit:
    def __init__(self):
        pass  # required since __init__ is empty

    def scaling_function(self, x_tr, x_ts):
        x_tr = x_tr.astype(float).copy()
        x_ts = x_ts.astype(float).copy()

        n_tr = x_tr.shape[1]

        for l in range(n_tr):
            col_tr = x_tr.iloc[:, l]
            col_ts = x_ts.iloc[:, l]

            x_min = col_tr.min()
            x_max = col_tr.max()

            if x_max == x_min:
                x_tr.iloc[:, l] = 0.0
                x_ts.iloc[:, l] = 0.0
            else:
                x_tr.iloc[:, l] = (col_tr - x_min) / (x_max - x_min)
                x_ts.iloc[:, l] = (col_ts - x_min) / (x_max - x_min)

        return x_tr, x_ts

    def split_shuffle(self, df, test_ratio, positive, negative, flag):
        # Remove duplicates
        df = df.drop_duplicates().reset_index(drop=True)

        # Get unique customers
        unique_customers = df['customer_unique_id'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_customers)

        # Split customers into train/test
        n_test = math.floor(len(unique_customers) * test_ratio)
        test_customers = set(unique_customers[:n_test])
        train_customers = set(unique_customers[n_test:])

        # Assign rows based on customer membership
        test_set = df[df['customer_unique_id'].isin(test_customers)].reset_index(drop=True)
        train_set = df[df['customer_unique_id'].isin(train_customers)].reset_index(drop=True)

        # Features
        x_tr = train_set[['Monetary', 'Recency', 'Frequency']].copy()
        x_ts = test_set[['Monetary', 'Recency', 'Frequency']].copy()

        # Apply scaling
        x_train, x_test = self.scaling_function(x_tr, x_ts)

        # Labels
        y_train = train_set['Churn']
        y_test = test_set['Churn']

        # Debug overlap (should now be zero)
        overlap_ids = set(train_set['customer_unique_id']) & set(test_set['customer_unique_id'])
        print("Number of overlapping customers:", len(overlap_ids))

        return x_train, y_train, x_test, y_test
