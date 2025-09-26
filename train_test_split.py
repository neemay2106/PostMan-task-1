import pandas as pd
import random
import math

def split_shuffle(df,test_ratio,positive, negative,flag):
    x_pos = df[df[f"{flag}"] == positive].sample(frac = 1, replace = False, random_state = 42).reset_index(drop = True)
    x_neg = df[df[f"{flag}"] == negative].sample(frac = 1, replace = False, random_state = 42).reset_index(drop = True)

    #stratified shuffling then split
    len_pos_test = math.floor(len(x_pos) * test_ratio)
    len_neg_test = math.floor(len(x_neg) * test_ratio)
    train_set = pd.concat([x_pos[:len_pos_test], x_neg[:len_neg_test]]).sample(frac = 1, replace = False, random_state = 42).reset_index(drop = True)
    test_set = pd.concat([x_pos[len_pos_test:], x_neg[len_neg_test:]]).sample(frac = 1, replace = False, random_state = 42).reset_index(drop = True)


    x_train = train_set[['Monetary', 'Recency', 'Frequency']].values
    y_train = train_set['Churn'].values
    x_test = test_set[['Monetary', 'Recency', 'Frequency']].values
    y_test = test_set['Churn'].values


    return x_train, y_train, x_test, y_test



