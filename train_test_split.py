import pandas as pd
import math

def split_shuffle(df, test_ratio, positive, negative, flag):
    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Shuffle positives and negatives
    x_pos = df[df[f"{flag}"] == positive].sample(frac=1, random_state=42).reset_index(drop=True)
    x_neg = df[df[f"{flag}"] == negative].sample(frac=1, random_state=42).reset_index(drop=True)

    len_pos_test = math.floor(len(x_pos) * test_ratio)
    len_neg_test = math.floor(len(x_neg) * test_ratio)

    # First chunk goes to test, remaining to train
    test_set = pd.concat([x_pos[:len_pos_test], x_neg[:len_neg_test]]).sample(frac=1, random_state=42).reset_index(drop=True)
    train_set = pd.concat([x_pos[len_pos_test:], x_neg[len_neg_test:]]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Extract features and labels
    x_train = train_set[['Monetary', 'Recency', 'Frequency']].values
    y_train = train_set['Churn'].values
    x_test = test_set[['Monetary', 'Recency', 'Frequency']].values
    y_test = test_set['Churn'].values

    # Safety check: make sure no rows overlap
    train_rows = set([tuple(row) for row in x_train])
    test_rows = set([tuple(row) for row in x_test])
    overlap = train_rows & test_rows
    assert len(overlap) == 0, f"Train/Test overlap detected! Overlapping rows: {len(overlap)}"

    return x_train, y_train, x_test, y_test
