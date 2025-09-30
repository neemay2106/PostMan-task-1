INTRODUCTION

Churn rate measures the proportion of customers who stop using a service or product over a given period. In this analysis, we consider a period of 180 days (6 months), and a customer who has not made a purchase within this period is 
considered churned. Using the customers, orders, and payments datasets, we computed RFM (Recency, Frequency, Monetary) features for each customer to summarize their purchasing behavior. Two models—Logistic Regression and 
Random Forest Classifier—were trained to predict whether a customer is likely to churn.

Data Understanding and Preparation

Datasets Used:
olist_customers
olist_orders
olist_order_payments

Joins:
Orders were joined to customers on customer_id.
Payments were joined to orders on order_id.

Granularity Choice:
We used customer_unique_id instead of customer_id, since multiple customers can share the same customer_id.

Filtering:
The dataset contained order statuses: ['delivered', 'invoiced', 'shipped', 'processing', 'unavailable', 'canceled', 'created', 'approved'].
Since the analysis focuses on completed transactions, we restricted the dataset to orders with status delivered. This ensures consistency in calculating customer purchase behavior, as incomplete or canceled orders do not reflect actual activity.

Feature Engineering

We constructed an RFM dataset to predict whether a customer is churned or not:
Recency: Number of days between a customer’s last order and the reference date (the latest order in the dataset).

Frequency: Total number of delivered orders per customer.
Monetary: Total amount spent by the customer, calculated by summing all payments within the given period.
Churn: A customer is labeled as churned (churn = 1) if Recency > 180 days; otherwise, churn = 0.
Class Imbalance
The dataset is slightly imbalanced, with 59% of customers classified as churners and 41% as non-churners. To address this, stratified shuffling was applied during the train-test split, maintaining the original class 
distribution in both training and test sets.

Log Transformation
Log transformation was applied to reduce skewness. This method adjusts small values slightly while significantly reducing large values, helping to produce a more balanced distribution.
Scaling
We used Min-Max normalization to scale the features between [0,1], preserving relative ordering and simplifying comparison:

Data Splitting
Train-Test Split
The RFM dataset was split into training and test sets using an 80/20 ratio. To avoid sequential bias, the data was shuffled prior to splitting. Stratified shuffling ensured that both sets maintained the same proportion of churned and 
non-churned customers.
Challenges:
 Initially, we explored using Python’s random module to shuffle the data, but eventually we leveraged Pandas’ sample method for efficient and straightforward shuffling while preserving dataset structure.

Modeling

1) Logistic Regression
Logistic Regression is a widely used classification algorithm that predicts the probability of a binary outcome. Despite its name, it is primarily used for classification rather than regression. In this analysis, Logistic Regression
serves as a baseline model.
Model Implementation:
Sigmoid Function: Maps predicted values to the range [0,1], taking the weighted sum of features as input and outputting the probability of churn.

Loss and Cost Functions: Penalize predictions far from the actual label and reward accurate predictions. The cost function sums the loss across all samples and divides by the number of samples.

Gradient Descent: Partial derivatives of the cost function with respect to weights and bias were computed iteratively to update the parameters for optimal predictions.

Threshold Selection: Using the default threshold of 0.5 produced a lower F1 score. A ROC curve was plotted to determine the optimal threshold of 0.62, improving the balance between precision and recall.

Implementation Challenge:
 Initially, gradient descent was implemented by looping over each row and feature, which was extremely slow due to the dataset size. Vectorizing the computations replaced loops with matrix operations, significantly improving performance.

2) Random Forest Classifier
The Random Forest Classifier builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting. As a non-linear model, it captures complex relationships between RFM features and customer churn.
Model Implementation:
Decision Trees: Entropy was used as the splitting criterion. Stopping conditions, such as maximum depth and minimum samples per node, were set to prevent overfitting.


Random Forest: Each tree was trained on a bootstrap sample of the training data. The number of trees was chosen to balance accuracy and efficiency.

Implementation Challenge:
 The initial best-fit function iterated over all rows for each feature to determine the optimal split, causing slow training. A sweep method was implemented, evaluating thresholds only where the label changes, significantly reducing 
 computation time.
Outcome:
 Random Forest effectively captured non-linear patterns and interactions among RFM features. Vectorized threshold calculations and bootstrap sampling improved performance, making it practical for the full dataset.
3) Cross-Validation
10-fold cross-validation was applied to both models to evaluate stability and generalization.
Logistic Regression F1 per fold: [96.3, 94.3, 98.8, 97.2, 98.9, 96.2, 96.3, 96.4, 94.9, 94.8]

Random Forest F1 per fold: [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

Results & Evaluation

Logistic Regression:
Accuracy: 0.96
Precision: 1.0
Recall: 0.933
F1-core: 0.965
Confusion Matrix: [[24709, 0], [1776, 18060]]

Random Forest Classifier:
Accuracy, Precision, Recall, F1-score: 1.0
Confusion Matrix: [[26485, 0], [0, 18060]]

Interpretation:
 While Random Forest outperforms Logistic Regression numerically, the perfect results suggest potential over-reliance on dominant features or data leakage.
Limitations
Despite high performance, the results may not fully reflect predictive ability. Frequency shows a high correlation (0.78) with churn, whereas Monetary and other features have near-zero correlation (-0.02, -0.01). 
This indicates the models, particularly Random Forest, may rely heavily on one dominant feature rather than learning robust patterns.
Data leakage is another concern, potentially allowing models to memorize the training data. Repeated customer IDs were checked, but further leakage investigation is recommended.

Next Steps
Revisit Feature Engineering: Ensure no feature inadvertently reveals the target. Explore new features capturing subtle customer behavior.
Check for Data Leakage: Confirm that no information from the test set is present in training data, including overlapping timestamps or repeated customer IDs.
Model Adjustments: Modify model training to prevent memorization, such as limiting tree depth or reducing highly correlated features.
Focus on Data Understanding: Prioritize data exploration and cleaning before tuning models to ensure robust and reliable predictions.



Reproducibility Checklist

Random Seeds:

A fixed random seed (21) was used for all operations involving randomness, ensuring consistent results.

Environment Details:

Python Version: 3.13.4

Key Libraries: pandas, numpy, matplotlib, seaborn


Clear Run Instructions:
 a. Place all datasets (olist_customers_dataset.csv, olist_orders_dataset.csv, olist_order_payments_dataset.csv) in the same folder as the notebook or update file paths accordingly.
 b. Ensure all custom modules and classes are imported:

LogisticReasonModel

RandomForestClassifier

train_test_split

CrossValidation
 c. Run Main.py sequentially to reproduce the full workflow, from data preparation to model evaluation.



