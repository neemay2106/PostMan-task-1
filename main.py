import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import random
from logistic_reason_model import LogisticReasonModel
import numpy as np
from train_test_split import TrainTestSplit
from K_Fold import CrossValidation
from Model2 import RandomForestClassifier

SEED = 21
np.random.seed(SEED)
random.seed(SEED)

FPR = []
Recall = []
#Load Datasets
customers = pd.read_csv('/Users/neemayrajan/Desktop/PostMan task 1 /E-commerece data set by olist 2/olist_customers_dataset.csv')
orders = pd.read_csv('/Users/neemayrajan/Desktop/PostMan task 1 /E-commerece data set by olist 2/olist_orders_dataset.csv')
payments = pd.read_csv('/Users/neemayrajan/Desktop/PostMan task 1 /E-commerece data set by olist 2/olist_order_payments_dataset.csv')


#Feature Engineering
orders = orders[orders['order_status'] == 'delivered']
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'] , format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
reference_date = orders['order_purchase_timestamp'].max()
print(reference_date)
payments = payments.groupby('order_id')['payment_value'].sum().reset_index() #dropped all colmns except order id and payment_value
merge_oc = pd.merge(customers, orders, how='inner', on='customer_id')

final_dataset = pd.merge(merge_oc , payments , how = 'inner' , on = 'order_id')
final_dataset = final_dataset.drop(columns=['customer_zip_code_prefix','order_approved_at',
       'order_delivered_carrier_date', 'order_delivered_customer_date',
       'order_estimated_delivery_date'], axis = 1)
#Frequency
f = final_dataset.groupby('customer_unique_id')['order_id'].count().reset_index()


#Recency
R = final_dataset.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
R['order_purchase_timestamp'] = reference_date - R['order_purchase_timestamp']
R['Recency'] = R['order_purchase_timestamp'].dt.days
R['Churn'] = (R['Recency'] > 180).astype(int)
R = R[['customer_unique_id', 'Recency', 'Churn']]

#Monetary
m1 = final_dataset.groupby('customer_unique_id')['payment_value'].sum().reset_index()



rfm_df = R.merge(f, on='customer_unique_id').merge(m1, on='customer_unique_id')
rfm_df.rename(columns={'payment_value': 'Monetary'}, inplace=True)
rfm_df.rename(columns={'order_id': 'Frequency'}, inplace=True)
#Heat Map for the RFM dataset
data = pd.DataFrame(rfm_df[['Monetary', 'Recency', 'Frequency','Churn']])
corr = data.corr()
plt.figure(figsize=(12, 8))  # Adjust size as needed
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


#RFM Dataset




#Input Validation

train_test_split = TrainTestSplit()
x_train, y_train, x_test, y_test = train_test_split.split_shuffle(df = rfm_df , test_ratio = 0.5 ,flag = "Churn" , positive = 1 , negative = 0)
print(type(x_train))
# print(x_test.head())



m_train, n_train = x_train.shape
m_test, n_test = x_test.shape


#Logistic Regression Model
j = []
thresholds = []
Precision = []
model = LogisticReasonModel()
model.gradient_function(x = x_train,y = y_train, alpha=0.01, iterations=1000)
for t in np.arange(0,1.01,.01):
    thresholds.append(t)
    pred =  model.predict(x = x_test,t = t)
    tp = fp = tn = fn= 0
    for i in range(len(pred)):
        if pred[i] == y_test[i] and pred[i] == 1 :
            tp += 1
        elif pred[i] == 1 and y_test[i] == 0:
            fp += 1
        elif  pred[i] == 0 and y_test[i] == 1:
            fn += 1
        else:
            tn += 1
    precision = (tp / (tp + fp)) if (tp +fp)>0 else 0
    Precision.append(precision)
    fpr = (fp/(fp + tn)) if (tn +fp)>0 else 0
    FPR.append(fpr)
    recall = (tp/(tp+fn)) if (tp +fn)>0 else 0
    Recall.append(recall)
    j.append(recall-fpr)
best_idx = np.argmax(j)
best_threshold = thresholds[best_idx]
print(f"Best threshold: {best_threshold}")
plt.plot(FPR, Recall)
plt.show()
pred = model.predict(x=x_test, t=best_threshold)
tp = fp = tn = fn = 0
for i in range(len(pred)):
    if pred[i] == y_test[i] and pred[i] == 1:
        tp += 1
    elif pred[i] == 1 and y_test[i] == 0:
        fp += 1
    elif pred[i] == 0 and y_test[i] == 1:
        fn += 1
    else:
        tn += 1
recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
accuracy = (tp +tn) / (tp + tn+ fn+ fp) if (tp + tn+ fn+ fp) > 0 else 0
precision = (tp / (tp + fp)) if (tp +fp)>0 else 0
f1 = 2*(Recall[best_idx]*Precision[best_idx])/(Recall[best_idx]+Precision[best_idx]) if (Recall[best_idx]+ Precision[best_idx])>0 else 0
confusion_matrix_lr = np.array([[tp,fp],[fn,tn]])
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:",confusion_matrix_lr)


#Cross Validation for logistic Regression
Cross = CrossValidation()
best_lr , scores_lr = Cross.split(x = x_train, y = y_train,model_class = LogisticReasonModel,args = (.01,1000),pred_args= (0.59,) )
print("F1 per fold:", scores_lr)
print("Best Model:", best_lr)

#Model 2 Random Forest Classifier
model_2 = RandomForestClassifier(
    n_trees=50,
    max_depth=5,
    min_samples=20,
    features=int(np.sqrt(3))
)
model_2.fit(x = x_train, y = y_train)
pred2 = model_2.predict(x_test)
tp_2 = fp_2 = tn_2 = fn_2 = 0
for i in range(len(pred2)):
    if pred2[i] == y_test[i] and pred2[i] == 1:
        tp_2 += 1
    elif pred2[i] == 1 and y_test[i] == 0:
        fp_2 += 1
    elif pred2[i] == 0 and y_test[i] == 1:
        fn_2 += 1
    else:
        tn_2 += 1
fpr_2 = (fp_2 / (fp_2 + tn_2)) if (fp_2 + tn_2) > 0 else 0
recall_2 = (tp_2 / (tp_2 + fn_2)) if (tp_2 + fn_2) > 0 else 0
precision_2 = (tp_2 / (tp_2 + fp_2)) if (tp_2 + fp_2) > 0 else 0
acc_2 = np.mean(pred2 == y_test)
confusion_matrix_rf = np.array([[tp_2,fp_2],[fn_2,tn_2]])
f1_2 = 2 * (recall_2 * precision_2) / (recall_2 + precision_2) if (recall_2 + precision_2) > 0 else 0
print("Accuracy:", acc_2)
print("Precision:", precision_2)
print("Recall:", recall_2)
print("F1-score:", f1_2)
print("Confusion Matrix:",confusion_matrix_rf)

#Cross Validation for Random Forest
print(type(x_train))
best_rf , scores_rf = Cross.split(x = x_train, y = y_train,model_class = RandomForestClassifier)
print("F1 per fold:", scores_rf)
print("Best Model:", best_rf)
