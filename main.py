import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import math
import random
from logistic_reason_model import LogisticReasonModel
import numpy as np
from train_test_split import split_shuffle
from K_Fold import CrossValidation
from Model2 import RandomForestClassifier

SEED = 21
np.random.seed(SEED)
random.seed(SEED)


customers = pd.read_csv('/Users/neemayrajan/Desktop/PostMan task 1 /E-commerece data set by olist 2/olist_customers_dataset.csv')
orders = pd.read_csv('/Users/neemayrajan/Desktop/PostMan task 1 /E-commerece data set by olist 2/olist_orders_dataset.csv')
payments = pd.read_csv('/Users/neemayrajan/Desktop/PostMan task 1 /E-commerece data set by olist 2/olist_order_payments_dataset.csv')


def scaling_function(x_given,x_min,x_max,list_ap):
    x_scaled = (x_given-x_min)/(x_max -x_min)
    list_ap.append(x_scaled)


orders = orders[orders['order_status'] == 'delivered']
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'] , format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
reference_date = orders['order_purchase_timestamp'].max()
payments = payments.groupby('order_id')['payment_value'].sum().reset_index() #dropped all colmns except order id and payment_value
merge_oc = pd.merge(customers, orders, how='inner', on='customer_id')

final_dataset = pd.merge(merge_oc , payments , how = 'inner' , on = 'order_id')
final_dataset = final_dataset.drop(columns=['customer_zip_code_prefix','order_approved_at',
       'order_delivered_carrier_date', 'order_delivered_customer_date',
       'order_estimated_delivery_date'], axis = 1)

#Frequency
f = final_dataset.groupby('customer_unique_id')['order_id'].count().reset_index()
frequency_log_trans = [math.log(n,10) for n in f["order_id"].tolist()]
frequency = []
rf_max = max(frequency_log_trans)
rf_min = min(frequency_log_trans)
for val in frequency_log_trans:
    scaling_function(val,rf_min,rf_max,frequency)


#Recency
R = final_dataset.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
r = []
churn = []
for t in R['order_purchase_timestamp'].to_list():
    log_tran_R = int((reference_date - t).days)
    if log_tran_R > 0:
       r.append(math.log(log_tran_R, 10))
    else:
        r.append(math.log(1, 10))

for y in r:
       if y> 2.25:
              churn.append(1)
       else:
              churn.append(0)
recency = []
rr_max = max(r)
rr_min = min(r)
for val in r:
    scaling_function(val,rr_min,rr_max,recency)

#Monetary
m1 = final_dataset.groupby('customer_unique_id')['payment_value'].sum().reset_index()
m = m1['payment_value'].tolist()
monetary = []
monetary_log_trans= []
for value in m:
    log_trans_m = math.log(value, 10)
    monetary_log_trans.append(log_trans_m)
rm_max = max(monetary_log_trans)
rm_min = min(monetary_log_trans)
for val in monetary_log_trans:
    scaling_function(val,rm_min,rm_max,monetary)

#RFM Dataset
comb_dict = {"customer_unique_id": final_dataset.customer_unique_id.unique().tolist(),
             "Monetary": monetary,
             "Recency": recency,
             "Frequency": frequency,
             "Churn": churn}

final = pd.DataFrame.from_dict(comb_dict)
final = final.drop_duplicates(subset=['Monetary', 'Recency', 'Frequency', 'Churn']).reset_index(drop=True)



#Input Validation
assert final["customer_unique_id"].isna().sum() == 0
assert final["customer_unique_id"].is_unique
assert final["Monetary"].between(0,1).all()
assert final["Recency"].between(0,1).all()
assert final["Frequency"].between(0,1).all()
assert set(final["Churn"].unique()).issubset({0,1})

x_train, y_train, x_test, y_test = split_shuffle(df = final , test_ratio = 0.5 ,flag = "Churn" , positive = 1 , negative = 0)
churn_count = pd.Series(y_test).value_counts()
print(f"Churn count:\n {churn_count}")

#check for Overlap
train_rows = set([tuple(row) for row in x_train])
test_rows = set([tuple(row) for row in x_test])
overlap = train_rows & test_rows
print("Number of overlapping rows:", len(overlap))

# Extract X and y
x = final[['Monetary', 'Recency', 'Frequency']].values
y = final['Churn'].values
m_train, n_train = x_train.shape
m_test, n_test = x_test.shape


#Logistic Regression Model
model = LogisticReasonModel()
w, b =  model.gradient_function(x = x_train,y = y_train, alpha=0.01, iterations=1000)

model.gradient_function(x = x_train,y = y_train, alpha=0.01, iterations=1000)
pred =  model.predict(x = x_test)

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



fpr = (fp/(fp + tn)) if (tn +fp)>0 else 0
recall = (tp/(tp+fn)) if (tp +fn)>0 else 0
precision = (tp / (tp + fp)) if (tp +fp)>0 else 0
accuracy = np.mean(pred == y_test)
f1 = 2*(recall*precision)/(recall+precision) if (recall+ precision)>0 else 0
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
confusion_matrix_1 = np.array([[tp, fp],
                        [fn, tn]])

print(confusion_matrix_1)


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
f1_2 = 2 * (recall_2 * precision_2) / (recall_2 + precision_2) if (recall_2 + precision_2) > 0 else 0
print("Accuracy:", acc_2)
print("Precision:", precision_2)
print("Recall:", recall_2)
print("F1-score:", f1_2)
confusion_matrix_2 = np.array([[tp_2, fp_2],
                        [fn_2, tn_2]])

print(confusion_matrix_2)

#cross Validating

Cross_1 = CrossValidation()
best_m , scores = Cross_1.split(x = x_train, y = y_train,model_class = LogisticReasonModel,args = (.01,1000) )
print("F1 per fold:", scores)

Cross_1 = CrossValidation()
best_m_2 , scores = Cross_1.split(x = x_train, y = y_train,model_class = RandomForestClassifier)
print("F1 per fold:", scores)

data = pd.DataFrame(final[['Monetary', 'Recency', 'Frequency','Churn']])
corr = data.corr()
plt.figure(figsize=(12, 8))  # Adjust size as needed
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

labels = [0, 1]
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix_1, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()