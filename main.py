import pandas as pd
import datetime
import seaborn as sns
from matplotlib import pyplot as plt
import math
from logistic_reason_model import LogisticReasonModel
import numpy as np
from train_test_split import split_shuffle
from K_Fold import CrossValidation
from sklearn.metrics import roc_curve, auc

FPR = []
Recall = []
Precision = []
F1 = []
Accuracy = []
J_statistic = []

customers = pd.read_csv('/Users/neemayrajan/Desktop/PostMan task 1 /E-commerece data set by olist 2/olist_customers_dataset.csv')
orders = pd.read_csv('/Users/neemayrajan/Desktop/PostMan task 1 /E-commerece data set by olist 2/olist_orders_dataset.csv')
payments = pd.read_csv('/Users/neemayrajan/Desktop/PostMan task 1 /E-commerece data set by olist 2/olist_order_payments_dataset.csv')


def scaling_function(x,x_min,x_max,list_ap):
    x_scaled = (x-x_min)/(x_max -x_min)
    list_ap.append(x_scaled)


print(customers["customer_unique_id"].duplicated().sum())
orders = orders[orders['order_status'] == 'delivered']
print(orders['order_status'].unique())
print(orders['order_purchase_timestamp'].dtype)
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'] , format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
reference_date = orders['order_purchase_timestamp'].max()
print(reference_date)
print(orders.loc[orders['order_purchase_timestamp'] == reference_date, 'customer_id'])
print(payments['order_id'].duplicated().sum())
payments = payments.groupby('order_id')['payment_value'].sum().reset_index() #dropped all colmns except order id and payment_value
print(payments['order_id'].duplicated().sum())
print(payments.columns)

merge_oc = pd.merge(customers, orders, how='inner', on='customer_id')
print(merge_oc.head())

print(merge_oc['order_approved_at'].head())

final_dataset = pd.merge(merge_oc , payments , how = 'inner' , on = 'order_id')
final_dataset = final_dataset.drop(columns=['customer_zip_code_prefix','order_approved_at',
       'order_delivered_carrier_date', 'order_delivered_customer_date',
       'order_estimated_delivery_date'], axis = 1)
print(final_dataset['payment_value'].dtype)
print(final_dataset['order_purchase_timestamp'].dtype)
#F
f = final_dataset.groupby('customer_unique_id')['order_id'].count().reset_index()
frequency_log_trans = [math.log(n,10) for n in f["order_id"].tolist()]
frequency = []
rf_max = max(frequency_log_trans)
rf_min = min(frequency_log_trans)
for val in frequency_log_trans:
    scaling_function(val,rf_min,rf_max,frequency)


#R
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

#M
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

print(final_dataset.head())
print(final_dataset.columns)


comb_dict = {"customer_unique_id": final_dataset.customer_unique_id.unique().tolist(),
             "Monetary": monetary,
             "Recency": recency,
             "Frequency": frequency,
             "Churn": churn}

final = pd.DataFrame.from_dict(comb_dict)
print(final.head())

x_train, y_train, x_test, y_test = split_shuffle(df = final , test_ratio = 0.2 ,flag = "Churn" , positive = 1 , negative = 0 )



# Extract X and y
X = final[['Monetary', 'Recency', 'Frequency']].values
y = final['Churn'].values   # <-- Make sure it's 1D

m_train, n_train = x_train.shape
m_test, n_test = x_test.shape

model = LogisticReasonModel()
# Train
# w, b =  model.gradient_function(x = x_train,y = y_train, alpha=0.01, iterations=1000)

# Predictions
# for j in np.arange(0, 1, 0.01):


y_axis = []
Cross = CrossValidation()
w,b = Cross.split(x = x_train,y = y_train, alpha=0.01, iterations=1000)
pred =  model.predict(x = x_test,threshold = .62,w = w,b =b)

tp = 0
fp = 0
tn = 0
fn = 0
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
FPR.append(fpr)
recall = (tp/(tp+fn)) if (tp +fn)>0 else 0
Recall.append(recall)

precision = (tp / (tp + fp)) if (tp +fp)>0 else 0
Precision.append(precision)

accuracy = np.mean(pred == y_test)
Accuracy.append(accuracy)
    #

f1 = 2*(recall*precision)/(recall+precision) if (recall+ precision)>0 else 0
F1.append(f1)
    #
#     J_statistic.append(recall-fpr)
#
# index_stat = J_statistic.index(max(J_statistic))
# threshold = y_axis[index_stat]



# plt.plot(Recall,Precision , label = 'PR')
# plt.show()
# plt.plot(F1,y_axis)
# plt.show()

# if 0 in pred and 1 in pred:
#     print("Array contains both 0 and 1")
# elif 1 in pred:
#     print("Only 1s found")
# elif 0 in pred:
#     print("Only 0s found")
# else:
#     print("Array is empty")



