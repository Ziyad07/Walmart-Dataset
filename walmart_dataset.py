# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 19:58:31 2020

@author: ziyad
"""

# This file follows https://medium.com/analytics-vidhya/walmart-sales-forecasting-d6bd537e4904
# in some aspects

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
import pdb
#%matplotlib inline

features = pd.read_csv('retaildataset/Features data set.csv')

###################################################
# Replace NaN values with 0 in markdown columns
###################################################

features['CPI'] = features['CPI'].fillna(np.mean(features['CPI']))
features['Unemployment'] = features['Unemployment'].fillna(np.mean(features['Unemployment']))
features['MarkDown1'] = features['MarkDown1'].fillna(0)
features['MarkDown2'] = features['MarkDown2'].fillna(0)
features['MarkDown3'] = features['MarkDown3'].fillna(0)
features['MarkDown4'] = features['MarkDown4'].fillna(0)
features['MarkDown5'] = features['MarkDown5'].fillna(0)

sales = pd.read_csv('retaildataset/sales data-set.csv')
stores = pd.read_csv('retaildataset/stores data-set.csv')

#print(features.info())
#spread = features.describe()


df_merge = pd.merge(features, sales, on=['Store', 'Date', 'IsHoliday'], how='inner')
df_merge_2 = pd.merge(df_merge, stores, on=['Store'], how='inner')
df_merge_2['Date'] = [pd.datetime.strptime(d, '%d/%m/%Y') for d in df_merge_2['Date']]

final_features = df_merge_2.sort_values(by='Date')

#######################################################
# Do some EDA
grouped_stores = stores.groupby(['Type'])
sizes = grouped_stores.count()['Size'].round(1)
print(sizes)
store_sizes = float(stores['Size'].count())
store_labels = 'A Store','B Store', 'C Store'
sizes = [(sizes[0]/store_sizes)*100,(sizes[1]/store_sizes)*100,(sizes[2]/store_sizes)*100]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=store_labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


store_sales = pd.concat([stores['Type'], sales['Weekly_Sales']], axis=1)
f, ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x='Type', y='Weekly_Sales', data=store_sales, showfliers=False)

# total count of sales on holidays and non holidays
store_sales_holiday = pd.concat([sales['IsHoliday'], sales['Weekly_Sales']], axis=1)
f1, ax1 = plt.subplots(figsize=(8,6))
fig1 = sns.boxplot(x='IsHoliday', y='Weekly_Sales', data=store_sales_holiday, showfliers=False)

print('sales on non-holiday : ',sales[sales['IsHoliday']==False]['Weekly_Sales'].count().round(1))
print('sales on holiday : ',sales[sales['IsHoliday']==True]['Weekly_Sales'].count().round(1))

#  Plot correlation between features
corr = final_features.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=True)
plt.plot()

########################################################
# Expand date into more features 

day = df_merge_2['Date'].apply(lambda x: x.day)
month = df_merge_2['Date'].apply(lambda x: x.month)
year = df_merge_2['Date'].apply(lambda x: x.year)

df_merge_2['Day'] = day
df_merge_2['Month'] = month
df_merge_2['Year'] = year


# Later add columns for days to next to public holidays/spending seasons(Black friday)

# List of final features to use 
#final_features_targets = ['Store', 'Temperature', 'Fuel_Price', 'MarkDown1', 
#                  'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
#                  'CPI', 'Unemployment', 'IsHoliday_x', 'Dept', 'Type',
#                  'Day', 'Month', 'Year', 'Weekly_Sales'
#                  ]
#                  
#df_merge_3 = df_merge_2[final_features_targets]                  

##final_data_frame_targets = df_merge_2['Weekly_Sales']
#final_data_frame_features = df_merge_3.sort_values(by=['Year', 'Month', 'Day', 'Store', 'Dept'], ascending=True)

#categorical_features = final_data_frame_features.select_dtypes(include=[np.object])
#numerical_features = final_data_frame_features.select_dtypes(include=[np.number])

categorical_features_type = pd.get_dummies(final_features['Type'])
final_features  = final_features.join([categorical_features_type])
final_features = final_features.drop(['Type'], axis=1)
final_features = final_features.drop(['Date'], axis=1)

#store = pd.get_dummies(final_data_frame_features['Store'])
#store = store.rename(columns=lambda s: 'store_'+str(s))
#
#dept = pd.get_dummies(final_data_frame_features['Dept'])
#dept = dept.rename(columns=lambda s: 'dept_'+str(s))
#
#holiday = pd.get_dummies(final_data_frame_features['IsHoliday_x'])
#types = pd.get_dummies(final_data_frame_features['Type'])
#
#final_data_frame_features = final_data_frame_features.drop(['Store', 'Dept', 'IsHoliday_x', 'Type'], axis=1)
#final_data_frame_features  = final_data_frame_features.join([store, dept, holiday, types])

def get_metrics(targets, preds):
    mse = np.sqrt(metrics.mean_squared_error(targets, preds))
    r2_score = metrics.r2_score(targets, preds)
    mae = metrics.mean_absolute_error(targets, preds)
    
    return mse, r2_score, mae


def plot_model_results(targets, preds):
    plt.figure()
    plt.scatter(targets, preds)
    plt.xlabel("Actual")
    plt.ylabel("Preds")
    plt.title("Actual vs. Preds")
    plt.show()
    
# Create model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

################################################
# TrainTest Split
train, test = train_test_split(final_features, test_size = 0.20)
print(len(train))
print(len(test))

train_features = train.drop(['Weekly_Sales'], axis=1)
train_targets = train.drop(list(train_features), axis=1)

test_features = test.drop(['Weekly_Sales'], axis=1)
test_targets = test.drop(list(test_features), axis=1)

print(train_features.head(5))
print(train_targets.head(5))
################################################

################################################
# Normalise Data
sc_X = StandardScaler()
train_features = sc_X.fit_transform(train_features)
test_features = sc_X.transform(test_features)
################################################


################################################
# Lets try linear regression as a first pass baseline
################################################
lr = LinearRegression(normalize=True)
lr.fit(train_features, train_targets)
lr_preds = lr.predict(train_features)
mse_l, r2_score_l, mae_l = get_metrics(train_targets, lr_preds)
plot_model_results(train_targets, lr_preds)
accuracy_l = lr.score(test_features, test_targets)
print("Linear Regression Accuracy: %f" % (accuracy_l))

################################################
## Now lets try XGBoost
################################################

xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05)
xgb_model.fit(train_features, train_targets)
#xgb_preds = xgb_model.predict(train_features)
#plot_model_results(train_targets, xgb_preds)
#mse, r2_score, mae = get_metrics(train_targets, xgb_preds)
xgb_test_preds = xgb_model.predict(test_features)
plot_model_results(test_targets, xgb_test_preds)
mse_x, r2_score_x, mae_x = get_metrics(test_targets, xgb_test_preds)
accuracy_x = xgb_model.score(test_features, test_targets)
print("XGBOOST Accuracy: %f" % (accuracy_x))

################################################
## Now lets try Decision Tree Regressor
################################################

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=0)
dt.fit(train_features,train_targets)
DTR_preds = dt.predict(test_features)
plot_model_results(test_targets, DTR_preds)
mse_DTR, r2_score_DTR, mae_DTR = get_metrics(test_targets, DTR_preds)
accuracy_DTR = dt.score(test_features, test_targets)
print("DTR Accuracy: %f" % (accuracy_DTR))

################################################
#  Randomforest
################################################
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 400,max_depth=15,n_jobs=5)        
rfr.fit(train_features,train_targets)
rf_pred=rfr.predict(test_features)
plot_model_results(test_targets, rf_pred)
mse_rf, r2_score_rf, mae_rf = get_metrics(test_targets, rf_pred)
accuracy_rf = rfr.score(test_features, test_targets)
print("RandomForrest Accuracy: %f" % (accuracy_rf))

################################################
# Now lets try ExtraTreeRegressor 
################################################

from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor(n_estimators=30,n_jobs=4) 
etr.fit(train_features,train_targets)
etr_pred=etr.predict(test_features)
plot_model_results(test_targets, etr_pred)
mse_et, r2_score_et, mae_et = get_metrics(test_targets, etr_pred)
accuracy_et = etr.score(test_features, test_targets)
print("Extra Trees Regressor Accuracy: %f" % (accuracy_et))

################################################
final_preds = (xgb_test_preds + DTR_preds + rf_pred + etr_pred)/4.0
plot_model_results(test_targets, final_preds)
mse_final, r2_score_final, mae_final = get_metrics(test_targets, final_preds)


from prettytable import PrettyTable
    
x = PrettyTable()
x.field_names = ["Model", "MAE", "RMSE", "Accuracy"]
x.add_row(["Linear Regression (Baseline)", mae_l, r2_score_l, accuracy_l])
x.add_row(["XGBRegressor", mae_x, r2_score_x, accuracy_x])
x.add_row(["DecisionTreeRegressor", mae_DTR, r2_score_DTR, accuracy_DTR])
x.add_row(["RandomForestRegressor", mae_rf, r2_score_rf, accuracy_rf])
x.add_row(["ExtraTreeRegressor", mae_et, r2_score_et, accuracy_et])

print(x)



#def pairplot(dataframe):
#    plt.figure()
#    sns.pairplot(dataframe)
#    display(plt.show())

# Convert to categorical etc.

def plot_corr_vars(df):

    plt.matshow(df.corr())
    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df.columns)), df.columns)
    plt.colorbar()
    plt.show()    
#plot_corr_vars(df_merge_2)

def plot_corr_2Vars(x, y):
    print(np.corrcoef(x,y))
    matplotlib.style.use('ggplot')
    plt.scatter(x, y)
    plt.show()
#plot_corr_2Vars(final_data_frame_features['Unemployment'], final_data_frame_targets)


def plot_correlation(df):
    f = plt.figure(figsize=(11, 8))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=8, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=8)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=8)
#plot_correlation(df_merge_2)

