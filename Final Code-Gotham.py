#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings 
warnings.filterwarnings('ignore')

import math
import time 
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import datetime as dt
import re
from haversine import haversine, Unit
from geopy.distance import geodesic
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

from sklearn.metrics.scorer import make_scorer
from sklearn import metrics

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LassoLarsCV

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, jaccard_score
from sklearn.neural_network import MLPRegressor


# In[2]:


df = pd.read_csv("Train.csv")


# In[3]:


df.head()


# In[4]:


#Dropoff Scatter Plot
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.lmplot('dropoff_x', 'dropoff_y', data=df, fit_reg=False, scatter_kws={"marker": "D","s": 10})


# In[5]:


#Pickup scatter plot
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.lmplot('pickup_x', 'pickup_y', data=df, fit_reg=False, scatter_kws={"marker": "D","s": 10})


# In[6]:


#Checking Density of the pickup and dropoff
start = time.time()
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2,2,figsize=(10, 10), sharex=False, sharey = False)
sns.despine(left=True)
sns.distplot(df['pickup_x'].values, label = 'pickup_latitude',color="m",bins = 100, ax=axes[0,0])
sns.distplot(df['pickup_y'].values, label = 'pickup_longitude',color="m",bins =100, ax=axes[0,1])
sns.distplot(df['dropoff_x'].values, label = 'dropoff_latitude',color="m",bins =100, ax=axes[1, 0])
sns.distplot(df['dropoff_y'].values, label = 'dropoff_longitude',color="m",bins =100, ax=axes[1, 1])
plt.setp(axes, yticks=[])
plt.tight_layout()
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))
plt.show()


# In[7]:


#testing to reduce the row of the data
train_data_new = df.copy()
train_data_new = train_data_new.loc[(train_data_new.pickup_x > 83.936425) & (train_data_new.pickup_x < 172.092496)]
train_data_new = train_data_new.loc[(train_data_new.pickup_y > 294.136100) & (train_data_new.pickup_y < 387.073991)]
train_data_new = train_data_new.loc[(train_data_new.dropoff_x > 91.124687) & (train_data_new.dropoff_x < 172.325441)]
train_data_new = train_data_new.loc[(train_data_new.dropoff_y > 287.035072) & (train_data_new.dropoff_y < 398.384175)]


# In[8]:


train_data_new.quantile([.05, .95]).loc[0.05]


# In[9]:


#Pickup spot after the data reduction
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.lmplot('pickup_y', 'pickup_x', data=train_data_new, fit_reg=False, scatter_kws={"marker": "D","s": 10})


# In[10]:


#Dropoff spot after the data reduction
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.lmplot('dropoff_y', 'dropoff_x', data=train_data_new, fit_reg=False, scatter_kws={"marker": "D","s": 10})


# In[11]:


#Define a plot function to check what quantile should I use for data reduction
def plot_lat_long(df, quantile):
    quantiles = df.quantile([quantile, 1-quantile])
    train_data_new = df.copy()
    train_data_new = train_data_new.loc[(df.pickup_x > quantiles.loc[quantile].pickup_x) & (df.pickup_x < quantiles.loc[1-quantile].pickup_x)]
    train_data_new = train_data_new.loc[(df.pickup_y > quantiles.loc[quantile].pickup_y) & (df.pickup_y < quantiles.loc[1-quantile].pickup_y)]
    train_data_new = train_data_new.loc[(df.dropoff_x > quantiles.loc[quantile].dropoff_x) & (df.dropoff_x < quantiles.loc[1-quantile].dropoff_x)]
    train_data_new = train_data_new.loc[(df.dropoff_y > quantiles.loc[quantile].dropoff_y) & (df.dropoff_y < quantiles.loc[1-quantile].dropoff_y)]
    start = time.time()
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), sharex=False, sharey = False)
    ax1.scatter('pickup_y', 'pickup_x', data=train_data_new, s=2)
    ax2.scatter('dropoff_y', 'dropoff_x', data=train_data_new, s=2)
    plt.setp(axes, yticks=[])
    plt.tight_layout()
    plt.show()


# In[12]:


plot_lat_long(df,0.005)


# In[13]:


plot_lat_long(df,0.008)


# In[14]:


plot_lat_long(df,0.05)


# In[15]:


#With the quantile of 0.008, it gives better of the picture of the city landscape
quantile = 0.008
quantiles = df.quantile([quantile, 1-quantile])
df_new = df.copy()
df_new = df_new.loc[(df_new.pickup_x > quantiles.loc[quantile].pickup_x) & (df_new.pickup_x < quantiles.loc[1-quantile].pickup_x)]
df_new = df_new.loc[(df_new.pickup_y > quantiles.loc[quantile].pickup_y) & (df_new.pickup_y < quantiles.loc[1-quantile].pickup_y)]
df_new = df_new.loc[(df_new.dropoff_x > quantiles.loc[quantile].dropoff_x) & (df_new.dropoff_x < quantiles.loc[1-quantile].dropoff_x)]
df_new = df_new.loc[(df_new.dropoff_y > quantiles.loc[quantile].dropoff_y) & (df_new.dropoff_y < quantiles.loc[1-quantile].dropoff_y)]


# In[16]:


df_new = df_new.reset_index()


# In[17]:


#Density with quantile of 0.008
start = time.time()
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2,2,figsize=(10, 10), sharex=False, sharey = False)
sns.despine(left=True)
sns.distplot(df_new['pickup_x'].values, label = 'pickup_latitude',color="m",bins = 100, ax=axes[0,0])
sns.distplot(df_new['pickup_y'].values, label = 'pickup_longitude',color="m",bins =100, ax=axes[0,1])
sns.distplot(df_new['dropoff_x'].values, label = 'dropoff_latitude',color="m",bins =100, ax=axes[1, 0])
sns.distplot(df_new['dropoff_y'].values, label = 'dropoff_longitude',color="m",bins =100, ax=axes[1, 1])
plt.setp(axes, yticks=[])
plt.tight_layout()
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))
plt.show()


# In[18]:


#Pickup with quantile 0.008
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.lmplot('pickup_y', 'pickup_x', data=df_new, fit_reg=False, scatter_kws={"marker": "D","s": 10})


# In[19]:


#Dropoff with quantile of 0.008
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.lmplot('dropoff_y', 'dropoff_x', data=df_new, fit_reg=False, scatter_kws={"marker": "D","s": 10})


# In[20]:


seed= 1
np.random.seed(seed)


# In[21]:


df_new.head()


# In[22]:


def euclideanDistance(_record):
    return sp.spatial.distance.euclidean([_record['pickup_x'], _record['pickup_y']], [_record['dropoff_x'], _record['dropoff_y']])
df_new['euclideanDistance'] = df_new.apply(lambda s: euclideanDistance(s), axis=1)
df_new['dist_long'] = df_new['pickup_x'] - df_new['dropoff_x']
df_new['dist_lat'] = df_new['pickup_y'] - df_new['dropoff_y']


# In[23]:


df_new['pickup_datetime'] = pd.to_datetime(df_new.pickup_datetime)


# In[24]:


df_new['month'] = df_new.pickup_datetime.dt.month
df_new['week'] = df_new.pickup_datetime.dt.week
df_new['weekday'] = df_new.pickup_datetime.dt.weekday
df_new['hour'] = df_new.pickup_datetime.dt.hour
df_new['minute'] = df_new.pickup_datetime.dt.minute
df_new['minute_oftheday'] = df_new['hour'] * 60 + df_new['minute']


# In[25]:


df_new.corr()


# In[26]:


kmeans_pickup = KMeans(n_clusters=10, random_state=42).fit(df_new[['pickup_x','pickup_y']])
pickup_clusters = kmeans_pickup.predict(df_new[['pickup_x','pickup_y']])
df_new["pickup_clusters"] = pickup_clusters


# In[27]:


kmeans_dropoff = KMeans(n_clusters = 10, random_state=42).fit(df_new[['dropoff_x','dropoff_y']])
dropoff_clusters = kmeans_dropoff.predict(df_new[['dropoff_x','dropoff_y']])
df_new["dropoff_clusters"] =dropoff_clusters


# In[28]:


df_new.head()


# In[29]:


df_new.corr()


# In[30]:


#heatmap
plt.figure(figsize=(14,12))
correlation = df_new.loc[:, df_new.columns != 'index'].corr()
sns.heatmap(correlation, linewidth=0.5, cmap='Blues')


# In[31]:


#df_with_dummies = pd.get_dummies(df_new, columns = ['Month', 'Hour', 'DayOfWeek', 'pickup_clusters', 'dropoff_clusters','NumberOfPassengers'])


# In[32]:


#df_with_dummies.info()


# In[33]:


df_with_dummies = df_new.drop([ 'index',
                                        'pickup_x','pickup_y','dropoff_x','dropoff_y','pickup_datetime',
                                       ], axis=1)


# In[34]:


df_with_dummies.info()


# In[35]:


y =  df_with_dummies['duration']
X = df_with_dummies.drop('duration',axis=1)


# In[36]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[37]:


Xcv,Xv,Zcv,Zv = train_test_split(X_val, y_val, test_size=0.4, random_state=1)


# In[38]:


X_train = X_train.dropna()
X_val = X_val.dropna()
y_train = y_train.dropna()
y_val = y_val.dropna()


# In[39]:


X_train.describe()


# In[41]:


from catboost import Pool, CatBoostRegressor
cbr = CatBoostRegressor(random_state=42)
cbr.fit(X_train, y_train)


# In[52]:


rf = RandomForestRegressor()
rf.fit(X_train, y_train)


# In[ ]:





# In[40]:


data_tr = xgb.DMatrix(X_train, label=y_train)
data_cv  = xgb.DMatrix(Xcv   , label=Zcv)
evallist = [(data_tr, 'train'), (data_cv, 'valid')]


# In[57]:


parms = {'max_depth':10, 
         'objective':'reg:linear',
         'eta'      :0.05,
         'subsample':0.8,
         'colsample_bylevel':1,
         'min_child_weight': 10,
         'nthread'  :3}
clf = xgb.train(parms, data_tr, num_boost_round=1000, evals = evallist,
                  early_stopping_rounds=100, maximize=False, 
                  verbose_eval=100)


# In[58]:


print('RMSE score = %1.5f, n_boost_round =%d.'%(clf.best_score,clf.best_iteration))


# In[59]:


data_test = xgb.DMatrix(X_val)
ztest = clf.predict(data_test)


# In[60]:


ytest = ztest
print(ytest[:10])


# In[45]:


df_test = pd.read_csv("Gotham_Test_Set.csv")


# In[ ]:





# In[46]:


df_test['euclideanDistance'] = df_test.apply(lambda s: euclideanDistance(s), axis=1)
df_test['dist_long'] = df_test['pickup_x'] - df_test['dropoff_x']
df_test['dist_lat'] = df_test['pickup_y'] - df_test['dropoff_y']

df_test['pickup_datetime'] = pd.to_datetime(df_test.pickup_datetime)
df_test['month'] = df_test.pickup_datetime.dt.month
df_test['week'] = df_test.pickup_datetime.dt.week
df_test['weekday'] = df_test.pickup_datetime.dt.weekday
df_test['hour'] = df_test.pickup_datetime.dt.hour
df_test['minute'] = df_test.pickup_datetime.dt.minute
df_test['minute_oftheday'] = df_test['hour'] * 60 + df_test['minute']

kmeans_pickup = KMeans(n_clusters=10, random_state=42).fit(df_test[['pickup_x','pickup_y']])
pickup_clusters = kmeans_pickup.predict(df_test[['pickup_x','pickup_y']])
df_test["pickup_clusters"] = pickup_clusters

kmeans_dropoff = KMeans(n_clusters = 10, random_state=42).fit(df_test[['dropoff_x','dropoff_y']])
dropoff_clusters = kmeans_dropoff.predict(df_test[['dropoff_x','dropoff_y']])
df_test["dropoff_clusters"] =dropoff_clusters



# In[47]:


df_test.info()


# In[48]:


xtest = df_test.drop(['pickup_x','pickup_y','dropoff_x','dropoff_y','pickup_datetime',
                                       ], axis=1)


# In[65]:


data_test = xgb.DMatrix(xtest)
ztest = clf.predict(data_test)
ytest = ztest
print(ytest[:10])


# In[49]:


y_pred_cbr = cbr.predict(xtest)


# In[53]:


y_pred_rf = rf.predict(xtest)


# In[66]:


submission = pd.DataFrame({'trip_duration': ytest})
submission.to_csv('submission.csv', index=False)


# In[43]:


actual = pd.read_csv('YTestG.csv')
pred = pd.read_csv('G_update.csv')


# In[3]:


actual.info()


# In[4]:


pred.info()


# In[11]:


#XG boost RMSE
math.sqrt(mean_squared_error(actual['duration'],pred['trip_duration']))


# In[50]:


#Catboost RMSE
math.sqrt(mean_squared_error(actual['duration'],y_pred_cbr))


# In[54]:


#Random Forest RMSE
math.sqrt(mean_squared_error(actual['duration'],y_pred_rf))


# In[ ]:




