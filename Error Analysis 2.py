#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
import pandas as pd


# In[92]:


df = pd.read_csv("C:\\Users\\35383\\Documents\\Master\\Thesis\\data\\erroranalysis.csv")


# In[91]:


locations = pd.read_csv("C:\\Users\\35383\\Documents\\Master\\Thesis\\data\\features_nlp2.csv")


# In[93]:


df['start_distance_x'] = locations['start_distance_x']
df['start_location_y'] = locations['start_location_y']


# In[94]:


df


# # create training and test dfs

# In[95]:


training = df.head(12500)
testing = df.tail(4140)


# In[100]:


x_train = training.iloc[:, 0:32].values
y_train = training.iloc[:, 32].values


# In[103]:


x_test = testing.iloc[:, 0:32].values
y_test = testing.iloc[:, 32].values


# # XGBoost model

# In[63]:


from xgboost import XGBClassifier
xg = XGBClassifier(objective = 'binary:logistic')


# In[102]:


xg.fit(x_train, y_train)


# # seperate test df into pitch zones

# In[66]:


from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

accuracy_xg = accuracy_score(y_test, y_pred_xg)
f1_xg = f1_score(y_test, y_pred_xg)
print("Accuracy: ", accuracy_xg)


# In[110]:


df1 = testing[testing['start_distance_x'] > 80]
df2 = testing[(testing['start_distance_x'] > 40) & (testing['start_distance_x'] <= 80)]
df3 = testing[(testing['start_distance_x'] > 0) & (testing['start_distance_x'] <= 40)]


# In[114]:


df1L = df1[(df1['start_location_y'] >=0) & (df1['start_location_y'] <= 25)]
df1C = df1[(df1['start_location_y'] > 25) & (df1['start_location_y'] <= 55)]
df1R = df1[(df1['start_location_y'] > 55) & (df1['start_location_y'] <= 80)]

df2L = df2[(df2['start_location_y'] >=0) & (df2['start_location_y'] <= 25)]
df2C = df2[(df2['start_location_y'] > 25) & (df2['start_location_y'] <= 55)]
df2R = df2[(df2['start_location_y'] > 55) & (df2['start_location_y'] <= 80)]

df3L = df3[(df3['start_location_y'] >=0) & (df3['start_location_y'] <= 25)]
df3C = df3[(df3['start_location_y'] > 25) & (df3['start_location_y'] <= 55)]
df3R = df3[(df3['start_location_y'] > 55) & (df3['start_location_y'] <= 80)]


# In[115]:


df1L


# In[117]:


df1Lx = df1L.iloc[:, 0:32].values
df1Cx = df1C.iloc[:, 0:32].values
df1Rx = df1R.iloc[:, 0:32].values

df2Lx = df2L.iloc[:, 0:32].values
df2Cx = df2C.iloc[:, 0:32].values
df2Rx = df2R.iloc[:, 0:32].values

df3Lx = df3L.iloc[:, 0:32].values
df3Cx = df3C.iloc[:, 0:32].values
df3Rx = df3R.iloc[:, 0:32].values


# In[118]:


df1Ly = df1L.iloc[:, 32].values
df1Cy = df1C.iloc[:, 32].values
df1Ry = df1R.iloc[:, 32].values

df2Ly = df2L.iloc[:, 32].values
df2Cy = df2C.iloc[:, 32].values
df2Ry = df2R.iloc[:, 32].values

df3Ly = df3L.iloc[:, 32].values
df3Cy = df3C.iloc[:, 32].values
df3Ry = df3R.iloc[:, 32].values


# In[119]:


pred1L = xg.predict(df1Lx)
pred1C = xg.predict(df1Cx)
pred1R = xg.predict(df1Rx)
pred2L = xg.predict(df2Lx)
pred2C = xg.predict(df2Cx)
pred2R = xg.predict(df2Rx)
pred3L = xg.predict(df3Lx)
pred3C = xg.predict(df3Cx)
pred3R = xg.predict(df3Rx)


# In[120]:


accuracy_1L = accuracy_score(df1Ly, pred1L)
accuracy_1C = accuracy_score(df1Cy, pred1C)
accuracy_1R = accuracy_score(df1Ry, pred1R)
accuracy_2L = accuracy_score(df2Ly, pred2L)
accuracy_2C = accuracy_score(df2Cy, pred2C)
accuracy_2R = accuracy_score(df2Ry, pred2R)
accuracy_3L = accuracy_score(df3Ly, pred3L)
accuracy_3C = accuracy_score(df3Cy, pred3C)
accuracy_3R = accuracy_score(df3Ry, pred3R)

print("Accuracy defensive third left: ", accuracy_1L)
print("Accuracy defensive third centre: ", accuracy_1C)
print("Accuracy defensive third right: ", accuracy_1R)
print("Accuracy midfield third left: ", accuracy_2L)
print("Accuracy midfield third centre: ", accuracy_2C)
print("Accuracy midfield third right: ", accuracy_2R)
print("Accuracy attacking third left: ", accuracy_3L)
print("Accuracy attacking third centre: ", accuracy_3C)
print("Accuracy attacking third rigth: ", accuracy_3R)


# In[ ]:




