#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib as plt
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# In[2]:


df = pd.read_csv("C:\\Users\\35383\\Documents\\Master\\Thesis\\data\\features.csv")
df


# In[ ]:





# In[4]:


df1 = df[df['start_distance_x'] > 80]
df1


# In[13]:


df2 = df[(df['start_distance_x'] > 40) & (df['start_distance_x'] <= 80)]
df2


# In[14]:


df3 = df[(df['start_distance_x'] > 0) & (df['start_distance_x'] <= 40)]
df3


# In[12]:


df1L = df1[(df1['start_location_y'] >=0) & (df1['start_location_y'] <= 25)]
df1C = df1[(df1['start_location_y'] > 25) & (df1['start_location_y'] <= 55)]
df1R = df1[(df1['start_location_y'] > 55) & (df1['start_location_y'] <= 80)]


# In[15]:


df2L = df2[(df2['start_location_y'] >=0) & (df2['start_location_y'] <= 25)]
df2C = df2[(df2['start_location_y'] > 25) & (df2['start_location_y'] <= 55)]
df2R = df2[(df2['start_location_y'] > 55) & (df2['start_location_y'] <= 80)]


# In[16]:


df3L = df3[(df3['start_location_y'] >=0) & (df3['start_location_y'] <= 25)]
df3C = df3[(df3['start_location_y'] > 25) & (df3['start_location_y'] <= 55)]
df3R = df3[(df3['start_location_y'] > 55) & (df3['start_location_y'] <= 80)]


# In[42]:


df1Lx = df1L.iloc[:, [1,2,4,5,6,13]].values
df1Cx = df1C.iloc[:, [1,2,4,5,6,13]].values
df1Rx = df1R.iloc[:, [1,2,4,5,6,13]].values
df2Lx = df2L.iloc[:, [1,2,4,5,6,13]].values
df2Cx = df2C.iloc[:, [1,2,4,5,6,13]].values
df2Rx = df2R.iloc[:, [1,2,4,5,6,13]].values
df3Lx = df3L.iloc[:, [1,2,4,5,6,13]].values
df3Cx = df3C.iloc[:, [1,2,4,5,6,13]].values
df3Rx = df3R.iloc[:, [1,2,4,5,6,13]].values


# In[27]:


df1Ly = df1L.iloc[:, 14].values
df1Cy = df1C.iloc[:, 14].values
df1Ry = df1R.iloc[:, 14].values
df2Ly = df2L.iloc[:, 14].values
df2Cy = df2C.iloc[:, 14].values
df2Ry = df2R.iloc[:, 14].values
df3Ly = df3L.iloc[:, 14].values
df3Cy = df3C.iloc[:, 14].values
df3Ry = df3R.iloc[:, 14].values


# In[32]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)


# In[33]:


x = df.iloc[:, [1,2,4,5,6,13]].values
y = df.iloc[:, 14].values


# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 30)


# In[35]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(X_train) 
x_test = sc_x.transform(X_test)
  
print (x_train[0:10, :])


# In[36]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 10)
X_bal, y_bal = sm.fit_resample(X_train, y_train)


# In[38]:


from xgboost import XGBClassifier
model = XGBClassifier(learning_rate = 0.1)
model.fit(X_bal, y_bal)


# In[45]:


pred1L = model.predict(df1Lx)
pred1C = model.predict(df1Cx)
pred1R = model.predict(df1Rx)
pred2L = model.predict(df2Lx)
pred2C = model.predict(df2Cx)
pred2R = model.predict(df2Rx)
pred3L = model.predict(df3Lx)
pred3C = model.predict(df3Cx)
pred3R = model.predict(df3Rx)


# In[49]:


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

