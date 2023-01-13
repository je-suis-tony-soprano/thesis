#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("C:\\Users\\35383\\Documents\\Master\\Thesis\\data\\features.csv")
df


# In[4]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)


# In[5]:


x = df.iloc[:, [1,2,3,4,5,6,7,8,9]].values
y = df.iloc[:, 10].values


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 30)


# In[7]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(X_train) 
x_test = sc_x.transform(X_test)


# In[8]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 10)
X_train_bal, y_train_bal = sm.fit_resample(x_train, y_train)


# In[11]:


# create best performing XGBoost model
from xgboost import XGBClassifier
xg = XGBClassifier(objective = 'binary:logistic',
                  learning_rate = 0.1,
                  colsample_bytree = 0.8,
                  max_depth = 7)


# In[12]:


xg.fit(X_train_bal, y_train_bal)


# In[14]:


# plot feature importance
xg.feature_importances_


# In[22]:


import shap

explainer = shap.TreeExplainer(xg)
shap_values = explainer.shap_values(x_test)


# In[29]:


shap.summary_plot(shap_values, x_test, plot_type="bar", feature_names=["number_of_passes", "number_of_dribbles",
                                                                      "number_of_carries", "attack_speed", "start_distance_x",
                                                                      "end_distance_x", "end_distance_x_2", "start_offcentre",
                                                                      "end_offcentre"])

