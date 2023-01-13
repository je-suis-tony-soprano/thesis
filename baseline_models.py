#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
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


# load data
df = pd.read_csv("C:\\Users\\35383\\Documents\\Master\\Thesis\\data\\features.csv")
df


# In[3]:


# splitting, balancing, standardising
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)


# In[4]:


x = df.iloc[:, 6].values
y = df.iloc[:, 10].values


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 30)


# In[9]:


X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)


# In[10]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(X_train) 
x_test = sc_x.transform(X_test)


# In[11]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 10)
X_train_bal, y_train_bal = sm.fit_resample(x_train, y_train)


# In[12]:


print(X_train_bal.shape)
print(y_train_bal.shape)


# # logistic regression

# In[14]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(random_state = 10)


# In[16]:


lg.fit(X_train_bal, y_train_bal)


# In[20]:


y_pred_lg = lg.predict(x_test)


# In[22]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

print("Accuracy : ", accuracy_score(y_test, y_pred_lg))
print("F1 Score : ", f1_score(y_test, y_pred_lg, average='macro'))

y_pred_proba = lg.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# # xgboost

# In[23]:


from xgboost import XGBClassifier
xg = XGBClassifier(objective = 'binary:logistic')


# In[24]:


xg.fit(X_train_bal, y_train_bal)


# In[25]:


y_pred_xgb = xg.predict(x_test)


# In[26]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

print("Accuracy : ", accuracy_score(y_test, y_pred_xgb))
print("F1 Score : ", f1_score(y_test, y_pred_xgb, average='macro'))

y_pred_proba = xg.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# # svm

# In[27]:


from sklearn import svm
from sklearn.svm import SVC

svm = SVC()


# In[28]:


svm.fit(X_train_bal, y_train_bal)


# In[29]:


y_pred_svm = svm.predict(x_test)


# In[30]:


accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
print("Accuracy: ", accuracy_svm)
print("F1-score :", f1_svm)


# In[31]:


from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve

plot_roc_curve(svm ,x_test, y_test)

