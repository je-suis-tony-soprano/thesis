#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


df = pd.read_csv("C:\\Users\\35383\\Documents\\Master\\Thesis\\sequences.csv")
df


# In[4]:


x = df.iloc[:, 2].values
y = df.iloc[:, 14].values


# # splitting and balancing

# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 30)
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)


# In[6]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 10)
X_bal, y_bal = sm.fit_resample(X_train, y_train)


# # logistic regression baseline

# In[7]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()


# In[8]:


lg.fit(X_bal, y_bal)


# In[9]:


y_pred = lg.predict(X_test)


# In[19]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

ax = plt.axes()
sns.heatmap(cm, ax = ax, annot=True, cmap='Blues', fmt='g')

ax.set_title('Logistic Regression Baseline')
plt.show()


# In[11]:


from sklearn.metrics import accuracy_score
print("Accuracy : ", accuracy_score(y_test, y_pred))
print("F1 Score : ", f1_score(y_test, y_pred, average='macro'))


# In[12]:


y_pred_proba = lg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# plot curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# # xgboost baseline

# In[13]:


from xgboost import XGBClassifier
xg = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')


# In[14]:


xg.fit(X_bal, y_bal)


# In[15]:


y_predxg = xg.predict(X_test)


# In[18]:


cm = confusion_matrix(y_test, y_predxg)
ax = plt.axes()
sns.heatmap(cm, ax = ax, annot=True, cmap='Blues', fmt='g')

ax.set_title('XGBoost Baseline')
plt.show()


# In[17]:


accuracy_xg = accuracy_score(y_test, y_predxg)
f1_xg = f1_score(y_test, y_predxg)
print("Accuracy: ", accuracy_xg)
print("F1-score :", f1_xg)


# In[20]:


y_pred_proba = xg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# plot curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

