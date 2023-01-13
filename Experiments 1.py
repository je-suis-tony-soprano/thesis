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


sns.heatmap(df.corr())


# In[4]:


# splitting, balancing, standardising
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)


# In[6]:


x = df.iloc[:, [1,2,3,4,5,6,7,8,9]].values
y = df.iloc[:, 10].values


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 30)


# In[12]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(X_train) 
x_test = sc_x.transform(X_test)


# In[13]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 10)
X_train_bal, y_train_bal = sm.fit_resample(x_train, y_train)


# In[14]:


print(X_train_bal.shape)
print(y_train_bal.shape)


# # logistic regression

# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# In[19]:


lggrid = {"C":[0.1,1,10,100], 
          "penalty":["l1", "l2"], 
          "solver":["lbfgs", "liblinear"]}
lg = LogisticRegression(random_state = 10)
lg_cv = GridSearchCV(lg, lggrid, cv=10)
lg_cv.fit(X_train_bal, y_train_bal)

print("tuned hpyerparameters :(best parameters) ",lg_cv.best_params_)
print("accuracy :",lg_cv.best_score_)


# In[23]:


y_pred_lg = lg_cv.predict(x_test)


# In[24]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_lg)

ax = plt.axes()
sns.heatmap(cm, ax = ax, annot=True, cmap='Blues', fmt='g')
ax.set_title('Logistic Regression')
plt.show()


# In[26]:


from sklearn.metrics import accuracy_score
print("Accuracy : ", accuracy_score(y_test, y_pred_lg))
print("F1 Score : ", f1_score(y_test, y_pred_lg, average='macro'))

y_pred_proba = lg_cv.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# # XGBoost

# In[29]:


from xgboost import XGBClassifier
xg = XGBClassifier(objective = 'binary:logistic')


# In[35]:


xggrid = {'learning_rate':[0.1, 0.05, 0.01],
          'colsample_bytree': [ 0.3, 0.5 , 0.8 ],
          'max_depth': range(4,8,1)
         }

xg_cv = GridSearchCV(estimator=xg,
    param_grid=xggrid,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 10,
    verbose=True
)


# In[36]:


xg_cv.fit(X_train_bal, y_train_bal)


# In[37]:


xg_cv.best_estimator_


# In[38]:


y_pred_xg = xg_cv.predict(x_test)


# In[39]:


cm = confusion_matrix(y_test, y_pred_xg)

ax = plt.axes()
sns.heatmap(cm, ax = ax, annot=True, cmap='Blues', fmt='g')

ax.set_title('XGBoost')
plt.show()


# In[41]:


accuracy_xg = accuracy_score(y_test, y_pred_xg)
f1_xg = f1_score(y_test, y_pred_xg)
print("Accuracy: ", accuracy_xg)
print("F1-score :", f1_xg)

y_pred_proba = xg_cv.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# # support vector machines

# In[60]:


from sklearn import svm
from sklearn.svm import SVC


# In[61]:


svmgrid = {'C': [0.1, 1, 10], 
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf']}

svm_cv = GridSearchCV(SVC(), svmgrid, refit = True, verbose = 3)


# In[62]:


svm_cv.fit(X_train_bal, y_train_bal)


# In[67]:


print(svm_cv.best_params_)
print(svm_cv.best_estimator_)


# In[66]:


y_pred_svm = svm_cv.predict(x_test)


# In[68]:


cm = confusion_matrix(y_test, y_pred_svm)

ax = plt.axes()
sns.heatmap(cm, ax = ax, annot=True, cmap='Blues', fmt='g')

ax.set_title('SVM')
plt.show()


# In[69]:


accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
print("Accuracy: ", accuracy_svm)
print("F1-score :", f1_svm)


# In[70]:


from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve

plot_roc_curve(svm_cv,x_test,y_test)

