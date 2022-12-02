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


df = pd.read_csv("C:\\Users\\35383\\Documents\\Master\\Thesis\\forexperiments.csv")
df


# In[9]:


sns.heatmap(df.corr())


# In[4]:


df['shot'].value_counts()


# In[5]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)


# In[7]:


x = df.iloc[:, [1,2,4,5,6,13]].values
y = df.iloc[:, 14].values


# In[8]:


from sklearn import preprocessing
from sklearn import utils

lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)
y_transformed


# # Splitting, Balancing and Stanardizing

# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 30)


# In[11]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(X_train) 
x_test = sc_x.transform(X_test)
  
print (x_train[0:10, :])


# In[12]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 10)
X_bal, y_bal = sm.fit_resample(X_train, y_train)


# In[13]:


print(np.bincount(y_bal))
print(len(X_bal))


# In[13]:


print(X_train.shape)
print(y_train.shape)
print(X_bal.shape)
print(y_bal.shape)


# # logistic regression

# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

grid = {"C":[0.1,1,10]}
lg = LogisticRegression(random_state = 10)
lg_cv = GridSearchCV(lg, grid, cv=10)
lg_cv.fit(X_bal, y_bal)

print("tuned hpyerparameters :(best parameters) ",lg_cv.best_params_)
print("accuracy :",lg_cv.best_score_)


# In[17]:


y_predlg = lg_cv.predict(X_test)


# In[18]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predlg)
  
print ("Confusion Matrix : \n", cm)


# In[19]:


ax = plt.axes()
sns.heatmap(cm, ax = ax, annot=True, cmap='Blues', fmt='g')

ax.set_title('Logistic Regression')
plt.show()


# In[20]:


from sklearn.metrics import accuracy_score
print("Accuracy : ", accuracy_score(y_test, y_predlg))
print("F1 Score : ", f1_score(y_test, y_predlg, average='macro'))


# In[22]:


y_pred_proba = lg_cv.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# plot curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# In[23]:


print("AUC: ", auc)


# # svm

# In[46]:


X_train_small = X_bal[1:1000]
X_train_small.shape


# In[47]:


y_train_small = y_bal[1:1000]
y_train_small.shape


# In[48]:


from sklearn.svm import SVC  
svm = SVC()
parameters = {'C': [0.1, 1, 10], 
              'kernel': ['rbf', 'linear']}


# In[49]:


svm_cv = GridSearchCV(SVC(), parameters, refit = True, verbose = 3)


# In[50]:


svm_cv.fit(X_train_small, y_train_small)


# In[51]:


print(svm_cv.best_params_)


# In[52]:


y_predsvm = svm_cv.predict(X_test)


# In[53]:


cm_svm = confusion_matrix(y_test, y_predsvm)
  
print ("Confusion Matrix : \n", cm_svm)


# In[54]:


accuracy_svm = accuracy_score(y_test, y_predsvm)
f1_svm = f1_score(y_test, y_predsvm)

print("Accuracy: ", accuracy_svm)
print("F1_score: ", f1_svm)


# In[57]:


test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_predsvm)

plt.grid()


plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
plt.plot([0,1],[0,1],'g--')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC(ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()


# In[ ]:





# # xgboost

# In[48]:


get_ipython().system('pip3 install xgboost')


# In[14]:


from xgboost import XGBClassifier


# In[15]:


xg = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')


# In[16]:


parameters = {
    'learning_rate': [0.1, 0.01, 0.05]
}


# In[19]:


from sklearn.model_selection import GridSearchCV

xg_cv = GridSearchCV(estimator=xg,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 10,
    verbose=True
)


# In[20]:


xg_cv.fit(X_bal, y_bal)


# In[21]:


xg_cv.best_estimator_


# In[22]:


y_predxg = xg_cv.predict(X_test)


# In[23]:


cm = confusion_matrix(y_test, y_predxg)
  
print ("Confusion Matrix : \n", cm)


# In[24]:


ax = plt.axes()
sns.heatmap(cm, ax = ax, annot=True, cmap='Blues', fmt='g')

ax.set_title('XGBoost')
plt.show()


# In[25]:


accuracy_xg = accuracy_score(y_test, y_predxg)
f1_xg = f1_score(y_test, y_predxg)
print("Accuracy: ", accuracy_xg)
print("F1-score :", f1_xg)


# In[26]:


y_pred_proba = xg_cv.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# plot curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# In[45]:


print(xg_cv.best_estimator_.feature_importances_)


# In[30]:


importances = xg_cv.best_estimator_.feature_importances_


# In[34]:


from xgboost import plot_importance

model = XGBClassifier(learning_rate = 0.1)
model.fit(X_bal, y_bal)
plot_importance(model)
pyplot.show()


# In[37]:




