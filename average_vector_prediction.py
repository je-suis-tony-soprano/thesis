#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
import gensim
from gensim import similarities, corpora, models
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings


# In[2]:


# remove warning messages
warnings.filterwarnings(action = 'ignore')


# # load data and convert to sentences

# In[3]:


df = pd.read_csv("C:\\Users\\35383\\Documents\\Master\\Thesis\\data\\tokenised_events.csv")


# In[4]:


corpus = (df['token']).tolist()


# In[5]:


# writing the tokens into a text file
with open(r'C:/Users/35383/Documents/Master/Thesis/data/tokens.txt', 'w') as fp:
    for token in corpus:
        fp.write("%s" % token)


# In[6]:


# opening the file and removing new lines
file = open("C:\\Users\\35383\\Documents\\Master\\Thesis\\data\\tokens.txt", "r")
f = file.read()
f = f.replace("\n", " ")


# In[7]:


# iterate through each sentence in the file
data = []
for i in sent_tokenize(f):
    temp = []
     
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
 
    data.append(temp)


# # implement Word2Vec

# In[8]:


model = gensim.models.Word2Vec(data, min_count = 1,
                              vector_size = 32, window = 5, sg = 0)


# In[9]:


words = list(w for w in model.wv.index_to_key)
len(words)


# In[10]:


my_dict = ({})

for word in words:
    my_dict[word] = model.wv[word]


# In[11]:


my_dict


# # evaluate embeddings

# In[12]:


# most_similar for four event types
model.wv.most_similar('passdefencecentreforwardground')


# In[13]:


model.wv.most_similar('carrymidfieldleftbackwards')


# In[14]:


model.wv.most_similar('dribbleattackcentre')


# In[23]:


# k means clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

x = model.wv.vectors

x = normalize(x)

num_clusters = 5

km = KMeans(n_clusters=num_clusters, random_state = 30)
km.fit(x)


# In[25]:


cluster_assignments = km.labels_
cluster_assignments


# In[27]:


for word, cluster in zip(words, cluster_assignments):
  print(f"{word}: {cluster}")


# In[28]:


cluster_0 = []
cluster_1 = []
cluster_2 = []
cluster_3 = []
cluster_4 = []

for word, cluster in zip(words, cluster_assignments):
  if cluster == 0:
    cluster_0.append(word)
  if cluster == 1:
    cluster_1.append(word)
  if cluster == 2:
    cluster_2.append(word)
  if cluster == 3:
    cluster_3.append(word)
  else:
    cluster_4.append(word)


# In[29]:


cluster_0


# In[30]:


cluster_1


# In[31]:


cluster_2


# In[32]:


cluster_3


# In[33]:


cluster_4


# # obtain average vector for every sequence

# In[15]:


df['token'] = df['token']. apply(str. lower)


# In[16]:


data


# In[17]:


def average_vector(sentence):
    vectors = []
    
    for word in sentence:
        if "shot" not in word:
            vectors.append(model.wv[word])
        
    sentence_vector = np.mean(vectors, axis = 0)
    
    return sentence_vector


# In[18]:


vectors = []

for sequence in data:
    sequence_vector = average_vector(sequence)
    vectors.append(sequence_vector)


# In[19]:


vectors


# In[20]:


len(vectors)


# # put average vectors into sequences dataframe

# In[21]:


df2 = pd.read_csv("C:\\Users\\35383\\Documents\\Master\\Thesis\\data\\features_nlp.csv")
df2


# In[22]:


df2['average_vector'] = vectors


# In[24]:


def split_array(arr):
    # Create a Pandas Series with 32 elements
    s = pd.Series(arr, index=range(1, 33))
    return s


# In[25]:


new_df = df2['average_vector'].apply(split_array)


# In[26]:


new_df['shot'] = df2['shot']


# In[27]:


new_df


# # use average vectors as inputs for classifier

# In[28]:


x = new_df.iloc[:, 0:32].values
y = new_df.iloc[:, 32].values


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 30)


# # xgboost

# In[38]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

xg = XGBClassifier(objective = 'binary:logistic')


# In[39]:


xggrid = {'learning_rate':[0.1, 0.05, 0.01],
          'colsample_bytree': [ 0.3, 0.5 , 0.8 ],
          'max_depth': range(4,8,1)
         }


# In[40]:


xg_cv = GridSearchCV(estimator=xg,
    param_grid = xggrid,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 10,
    verbose=True
)


# In[41]:


xg_cv.fit(X_train, y_train)


# In[42]:


xg_cv.best_estimator_


# In[43]:


y_pred_xg = xg_cv.predict(X_test)


# In[44]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred_xg)

ax = plt.axes()
sns.heatmap(cm, ax = ax, annot=True, cmap='Blues', fmt='g')

ax.set_title('XGBoost using average vectors')
plt.show()


# In[46]:


from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

accuracy_xg = accuracy_score(y_test, y_pred_xg)
f1_xg = f1_score(y_test, y_pred_xg)
print("Accuracy: ", accuracy_xg)
print("F1-score :", f1_xg)

y_pred_proba = xg_cv.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# # error analysis

# In[97]:


new_df.to_csv(r'C:\Users\35383\Documents\Master\Thesis\data\erroranalysis.csv', index = False, header=True)

