#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install gensim')


# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("C:\\Users\\35383\\Documents\\Master\\Thesis\\RQ2nlp.csv")


# In[3]:


df


# In[4]:


# tokenisation
tokens = []
for index, row in df.iterrows():
    if row['type.name'] == "Pass":
        if row['possession'] == row['next_possession']:
            token = str(row['type.name']) + str(row['new.location.x.word']) + str(row['new.location.y.word']) + " "
            tokens.append(token)
        else:
            token = str(row['type.name']) + str(row['new.location.x.word']) + str(row['new.location.y.word']) + ". "
            tokens.append(token)
    if row['type.name'] == "Carry":
        if row['possession'] == row['next_possession']:
            token = str(row['type.name']) + str(row['new.location.x.word']) + str(row['new.location.y.word']) + " "
            tokens.append(token)
        else:
            token = str(row['type.name']) + str(row['new.location.x.word']) + str(row['new.location.y.word']) + ". "
            tokens.append(token)
    if row['type.name'] == "Dribble":
        if row['possession'] == row['next_possession']:
            token = str(row['type.name']) + str(row['new.location.x.word']) + str(row['new.location.y.word']) + str(row['dribble.outcome.name']) + " "
            tokens.append(token)
        else:
            token = str(row['type.name']) + str(row['new.location.x.word']) + str(row['new.location.y.word']) + str(row['dribble.outcome.name']) + ". "
            tokens.append(token)
    if row['type.name'] == "Shot":
        if row['possession'] == row['next_possession']:
            token = str(row['type.name']) + str(row['new.location.x.word']) + str(row['new.location.y.word']) + " "
            tokens.append(token)
        else:
            token = str(row['type.name']) + str(row['new.location.x.word']) + str(row['new.location.y.word']) + ". "
            tokens.append(token)


# In[5]:


df['token'] = tokens


# In[6]:


df


# In[7]:


corpus = (df['token']).tolist()
corpus


# In[8]:


len(corpus)


# In[20]:


import nltk
import gensim
from gensim import similarities, corpora, models
nltk.download('punkt')


# In[14]:


with open(r'C:/Users/35383/Documents/Master/Thesis/sequences.txt', 'w') as fp:
    for token in corpus:
        fp.write("%s" % token)


# In[18]:


from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
 
warnings.filterwarnings(action = 'ignore')


# In[12]:


file = open("C:\\Users\\35383\\Documents\\Master\\Thesis\\sequences.txt", "r")
f = file.read()


# In[13]:


f = f.replace("\n", " ")


# In[14]:


# iterate through each sentence in the file
data = []
for i in sent_tokenize(f):
    temp = []
     
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
 
    data.append(temp)


# In[15]:


data


# In[16]:


len(data)


# # implement word2vec

# In[21]:


model1 = gensim.models.Word2Vec(data, min_count = 1,
                              vector_size = 32, window = 5)


# In[23]:


print("Cosine similarity between 'passmidfieldcentre' " +
               "passdefenceleft' - CBOW : ",
    model1.wv.similarity('dribbleattackcentrecomplete', 'shotattackcentre'))


# In[24]:


model1.wv.most_similar('passmidfieldright')


# In[25]:


kv = model1.wv
kv.vectors

