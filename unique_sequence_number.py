#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[5]:


df = pd.read_csv("C:\\Users\\35383\\Documents\\Master\\Thesis\\data\\allevents.csv")
df


# In[6]:


next_event = df.shift(-1, fill_value=0)
df["next.event.possession"] = next_event["possession"]
df


# In[7]:


df['sequence'] = np.nan


# In[10]:


i = 1
sequence = []
for index, row in df.iterrows():
    if row['possession'] == row['next.event.possession']:
        sequence.append(i)
    if row['possession'] != row['next.event.possession']:
        sequence.append(i)
        i += 1
        
print(sequence)


# In[11]:


df['sequence'] = sequence


# In[ ]:


df.to_csv(r'C:\Users\35383\Documents\Master\Thesis\data\sequences.csv', index = False, header=True)

