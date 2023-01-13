#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("C:\\Users\\35383\\Documents\\Master\\Thesis\\data\\events_nlp.csv")
df


# In[3]:


next_event_sequence = df.shift(-1, fill_value=0)
df["next.event.sequence"] = next_event_sequence["sequence"]


# In[4]:


df


# # prepare for tokenising

# In[5]:


# convert pass height column to single word
df['pass.height.name'].unique()


# In[6]:


df['pass.height.name'] = df['pass.height.name'].replace({'Ground Pass': 'Ground', 'High Pass': 'High', 'Low Pass': 'Low'})


# In[7]:


# create columns for locations
df['token_location_x'] = np.nan
df['token_location_y'] = np.nan


# In[8]:


token_locations_x = []

for index, row in df.iterrows():
    if row['location.x'] <= 40:
        token_locations_x.append('Defence')
    if row['location.x'] > 40 and row['location.x'] <= 80:
        token_locations_x.append('Midfield')
    if row['location.x'] > 80 and row['location.x'] <= 120:
        token_locations_x.append('Attack')


# In[9]:


token_locations_y = []

for index, row in df.iterrows():
    if row['location.y'] <= 25:
        token_locations_y.append('Left')
    if row['location.y'] > 25 and row['location.y'] <= 55:
        token_locations_y.append('Centre')
    if row['location.y'] > 55 and row['location.y'] <= 80:
        token_locations_y.append('Right')


# In[10]:


df['token_location_x'] = token_locations_x
df['token_location_y'] = token_locations_y


# In[11]:


# create columns for directions
df['direction'] = np.nan


# In[12]:


directions = []

for index, row in df.iterrows():
    if row['type.name'] == "Pass":
        if row['pass.end_location.x'] > row['location.x']:
            directions.append('Forward')
        if row['pass.end_location.x'] == row['location.x']:
            directions.append('Sideways')
        if row['pass.end_location.x'] < row['location.x']:
            directions.append('Backwards')
    
    if row['type.name'] == "Carry":
        if row['carry.end_location.x'] > row['location.x']:
            directions.append('Forward')
        if row['carry.end_location.x'] == row['location.x']:
            directions.append('Sideways')
        if row['carry.end_location.x'] < row['location.x']:
            directions.append('Backwards')
            
    if row['type.name'] == "Dribble":
        directions.append("NaN_dribble")
            
    if row['type.name'] == "Shot":
        directions.append('NaN_shot')


# In[13]:


df['direction'] = directions


# In[14]:


df


# # tokenisation time

# In[15]:


tokens = []

for index, row in df.iterrows():
    if row['type.name'] == "Pass":
        if row['sequence'] == row['next.event.sequence']:
            token = str(row['type.name']) + str(row['token_location_x']) + str(row['token_location_y']) + str(row['direction']) + str(row['pass.height.name']) + " "
            tokens.append(token)
        else:
            token = str(row['type.name']) + str(row['token_location_x']) + str(row['token_location_y']) + str(row['direction']) + str(row['pass.height.name'])+ ". "
            tokens.append(token)
            
    if row['type.name'] == "Carry":
        if row['sequence'] == row['next.event.sequence']:
            token = str(row['type.name']) + str(row['token_location_x']) + str(row['token_location_y']) + str(row['direction'])+ " "
            tokens.append(token)
        else:
            token = str(row['type.name']) + str(row['token_location_x']) + str(row['token_location_y']) + str(row['direction'])+ ". "
            tokens.append(token)
            
    if row['type.name'] == "Dribble":
        if row['sequence'] == row['next.event.sequence']:
            token = str(row['type.name']) + str(row['token_location_x']) + str(row['token_location_y']) + " "
            tokens.append(token)
        else:
            token = str(row['type.name']) + str(row['token_location_x']) + str(row['token_location_y'])  + ". "
            tokens.append(token)
            
    if row['type.name'] == "Shot":
        if row['sequence'] == row['next.event.sequence']:
            token = str(row['type.name']) + str(row['token_location_x']) + str(row['token_location_y']) + " "
            tokens.append(token)
        else:
            token = str(row['type.name']) + str(row['token_location_x']) + str(row['token_location_y']) + ". "
            tokens.append(token)


# In[17]:


df['token'] = tokens


# In[18]:


df.to_csv(r'C:\Users\35383\Documents\Master\Thesis\data\tokenised_events.csv', index = False, header=True)


# In[ ]:




