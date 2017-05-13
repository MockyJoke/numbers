
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

totals = pd.read_csv('totals.csv').set_index(keys=['name'])
counts = pd.read_csv('counts.csv').set_index(keys=['name'])
#totals


# In[3]:

#counts


# In[4]:

print("City with lowest total preciptiation:")
print(totals.sum(axis=1).idxmin())


# In[5]:

print("Average precipitation in each month:")
print(totals.sum()/counts.sum())


# In[6]:

print("Average precipitation in each city:")
print(totals.sum(axis=1)/counts.sum(axis=1))

