
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']


# In[5]:

totals_annual = np.sum(totals, axis=1)
min_total_annual = np.argmin(totals_annual)
print("Row with lowest total preciptiation:")
print(min_total_annual)


# In[6]:

prep_average = np.sum(totals, axis=0)/np.sum(counts, axis=0)
print("Average precipitation in each month:")
print(prep_average)


# In[7]:

place_daily_average = np.sum(totals, axis=1)/np.sum(counts, axis=1)
print("Average precipitation in each city:")
print(place_daily_average)


# In[8]:

totals_quarterly = np.reshape(totals, (totals.shape[0],4,3))
totals_quarterly_sum = np.sum(totals_quarterly, axis=2)
print("Quarterly precipitation totals:")
print(totals_quarterly_sum)

