
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import matplotlib.pyplot as plt

filename1 = sys.argv[1]
filename2 = sys.argv[2]
#filename1 = "pagecounts-20160802-150000.txt"
#filename2 = "pagecounts-20160803-150000.txt"


# In[2]:

dataframe1 = pd.read_table(filename1, sep=' ', header=None, index_col=1,
        names=['lang', 'page', 'views', 'bytes'])
dataframe2 = pd.read_table(filename2, sep=' ', header=None, index_col=1,
        names=['lang', 'page', 'views', 'bytes'])


# In[3]:

dataframe1 = dataframe1.sort_values(["views"],ascending=False)


# In[4]:

combo = pd.concat([dataframe1, dataframe2], axis=1, join_axes=[dataframe1.index])
new_columns = combo.columns.values
new_columns[4]="views2"
combo.column = new_columns
#combo


# In[5]:

plt.figure(figsize=(10, 5)) # change the size to something sensible
plt.subplot(1, 2, 1) # subplots in 1 row, 2 columns, select the first
plt.title('Popularity Distribution')
plt.xlabel("Rank")
plt.ylabel("Views")
plt.plot(dataframe1['views'].values)

plt.subplot(1, 2, 2) # ... and then select the second
plt.title('Daily Correlation')
plt.xlabel("Day 2 views")
plt.ylabel("Day 1 views")
plt.plot(combo['views'].values,combo['views2'].values,'b.')
plt.xscale('log')
plt.yscale('log')

#plt.show()
plt.savefig('wikipedia.png')

