
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import difflib


# In[2]:


filename1 = sys.argv[1]
filename2 = sys.argv[2]
filename3 = sys.argv[3]


# In[3]:

with open(filename1) as f:
    df_movie_list = pd.read_table(f, header=None, names=['title'], lineterminator='\n')
    df_movie_list = df_movie_list.sort_values("title")
with open(filename2) as f:
    df_rating_list = pd.read_table(f, sep=',', lineterminator='\n')


# In[13]:

def match(x):
    matches = difflib.get_close_matches(x,df_movie_list["title"])
    if len(matches)==0:
        return float('NaN')
    else:
        return matches[0]
    
df_rating_list ["match"] = df_rating_list["title"].apply(match)
df_rating_list ["rating"] = df_rating_list["rating"].apply(float)
df_rating_list = df_rating_list[df_rating_list["match"].notnull()]
df_rating_list = df_rating_list.sort_values("match")
joined_df = df_rating_list.groupby(["match"]).sum().reset_index()
joined_df["rating"] = df_rating_list.groupby(['match']).sum().reset_index()["rating"] /df_rating_list.groupby(['match']).count().reset_index()["rating"]
joined_df["rating"]=joined_df["rating"].round(2)
joined_df.columns = ["title", "rating"]
final_df = pd.merge(joined_df, df_movie_list, on=['title', 'title'])
final_df.to_csv(filename3,index=False)
final_df

