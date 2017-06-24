
# coding: utf-8

# In[154]:

import sys
import pandas as pd
import numpy as np
import difflib
import gzip
from scipy import stats
from statsmodels.stats import multicomp


# In[173]:

df = pd.read_table("data.csv",sep=',')
stats.normaltest(np.sqrt(df['qs1']))
df_sum = pd.DataFrame(columns =["name","normalTest_p"])
df_sum['name']=df.columns
df_sum["normalTest_p"] = df.apply(lambda x: stats.normaltest(x.values).pvalue).values

p_anova = stats.f_oneway(df["qs1"],df["qs2"],df["qs3"],df["qs4"],df["qs5"],df["merge1"],df["partition_sort"])
print("p-value for ANOVA test: "+str(p_anova.pvalue))
x_melt = pd.melt(df)
posthoc = multicomp.pairwise_tukeyhsd(x_melt['value'], x_melt['variable'], alpha=0.05)
print(posthoc)
print("""If we do a normal test on the data set""")
print(df_sum)
print("----------------------------------------------------------------")
print("""The data showes the distribution is sometimes not normal......Hmm...""")
print("""It could means that not conclusion can be drawn from ANOVA test since ANOVA test requires data in nomal distribution.""")
print("""But we will assume they are normal, by the nature of the test.""")
print("""From the scipy ANOVA website https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.f_oneway.html""")
print("""It hints that we could use Kruskal-Wallis H-test (which does not require data in normal distribution). Let's try it !""")
print("")



# In[176]:

p_kwh = stats.kruskal(df["qs1"],df["qs2"],df["qs3"],df["qs4"],df["qs5"],df["merge1"],df["partition_sort"])
print("p-value for Kruskal-Wallis H-test: "+str(p_kwh.pvalue))
print("since p-value is < 0.05, we could conclude that they have different means.")


