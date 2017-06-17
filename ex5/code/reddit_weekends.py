
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import numpy as np
import difflib
import gzip
# import matplotlib.pyplot as plt
from scipy import stats 

# filename1 = sys.argv[1]

#filename1 = "reddit-counts.json.gz"


# In[16]:

def filter_data(df):
# data_file = gzip.open(filename1, 'rt', encoding='utf-8')
# data = pd.read_json(data_file, lines=True)
    data=df.copy()
    data['year']=data.apply(lambda x: x['date'].year,axis = 1)
    data['iso_year']=data.apply(lambda x: x['date'].isocalendar()[0],axis = 1)
    data['week_sq']=data.apply(lambda x: x['date'].isocalendar()[1],axis = 1)
    data['weekday']=data.apply(lambda x: x['date'].isocalendar()[2],axis = 1)
    data = data[((data['year']==2012)|(data['year']==2013))&(data['subreddit']=="canada")]
    data['isWeekend'] = data.apply(lambda x: False if x['weekday'] < 6 else True, axis = 1)
    return data


# In[3]:

# weekends = data[data["isWeekend"]==True]
# weekdays = data[data["isWeekend"]==False]


# <h2>Normality and Equal variance Test</h2>

# In[4]:

# print(stats.normaltest(weekends["comment_count"]).pvalue)
# print(stats.normaltest(weekdays["comment_count"]).pvalue)
# print(stats.levene(weekdays["comment_count"],weekends["comment_count"]).pvalue)


# In[5]:

def fix_1(func):
    print(func)
    print(stats.normaltest(func(weekends["comment_count"])).pvalue)
    print(stats.normaltest(func(weekdays["comment_count"])).pvalue)
    print(stats.levene(func(weekdays["comment_count"]),func(weekends["comment_count"])).pvalue)
# fix_1(np.log)
# fix_1(np.log)
# not working for exp
# fix_1(np.exp)
# fix_1(np.sqrt)
# fix_1(lambda x:x*x)




def main():
    OUTPUT_TEMPLATE = (
    "Initial (invalid) T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann–Whitney U-test p-value: {utest_p:.3g}"
)
    
    reddit_counts = sys.argv[1]

    # ...
    data_file = gzip.open(reddit_counts, 'rt', encoding='utf-8')
    df = pd.read_json(data_file, lines=True)
    data = filter_data(df)
    weekends = data[data["isWeekend"]==True]
    weekdays = data[data["isWeekend"]==False]
    
    weekends_by_week = weekends.groupby(["iso_year","week_sq"]).mean().reset_index()
    weekdays_by_week = weekdays.groupby(["iso_year","week_sq"]).mean().reset_index()
    
    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=stats.ttest_ind(weekdays["comment_count"],weekends["comment_count"]).pvalue,
        initial_weekday_normality_p=stats.normaltest(weekdays["comment_count"]).pvalue,
        initial_weekend_normality_p=stats.normaltest(weekends["comment_count"]).pvalue,
        initial_levene_p=stats.levene(weekdays["comment_count"],weekends["comment_count"]).pvalue,
        transformed_weekday_normality_p=stats.normaltest(np.sqrt(weekdays["comment_count"])).pvalue,
        transformed_weekend_normality_p=stats.normaltest(np.sqrt(weekends["comment_count"])).pvalue,
        transformed_levene_p=stats.levene(np.sqrt(weekdays["comment_count"]),np.sqrt(weekends["comment_count"])).pvalue,
        weekly_weekday_normality_p=stats.normaltest(weekdays_by_week["comment_count"]).pvalue,
        weekly_weekend_normality_p=stats.normaltest(weekends_by_week["comment_count"]).pvalue,
        weekly_levene_p=stats.levene(weekdays_by_week["comment_count"],weekends_by_week["comment_count"]).pvalue,
        weekly_ttest_p=stats.ttest_ind(weekdays_by_week["comment_count"],weekends_by_week["comment_count"]).pvalue,
        utest_p=stats.mannwhitneyu(weekdays["comment_count"],weekends["comment_count"]).pvalue,
    ))

if __name__ == '__main__':
    main()

