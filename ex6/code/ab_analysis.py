
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import numpy as np
import difflib
import gzip
from scipy import stats 

def main():
    
    OUTPUT_TEMPLATE = (
        '"Did more/less users use the search feature?" p-value: {more_users_p:.3g}\n'
        '"Did users search more/less?" p-value: {more_searches_p:.3g}\n'
        '"Did more/less instructors use the search feature?" p-value: {more_instr_p:.3g}\n'
        '"Did instructors search more/less?" p-value: {more_instr_searches_p:.3g}'
    )
#     searchdata_file = sys.argv[1]

    # ...
    
    filename = sys.argv[1]
#     filename = "searches.json"
    searches = pd.read_json(filename,orient='records', lines=True)
    even_samples = searches[searches['uid'] % 2 == 0]
    odd_samples = searches[searches['uid'] % 2 != 0]
    
    even_searched = even_samples[even_samples['search_count'] != 0].shape[0]
    even_unsearched = even_samples[even_samples['search_count'] == 0].shape[0]
    odd_searched = odd_samples[odd_samples['search_count'] != 0].shape[0]
    odd_unsearched = odd_samples[odd_samples['search_count'] == 0].shape[0]

    p_more_searches = stats.mannwhitneyu(even_samples['search_count'],odd_samples['search_count']).pvalue

    contingency = [[even_searched, even_unsearched], [odd_searched, odd_unsearched]]
    chi2, p_more_users, dof, expected = stats.chi2_contingency(contingency)
    
    inst_samples = searches[searches['is_instructor']]
    inst_even_samples = inst_samples[inst_samples['uid'] % 2 == 0]
    inst_odd_samples = inst_samples[inst_samples['uid'] % 2 != 0]
    
    p_more_instr_searches = stats.mannwhitneyu(inst_even_samples['search_count'],inst_odd_samples['search_count']).pvalue

    inst_even_searched = inst_even_samples[inst_even_samples['search_count'] != 0].shape[0]
    inst_even_unsearched = inst_even_samples[inst_even_samples['search_count'] == 0].shape[0]
    inst_odd_searched = inst_odd_samples[inst_odd_samples['search_count'] != 0].shape[0]
    inst_odd_unsearched = inst_odd_samples[inst_odd_samples['search_count'] == 0].shape[0]

    inst_contingency = [[inst_even_searched, inst_even_unsearched], [inst_odd_searched, inst_odd_unsearched]]
    inst_chi2, p_more_instr, inst_dof, inst_expected = stats.chi2_contingency(inst_contingency)
    
    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p=p_more_users,
        more_searches_p=p_more_searches,
        more_instr_p=p_more_instr,
        more_instr_searches_p=p_more_instr_searches,
    ))
    
if __name__ == '__main__':
    main()

