
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import numpy as np
import difflib
import gzip
from scipy import stats
import time
from implementations import all_implementations
from random import randint


# In[ ]:

def main():
    ARR_SIZE = 18000
    SORT_TRIALS = 50
    random_arrays = []
    for i in range(SORT_TRIALS):
        random_arrays.append(np.random.randint(0, ARR_SIZE, ARR_SIZE))
        
    df_result = pd.DataFrame(np.nan, index=np.array(range(SORT_TRIALS)),columns = [fn.__name__ for fn in all_implementations])
    # start = time.time()
    for sort in all_implementations:
        for i in range(SORT_TRIALS):
            st = time.time()
            res = sort(random_arrays[i])
            en = time.time()
            df_result.iloc[i][sort.__name__]=en-st
    # print("Sorted all data: in "+str(time.time()-start)+" sec(s).")
    df_result.to_csv('data.csv', index=False)
if __name__ == '__main__':
    main()

