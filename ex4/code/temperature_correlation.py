
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import numpy as np
import difflib
import gzip
import matplotlib.pyplot as plt


# In[2]:

# filename1 = "stations.json.gz"
# filename2 = "city_data.csv"
# filename3 = "output.svg" 
filename1 = sys.argv[1]
filename2 = sys.argv[2]
filename3 = sys.argv[3]


# In[3]:

station_fh = gzip.open(filename1, 'rt', encoding='utf-8')
stations = pd.read_json(station_fh, lines=True)
cities = pd.read_table(filename2,sep=',')
# devide by 10 on temperature
stations["avg_tmax"] = stations["avg_tmax"].apply(lambda x: x/10)

# Exclude cities with area > 10000 sq km
cities["area"] = cities["area"].apply(lambda a : a /1000000.0)
cities = cities[cities["area"]<=10000]
cities = cities[cities["area"].notnull()]
cities = cities[cities["population"].notnull()]
cities["pop_den"]=cities["population"]/cities["area"]
# stations


# In[4]:

# cities


# In[5]:

# Calculate distance from lat/lon. Adapted from stackoverflow.com
# https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def distance(city, stations):
    p = 0.017453292519943295     #Pi/180
    lat1 = city["latitude"]
    lon1 = city["longitude"]
    lat2 = stations["latitude"]
    lon2 = stations["longitude"]
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 12742 * np.arcsin(np.sqrt(a)) #2*R*asin...

def best_tmax(city, stations):
    distances = distance(city, stations)
    station_min_dist = distances.idxmin()
    station_min = stations.iloc[station_min_dist]
    return station_min["avg_tmax"]

cities["avg_tmax"] = cities.apply(best_tmax, stations=stations, axis =1)


# In[6]:

plt.title('Temperature vs Population Density')
plt.xlabel("Avg Max Temperature (\u00b0C)")
plt.ylabel("Population Density (people/km\u00b2)")
plt.plot(cities["avg_tmax"],cities["pop_den"].values,'b.')
# plt.show()
plt.savefig(filename3)

