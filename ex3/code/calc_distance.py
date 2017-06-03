
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import xml.etree.ElementTree as ET
from math import radians, cos, sin, asin, sqrt
from pykalman import KalmanFilter


# In[2]:

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')
        
def get_data(filename):
    tree = ET.parse(filename)
    ns = tree.getroot().tag.split('}')[0]+'}'
    pt_nodes = tree.findall("./{0}trk/{0}trkseg/{0}trkpt".format(ns))
    pts = [l.attrib for l in pt_nodes]
    df = pd.DataFrame(pts)
    df['lat'] =df['lat'].apply(float)
    df['lon'] =df['lon'].apply(float)
    return df

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    r = r * 1000 # convert to meters
    return c * r


# In[3]:

def distance(points_df):
    df2 = points_df.copy()
    df2.columns = ['lat1','lon1']
    points_shift = df2.shift(1)
    points_shift.columns = ['lat2','lon2']
    points_comb = df2.join(points_shift)[1:]
    points_comb['dist'] = points_comb.apply(lambda x: haversine(x.lat1,x.lon1,x.lat2,x.lon2),axis=1)
    return points_comb['dist'].sum()


# In[4]:

def smooth(points_df):
    initial_state = points_df.iloc[0]
    observation_stddev = 15 / (10 ** 5)
    transition_stddev = 7 / (10 ** 5)
    observation_covariance = [[observation_stddev ** 2, 0], [0, observation_stddev ** 2]]
    transition_covariance = [[transition_stddev ** 2, 0], [0, transition_stddev ** 2]]


    kf = KalmanFilter(initial_state_mean=initial_state,
                    transition_matrices = [[1, 0], [0, 1]], 
                    observation_matrices = [[1, 0], [0, 1]],
                    observation_covariance = observation_covariance,
                    transition_covariance = transition_covariance)
    kalman_smoothed, _ = kf.smooth(points_df)
    df = pd.DataFrame(columns= ['lat','lon'])
    df['lat'] = pd.to_numeric(kalman_smoothed[:,0])
    df['lon'] = pd.to_numeric(kalman_smoothed[:,1])
    return df


# In[6]:

def main():
    points = get_data(sys.argv[1])

    print('Unfiltered distance: %0.2f' % (distance(points),))
    
    smoothed_points = smooth(points)
    print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    output_gpx(smoothed_points, 'out.gpx')

    # smoothed_points
if __name__ == '__main__':
    main()

