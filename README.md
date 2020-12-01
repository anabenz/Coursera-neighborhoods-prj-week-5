# Coursera-neighborhoods-prj-week-5
import requests
import pandas as pd
from bs4 import BeautifulSoup
List_url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
source = requests.get(List_url).text

soup = BeautifulSoup(source, 'xml')
table = soup.find('table')
#dataframe will consist of three columns: PostalCode, Borough, and Neighborhood
column_names = ['Postalcode','Borough','Neighborhood']
df = pd.DataFrame(columns = column_names)

# Search all the data
for tr_cell in table.find_all('tr'):
    row_data=[]
    for td_cell in tr_cell.find_all('td'):
        row_data.append(td_cell.text.strip())
    if len(row_data)==3:
        df.loc[len(df)] = row_data

df.head()
# Get index where Borough = Not assigned and Drop the row
df.drop(labels = df[df['Borough']=='Not assigned'].index, axis = 0, inplace = True)
df.head()
# Make sure that if a cell has a borough but a 'Not assigned' neighborhood, then the neighborhood will be the same as the borough
df.loc[(df.Neighborhood == 'Not assigned'),'Neighborhood']=df['Borough']
df.head()
# Make sure than no more than one neighborhood can exist in one postal code area by groupind them and separate them by a coma
df = df.groupby('Postalcode').agg({'Borough' : 'first', 'Neighborhood' : ','.join}).reset_index()
df.head()

# Verify that all Postalcode data are now unique
boolean = df['Postalcode'].duplicated().any()
print(boolean)

print("The final shape of the dataframe is :", df.shape)
# Create a loop to get the coordinates

def get_geocode(postal_code):
    # initialize your variable to None
    lat_lng_coords = None
    while(lat_lng_coords is None):
        g = geocoder.google('{}, Toronto, Ontario'.format(postal_code))
        lat_lng_coords = g.latlng
    latitude = lat_lng_coords[0]
    longitude = lat_lng_coords[1]
    return latitude,longitude
    
# Get the dataset that includes latitude and longitude coordinates for each neighborhood 
df_geo=pd.read_csv('http://cocl.us/Geospatial_data')
df_geo.head()
# Merge the dataframe with latitude and longitude coordinates and the previous dataframe
df_geo.rename(columns={'Postal Code':'Postalcode'},inplace=True)
df_merge = pd.merge(df_geo, df, on='Postalcode')
df_tot = df_merge[['Postalcode','Borough','Neighborhood','Latitude','Longitude']]
df_tot.head()

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#conda config --append channels conda-forge

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
!pip install folium
import folium # map rendering library

import sys
!{sys.executable} -m pip install reverse_geocoder
!pip install geopy
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

print('Libraries imported.')
print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(df_tot['Borough'].unique()),
        df_tot.shape[0]
    )
)

# Get latitude and longitude coordinates of Toronto City

address = 'Toronto, Canada'

geolocator = Nominatim(user_agent="can_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))

# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_tot['Latitude'], df_tot['Longitude'], df_tot['Borough'], df_tot['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=2,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto

# Define Foursquare Credentials and Version

LIMIT = 100

CLIENT_ID = 'PXNGKJNS1KJ23E5I4MF1H1TW0K4FPJ50M1JZHXPJCN0QNN1Q' # your Foursquare ID
CLIENT_SECRET = 'R1PH2KT1WRSKXTZ5J20AMRLGTU2HOB5UFCT0HP124YBS13CI' # your Foursquare Secret
ACCESS_TOKEN = 'JVF5MPJBK1NWV355XZD0LC1USC2SEVS4LASKVPPJOSWAYODN' # your FourSquare Access Token
VERSION = '20180604'
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
#create a function loop over all the neighborhoods in Toronto
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
     nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
    # run the above function on each neighborhood and create a new dataframe called toronto_venues
toronto_venues = pd.DataFrame(getNearbyVenues(names=df_tot['Neighborhood'],
                                   latitudes=df_tot['Latitude'],
                                   longitudes=df_tot['Longitude'],
                                  ))
                                  
# We will print the size of the new dataframe toronto_venues and see more about this dataframe
print(toronto_venues.shape)
toronto_venues.head()
# General informations
print('There are {} venues in Toronto.'.format(toronto_venues['Venue Category'].count()))
print('There are {} uniques categories of venue.'.format(len(toronto_venues['Venue Category'].unique())))
print('There are {} uniques Neighborhood.'.format(len(toronto_venues['Neighborhood'].unique())))
# Count the number of total venue in each Neighborhood
venues_tot = toronto_venues.groupby('Neighborhood').count()
venues_tot.sort_values('Venue',ascending=False,inp lace=True)
venues_tot.drop(labels=['Neighborhood Latitude', 'Neighborhood Longitude', 'Venue Latitude', 'Venue Longitude','Venue Category'], axis=1, inplace=True)
venues_tot.head()
# Take the top 20 of Neighborhood in Toronto city that count the higher number of venues 
venues_top20 = venues_tot.head(20)
venues_temp = toronto_venues.groupby('Neighborhood').agg({'Venue Category': 'first', 'Venue Category' : ','.join}).reset_index()
venues_temp.head(20)

#Let's explore the first neighborhood in terms of higher number of venue  

Calgary_venues_NE = venues_temp[venues_temp.Neighborhood=='Toronto Dominion Centre, Design Exchange']
Calgary_venues_NE
import matplotlib.pyplot as plt

#plot a bar graph to see which neighborhoods have the most restaurants

bar_graph_data=pd.DataFrame(venues_top20['Venue']).reset_index()
x=bar_graph_data.iloc[:,0]
y=bar_graph_data.iloc[:,1]

plt.figure(figsize=(20,4))
plt.bar(x, y)

#plot labels and titles
plt.xlabel('Neighborhood')
plt.xticks(rotation=70, ha='right')
plt.ylabel('Number of Venue')

plt.show()
# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()
# Examine the new data frame
toronto_onehot.shape
# Here rows are grouped by neighborhood by taking the mean of the frequency of occurrence of each category

toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped

# print each neighborhood along with the top 5 most common venues
num_top_venues = 5

for hood in toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')
    # We will write a function to sort the venues and then create a new dataframe and display the top 10 venues for each neighborhood.

import numpy as np 

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
# set number of clusters
kclusters = 7

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = df_tot

# merge manhattan_grouped with manhattan_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head() # check the last columns!

# Map all clusters
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        fill=True,
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters

toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]

toronto_merged.loc[toronto_merged['Cluster Labels'] == 5, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
toronto_merged.loc[toronto_merged['Cluster Labels'] == 6, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
