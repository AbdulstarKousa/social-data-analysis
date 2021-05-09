# ======================= Link to data:
""" 
https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95 
https://www.ncdc.noaa.gov/cdo-web/search 
https://data.cityofnewyork.us/Transportation/VZV_Speed-Limits/7n5j-865y
"""

# ======================= Load Libraries: 
""" IPython """
from IPython.display import display
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

""" Data Handeling """
import numpy as np 
import pandas as pd 
from scipy import stats
import calendar
import os 

""" Plot """ 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import folium


# ======================= Load data:
""" Path """
fileName = 'Motor_Vehicle_Collisions.csv'
filePath = os.path.abspath(os.path.join(os.getcwd(), fileName))

""" Load """
Data =  pd.read_csv(filePath)



# ======================= Important: Main Idea
"""
First,
We make a new column, 'kills_or_injures_occurred', 
which is a binary future that says 0 if there is no injures or killed person and 1 other wise. 

Then,
we have a similar task to the one we have been doing with the crime dataset. 
Where we can use the Tempo-Spatial (+ Weather and Speed-Limit which will be download later) features to predict if there is kills or injures occurred (0 or 1).
"""

# ======================= Getting to know the Dataset: 
""" Overview """
Data.head(n=5)

""" Data shape """
Data.shape

""" Data info """
Data.info()

""" Columns types """
Data.dtypes

""" Columns' names """
Data.columns

""" Count columns' non-NaN values """
Data.count()

""" Count columns' NaN values """
Data.isna().sum(axis=0)

""" Return the columns with the must Nan values """
Data.count().idxmin()

""" Count columns' zeros values """
(Data == 0).sum(axis=0)

""" Count columns' empty strings """
(Data == '').sum(axis=0)


# ======================= Data Prepration (Cleaning and Transformation): 

# Drop unneeded features and observation 
""" Drop 'COLLISION_ID' since it's not informative """
Data = Data.drop(columns=['COLLISION_ID'])

""" Drop 'LOCATION' since 'LATITUDE', 'LONGITUDE' """
Data = Data.drop(columns=['LOCATION'])

""" Drop 'CROSS STREET NAME' and 'OFF STREET NAME' since we have 'ON STREET NAME' """
Data = Data.drop(columns=['CROSS STREET NAME', 'OFF STREET NAME'])

""" Drop PEDESTRIANS, CYCLIST and MOTORIST features since we have PERSONS features """
Data = Data.drop(columns = ['NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED', 
                            'NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED', 
                            'NUMBER OF MOTORIST INJURED','NUMBER OF MOTORIST KILLED'])

""" Consider only Collisions with two vehicles involve and Drop other unrelated features """
Data = Data[
        (Data['CONTRIBUTING FACTOR VEHICLE 3'].isna())|
        (Data['CONTRIBUTING FACTOR VEHICLE 4'].isna())|
        (Data['CONTRIBUTING FACTOR VEHICLE 5'].isna())|
        (Data['VEHICLE TYPE CODE 3'].isna())|
        (Data['VEHICLE TYPE CODE 4'].isna())|
        (Data['VEHICLE TYPE CODE 5'].isna())]
Data = Data.drop(columns=['CONTRIBUTING FACTOR VEHICLE 3','CONTRIBUTING FACTOR VEHICLE 4','CONTRIBUTING FACTOR VEHICLE 5','VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5'])

""" Drop missing values from the remaning features """
Data = Data.dropna()

""" Drop raws with LATITUDE or LONGITUDE = 0 """
Data = Data[(Data['LATITUDE']!=0)|(Data['LONGITUDE']!=0)]


# Prepare Categorical features: Vehicle type, Contributing Factor and Zip Features
""" Classify Vehicle type """
Data['VEHICLE TYPE CODE 1'] = Data['VEHICLE TYPE CODE 1'].str.lower()
Data['VEHICLE TYPE CODE 2'] = Data['VEHICLE TYPE CODE 2'].str.lower()
Data['VEHICLE TYPE CODE 1'] = Data['VEHICLE TYPE CODE 1'].str.strip()
Data['VEHICLE TYPE CODE 2'] = Data['VEHICLE TYPE CODE 2'].str.strip()
"""
More work to be done later, see draft.
    Data.groupby(['VEHICLE TYPE CODE 1'])['VEHICLE TYPE CODE 1'].count().sort_values(ascending=False).head(60)
    Data.groupby(['VEHICLE TYPE CODE 2'])['VEHICLE TYPE CODE 2'].count().sort_values(ascending=False).head(60)
"""

""" Classify Contributing Factor type """
Data['CONTRIBUTING FACTOR VEHICLE 1'] = Data['CONTRIBUTING FACTOR VEHICLE 1'].str.lower()
Data['CONTRIBUTING FACTOR VEHICLE 2'] = Data['CONTRIBUTING FACTOR VEHICLE 2'].str.lower()
"""
More work to be done later, see draft. 
"""

""" Zip Feature """
"""
More work to be done later, see draft. 
"""

# Adding new feutres:
""" Add the 'Respone' feature, which is a binary future that says 0 if there is no injures or killed person and 1 other wise. """
Data['Response'] = Data[['NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED']].sum(axis=1)
Data['Response'] = Data['Response'].apply(lambda y: 1 if y > 0 else 0)

""" Add 'Year' feature """
Data['Year']    = pd.to_datetime(Data['CRASH DATE']).dt.year

""" Add 'Month' feature """
Data['Month']    = pd.to_datetime(Data['CRASH DATE']).dt.month

""" Add 'Hour' feature """
Data['Hour'] = pd.to_datetime(Data['CRASH TIME']).dt.hour

""" Add 'Minute' feature """
Data['Minute'] = pd.to_datetime(Data['CRASH TIME']).dt.minute

# Drop uncompleted years
""" Drop rows from 2012 since they are not completed  """
Data = Data[Data['Year']!=2012]

""" Drop rows from 2021 since they are not completed  """
Data = Data[Data['Year']!=2021]

# Adding Speed_Limits Mode data:

""" path """
fileName = 'dot_VZV_Speed_Limits_20210507.csv'
filePath = os.path.abspath(os.path.join(os.getcwd(), fileName))

""" load """
speed_limits =  pd.read_csv(filePath)

""" Drop speed limits missing values """
speed_limits = speed_limits.dropna()

""" Prepare street name features of both datasets for merging """
Data.loc[:,'ON STREET NAME'] = Data['ON STREET NAME'].str.lower()
Data.loc[:,'ON STREET NAME'] = Data['ON STREET NAME'].str.strip()
speed_limits.loc[:,'street'] = speed_limits['street'].str.lower()
speed_limits.loc[:,'street'] = speed_limits['street'].str.strip()

Matched_streets = Data['ON STREET NAME'][Data['ON STREET NAME'].isin(speed_limits['street'])].unique()
print('Merging Speed Limits:') 
print(f"    Number of Matched Streets = {len(Matched_streets)}")
print(f"    Number of Unmatched Streets = {len(Data['ON STREET NAME'].unique()) - len(Matched_streets)}")

""" calculate speed limits mode the mode """
Street_Speed_Mode = {}
streets = speed_limits['street'].unique()
for street in streets:
    Street_Values = speed_limits[speed_limits['street']==street]['postvz_sl']
    Street_Mode = stats.mode(Street_Values)[0][0]
    Street_Speed_Mode[street]= Street_Mode

""" Add speed limits mode to Data """
Data = Data[Data['ON STREET NAME'].isin(streets)]
Data['SPEED LIMIT MODE'] = Data['ON STREET NAME'].apply(lambda street: Street_Speed_Mode[street])

""" Free memory """
del(speed_limits)



# Adding weather data:
"""
Attributes description:
    AWND : Average wind speed

    TMAX : Maximum temperature
    TMIN : Minimum temperature

    PRCP : Precipitation
    WT16 : Rain(may include freezing rain, drizzle, and freezing drizzle)"

    SNOW : Snowfall
    SNWD : Snow depth
    WT18 : Snow, snow pellets, snow grains, or ice crystals

    WT08 : Smoke or haze
    WT22 : Ice fog or freezing fog
    WT01 : Fog, ice fog, or freezing fog (may include heavy fog)
    WT02 : Heavy fog or heaving freezing fog (not always distinguished from fog)
    WT13 : Mist

    WT06 : Glaze or rime
"""


""" Path """
fileName = 'weather.csv'
filePath = os.path.abspath(os.path.join(os.getcwd(), fileName))

""" Load """
weather =  pd.read_csv(filePath)

""" Slice needed features for further investigation """
weather_features = (
    ['DATE'] + # date
    ['AWND'] + # wind related 
    ['TMAX','TMIN'] + # temp related
    ['PRCP','WT16'] + # rain related
    ['SNOW','SNWD','WT18'] + # snow related
    ['WT08','WT22','WT01','WT02','WT13'] + # fog/vision related
    ['WT06'] # rime related
    )
weather = weather[weather_features]
weather = weather.fillna(0)


""" prepare rain related features: 
        PRCP : Precipitation
        WT16 : Rain(may include freezing rain, drizzle, and freezing drizzle)"
"""
weather[['PRCP','WT16']]
weather['PRCP'].value_counts()
weather['WT16'].value_counts() # 23 

weather['Precipitation'.upper()] = weather['PRCP'].copy()
weather = weather.drop(columns=['PRCP','WT16'])

weather['Precipitation'.upper()].value_counts()


""" prepare snow related features:
        SNOW : Snowfall
        SNWD : Snow depth
        WT18 : Snow, snow pellets, snow grains, or ice crystals
"""
weather[['SNOW','SNWD','WT18']]
weather['SNOW'].value_counts()
weather['SNWD'].value_counts()
weather['WT18'].value_counts() # 21

weather['Snow fall'.upper()] = weather['SNOW'].copy()
weather['Snow depth'.upper()] = weather['SNWD'].copy()
weather = weather.drop(columns=['SNOW','SNWD','WT18'])

weather['Snow fall'.upper()].value_counts()
weather['Snow fall'.upper()].value_counts()


""" prepare fog/vision related features:
        WT01 : Fog, ice fog, or freezing fog (may include heavy fog)
        WT08 : Smoke or haze
        WT02 : Heavy fog or heaving freezing fog (not always distinguished from fog)
        WT13 : Mist
        WT22 : Ice fog or freezing fog
"""
weather[['WT08','WT22','WT01','WT02','WT13']]
weather['WT01'].value_counts()
weather['WT08'].value_counts()
weather['WT02'].value_counts()
weather['WT13'].value_counts() # 27
weather['WT22'].value_counts() # 2

weather['Fog, Smoke or haze'.upper()] = np.where(weather[['WT01','WT08','WT02']].sum(axis=1) == 0, 0, 1)
weather = weather.drop(columns=['WT08','WT22','WT01','WT02','WT13'])

weather['Fog, Smoke or haze'.upper()].value_counts()


""" prepare rime related features """
"""
    WT06 : Glaze or rime
"""
weather['WT06']
weather['WT06'].value_counts() # 14
weather = weather.drop(columns=['WT06'])


""" Merage weather data with Data """
weather['DATE'] = pd.to_datetime(weather['DATE']).dt.date
Data['DATE'] = pd.to_datetime(Data['CRASH DATE']).dt.date

Data = pd.merge(Data, weather, on='DATE', how='left')
Data = Data.drop(columns=['DATE'])


""" free memory """
del(weather)


""" View and rename """
Data['Average wind speed'.upper()] = Data['AWND'].copy()
Data['Maximum temperature'.upper()] = Data['TMAX'].copy()
Data['Minimum temperature'.upper()] = Data['TMIN'].copy()
Data = Data.drop(columns=['AWND','TMAX','TMIN'])

Data.head(n=2)
Data.columns


















# ======================= Data Exploration:
# Summary Statistics:
""" for object features """
Desc_O = Data.select_dtypes(include=object).describe()
display(Desc_O)

""" for non-object features """
Desc_N = Data.select_dtypes(exclude=object).describe()
Desc_N.loc['range'] = Desc_N.loc['max'] - Desc_N.loc['min']
Desc_N = Desc_N.append( Data.select_dtypes(exclude=object)
                            .mode()
                            .rename({0: 'mode'}, axis='index'))

from pandas import set_option
set_option('precision', 2)
display(Desc_N)


# Some intressteing counts:
""" Number of MVC injured persons 2013 - 2020 """
Data['NUMBER OF PERSONS INJURED'].sum()

""" Number of MVC killed persons 2013 - 2020 """
Data['NUMBER OF PERSONS KILLED'].sum()

""" Number Respone since 2013 - 2020 """
Data['Response'].sum()

""" Number of No-Respone since 2013 - 2020 """
(Data['Response'] == 0).sum()

""" Respone over Years"""
grouping = Data.groupby(['Response','Year']).count()['Hour'].sort_values(ascending=False)
print(grouping[0])
print(grouping[1])


# Correlation Matrix 
display(Data.corr(method='pearson'))


# Box and whisker plots
Box_lst = ['LATITUDE', 'LONGITUDE', 'NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED', 'Response', 'Year', 'Month', 'Hour', 'Minute']
Data[Box_lst].plot(
                kind='box', 
                subplots=True, 
                sharex=False, 
                sharey=False, 
                fontsize=10, 
                layout=(3,3), 
                figsize=(10,7))
plt.show()


# Respone over Years - Plot  
grouping = Data.groupby(['Response','Year']).count()['Hour']

fig, axs = plt.subplots(1, 2,figsize=(10, 5))
fig.suptitle('Development of Respone over Years')

axs.flat[0].set_title('NYC MVC Without injures or kills')
axs.flat[0].set_ylabel('Number of MVC')
grouping[0].plot(ax=axs.flat[0], color= 'tab:gray')    

axs.flat[1].set_title('NYC MVC With injures or kills')
grouping[1].plot(ax=axs.flat[1], color= 'tab:gray')    
axs.flat[1].set_ylabel('Number of MVC')

plt.show


# Correlation Matrix Plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(Data.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0,len(Data.select_dtypes(exclude=object).columns),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(['LAT.', 'LONG.', 'Injured.','killed', 'Response', 'Year', 'Month', 'Hour','Minute'], rotation=90)
ax.set_yticklabels(['LAT.', 'LONG.', 'Injured.','killed', 'Response', 'Year', 'Month', 'Hour','Minute'], rotation=0)
plt.grid(False)
plt.title('Correlation Matrix')
plt.show()


# Jitter - plots
"""
Response = 0
Month = Jan. 
Minutes   = 13:00 - 14:00  
"""
data_Jit = Data[Data['Response']==0] 
data_Jit = data_Jit[data_Jit['Month']==1]
data_Jit = data_Jit[data_Jit['Hour']==13].reset_index(drop=True)
plt.figure(figsize=(7, 4))
sns.stripplot(data_Jit['Minute'].values, jitter=True, edgecolor='none', alpha=.50 ,color='k')
plt.title('NYC MVC without injuries or kills\nJan 13:00-14:00')
plt.show()

"""
Response = 1
Month = Jan. 
Minutes   = 13:00 - 14:00  
"""
data_Jit = Data[Data['Response']==1] 
data_Jit = data_Jit[data_Jit['Month']==1]
data_Jit = data_Jit[data_Jit['Hour']==13]
plt.figure(figsize=(7, 4))
sns.stripplot(data_Jit['Minute'].values, jitter=True, edgecolor='none', alpha=.50 ,color='k')
plt.title('NYC MVC with injuries or kills\nJan 13:00-14:00')
plt.show()


# Histogram - Plots 
"""
Latitude
Response = 0
Month = Jan. 
"""
data_Hist = Data[Data['Response']==0]
data_Hist = data_Hist[data_Hist['Month'] == 1]
plt.figure(figsize=(7, 5))
plt.hist(data_Hist['LATITUDE'],bins= 50) 
plt.title("NYC MVC\nwithout injuries or kills\nJan.")
plt.xlabel("Latitude")
plt.ylabel("Number of Observations")
plt.show()                      

"""
Latitude
Response = 1
Month = Jan. 
"""
data_Hist = Data[Data['Response']==1]
data_Hist = data_Hist[data_Hist['Month'] == 1]
plt.figure(figsize=(7, 5))
plt.hist(data_Hist['LATITUDE'], bins=50) 
plt.title("NYC MVC\nwith injuries or kills\nJan")
plt.xlabel("Latitude")
plt.ylabel("Number of Observations")
plt.show()


# Map - plot
""" Create a NYC Map instances """
MapNYC = folium.Map(
            location = [40.730610, -73.935242], 
            tiles = 'Stamen Toner',
            zoom_start = 12)

""" Display Map """
display(MapNYC)


""" Add Marker for the City Hall to Map"""
folium.Marker(
    location = [40.712772, -74.006058],
    popup = 'City Hall',
    icon = folium.Icon( 
                color='blue',
                icon='university',
                prefix='fa')).add_to(MapNYC)

""" Display Map"""
display(MapNYC)


"""
Add NYC MVC Response location
Year  = 2018 - 2020
Month = Jan. 
"""
""" filtered data """ 
data_Map = Data[
            (Data['Month'] == 1)&
            (Data['Year'] >= 2018)&
            (Data['Year']  < 2021)
            ].reset_index(drop=True)


""" Start adding points """
for i, row in data_Map.iterrows():
    if(data_Map.iloc[i]['Response']==1):
        folium.CircleMarker(
            location = [row['LATITUDE'], row['LONGITUDE']],
            radius=1,
            popup='Either Injure or Kill Occurred\nin ' + str(row['CRASH DATE']) +"\nat " + str(row['CRASH TIME']),
            color='red',
            opacity=0.5).add_to(MapNYC)

    else:
        folium.CircleMarker(
            location = [row['LATITUDE'], row['LONGITUDE']],
            radius=1,
            popup='Neither Injure nor Kill Occurred\nin ' + str(row['CRASH DATE']) +"\nat " + str(row['CRASH TIME']),
            color='blue',
            opacity=0.5).add_to(MapNYC)

""" Display Map"""
display(MapNYC)


