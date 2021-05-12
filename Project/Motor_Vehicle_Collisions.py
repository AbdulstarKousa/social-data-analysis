# ======================= Link to data:
""" 
https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95 
https://www.ncdc.noaa.gov/cdo-web/search 
https://data.cityofnewyork.us/Transportation/VZV_Speed-Limits/7n5j-865y
"""


# ======================= Important: Main Idea
"""
First,
We make a new column, 'kills_or_injures_occurred', 
which is a binary future that says 0 if there is no injures or killed person and 1 other wise. 

Then,
we have a similar task to the one we have been doing with the crime dataset. 
Where we can use the Tempo-Spatial (+ Weather and Speed-Limit which will be download later) features to predict if there is kills or injures occurred (0 or 1).
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
from bokeh.io import show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource
from bokeh.models import FactorRange
from bokeh.models import Legend
from bokeh.plotting import figure
from bokeh import palettes
output_notebook() # open the bokeh viz on the notebook.




# ======================= Load data:
""" Path """
fileName = 'Motor_Vehicle_Collisions.csv'
filePath = os.path.abspath(os.path.join(os.getcwd(), fileName))

""" Load """
Data =  pd.read_csv(filePath)



# ======================= Functions:
""" Define a function to track Reduction in data """
Reduction = {}
Reduction_Percentage = {}
N = Data.shape[0]
def reduc(step):
    global N 
    global Reduction
    global Reduction_Percentage
    N_before = N
    N_after = Data.shape[0]
    Reduction[step] = N_after
    Reduction_Percentage[step] = (N_before - N_after) / N_before
    print(f'Number of observation: {N_after}  (--{N_before-N_after})')
    print(f'Reduction: {N_before - N_after}  ({(N_before - N_after) / N_before} %)')
    N = Data.shape[0]


# ======================= Getting to know the Dataset: 
""" Initilize Reduction in data """
reduc('Init')

""" Overview """
Data.head(n=5)

""" Data shape """
Data.shape

""" Columns' names """
Data.columns

""" Columns types """
Data.dtypes

""" Count columns' NaN values in desending order """
sorted(list(zip(Data.columns,Data.isna().sum(axis=0).values)) , key= lambda row: row[1], reverse=True)

""" Count columns' Non-NaN values in desending order """
sorted(list(zip(Data.count().keys(),Data.count().values)), key= lambda row: row[1], reverse=True)

""" Count columns' zeros values """
(Data == 0).sum(axis=0)

""" Count columns' empty strings """
(Data == '').sum(axis=0)



# ======================= Data Cleaning: 
# Drop unneeded features:
""" Drop 'COLLISION_ID' since it's not informative """
Data = Data.drop(columns=['COLLISION_ID'])

""" Drop 'LOCATION' since we have 'LATITUDE', 'LONGITUDE' """
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

""" Track Reduction in data """
reduc('MVC with only two vehicles involves')


# Missing Data:
""" Count columns' NaN values in desending order """
sorted(list(zip(Data.columns,Data.isna().sum(axis=0).values)) , key= lambda row: row[1], reverse=True)

""" Count columns' zeros values """
(Data == 0).sum(axis=0)

""" Count columns' empty strings """
(Data == '').sum(axis=0)

""" Drop rows that has a messing value in one of important features """
Data = Data[
    Data['ON STREET NAME'].notna()  & # important feature for adding speed limit data later on.
    Data['LATITUDE' ].notna()       & # imporatnt feature for map plots 
    Data['LONGITUDE'].notna()       & # imporatnt feature for map plots
    Data['NUMBER OF PERSONS INJURED'].notna()   & # imporatnt feature since one of the main features of intress
    Data['NUMBER OF PERSONS KILLED'].notna()      # imporatnt feature since one of the main features of intress
    ].copy()

""" Drop raws with LATITUDE or LONGITUDE = 0 """
Data = Data[(Data['LATITUDE']!=0)|(Data['LONGITUDE']!=0)].copy()

""" Track Reduction in data """
reduc('Drop missing values in important features')




# ======================= Features Prepration: 
# ===== Prepare Vehicle types:
# Prepare Vehicle type 1:
""" Unify Vehicle type recording way """
Data['VEHICLE TYPE CODE 1'] = Data['VEHICLE TYPE CODE 1'].str.lower()
Data['VEHICLE TYPE CODE 1'] = Data['VEHICLE TYPE CODE 1'].str.strip()

""" Fixing recording issus of Vehicle types that has more than 50 MVC occurrences """
Frequent_MVC_Vehicles = (Data['VEHICLE TYPE CODE 1'].value_counts().keys()[Data['VEHICLE TYPE CODE 1'].value_counts().values > 50])
print(Frequent_MVC_Vehicles)

Mapping = {
    np.nan: 'unknown',
    'station wagon/sport utility vehicle': 'sport utility vehicle', 
    'sport utility / station wagon':'sport utility vehicle', 
    '4 dr sedan': 'sedan', 
    'ambul': 'ambulance',  
    'school bus': 'school bus', 
    'e-sco': 'e-scooter', 
    'schoo': 'school bus', 
    'bicycle': 'bike'
    }

Data['VEHICLE TYPE CODE 1'] = Data['VEHICLE TYPE CODE 1'].replace(Mapping)

""" Consider only 95 % Frequent MVC Vehicle types """
VT1 = pd.DataFrame()
VT1['VEHICLE TYPE CODE 1'] = Data['VEHICLE TYPE CODE 1'].value_counts(normalize=True).keys()
VT1['Frequencies'] = Data['VEHICLE TYPE CODE 1'].value_counts(normalize=True).values

threshold = 0
for i in range(len(VT1['VEHICLE TYPE CODE 1'].unique())):
    Sum = VT1['Frequencies'][0:i+1].sum()
    if Sum > 0.95:
         threshold = i + 1
         print("Threshold that covers 95% of " + "VEHICLE TYPEs".lower() +  " = " + f"{threshold}")
         break 
Focus_Vehicles_Type_1 = list(VT1['VEHICLE TYPE CODE 1'][0:threshold].values)
print(Focus_Vehicles_Type_1)

# Prepare Vehicle type 2:
""" Unify Vehicle type recording way """
Data['VEHICLE TYPE CODE 2'] = Data['VEHICLE TYPE CODE 2'].str.lower()
Data['VEHICLE TYPE CODE 2'] = Data['VEHICLE TYPE CODE 2'].str.strip()

""" Fixing recording issus of Vehicle types that has more than 50 MVC occurrences """
Frequent_MVC_Vehicles = (Data['VEHICLE TYPE CODE 2'].value_counts().keys()[Data['VEHICLE TYPE CODE 2'].value_counts().values > 50])
print(Frequent_MVC_Vehicles)

Mapping = {
    np.nan: 'unknown',
    'unkno': 'unknown',
    'unk': 'unknown',
    'station wagon/sport utility vehicle': 'sport utility vehicle', 
    'sport utility / station wagon':'sport utility vehicle', 
    '4 dr sedan': 'sedan', 
    'ambul': 'ambulance',  
    'school bus': 'school bus', 
    'e-sco': 'e-scooter', 
    'schoo': 'school bus', 
    'bicycle': 'bike', 
    }

Data['VEHICLE TYPE CODE 2'] = Data['VEHICLE TYPE CODE 2'].replace(Mapping)

""" Consider only 95 % Frequent MVC Vehicle types """
VT2 = pd.DataFrame()
VT2['VEHICLE TYPE CODE 2'] = Data['VEHICLE TYPE CODE 2'].value_counts(normalize=True).keys()
VT2['Frequencies'] = Data['VEHICLE TYPE CODE 2'].value_counts(normalize=True).values

threshold = 0
for i in range(len(VT2['VEHICLE TYPE CODE 2'].unique())):
    Sum = VT2['Frequencies'][0:i+1].sum()
    if Sum > 0.95:
         threshold = i + 1
         print("Threshold that cover 95% of " + "VEHICLE TYPEs".lower() +  " = " + f"{threshold}")
         break 
Focus_Vehicles_Type_2 = VT2['VEHICLE TYPE CODE 2'][0:threshold].values
print(Focus_Vehicles_Type_2)

# Slice Focus Vehicle Types (covers more than 95 % of MVC occurrences)
""" Slice """
Focus_Vehicle_Types = list(set(list(Focus_Vehicles_Type_1) + list(Focus_Vehicles_Type_2))) 
Data = Data[Data['VEHICLE TYPE CODE 1'].isin((Focus_Vehicle_Types)) & (Data['VEHICLE TYPE CODE 2'].isin(Focus_Vehicle_Types))].copy()
print(Focus_Vehicle_Types)

""" Track Reduction in data """
reduc('Slice Focus Vehicle Types')

""" free memory """
del(VT1,VT2)


# ===== Prepare Contributing Factors:
# Prepare Contributing Factor 1:
""" Unify Contributing Factor string """
Data['CONTRIBUTING FACTOR VEHICLE 1'] = Data['CONTRIBUTING FACTOR VEHICLE 1'].str.lower()
Data['CONTRIBUTING FACTOR VEHICLE 1'] = Data['CONTRIBUTING FACTOR VEHICLE 1'].str.strip()

""" Fixing recording issus of Contributing Factor that has more than 50 MVC occurrences """
Frequent_MVC_Factors = (Data['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().keys()[Data['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().values > 50])
print(Frequent_MVC_Factors)

Mapping = {
    np.nan: 'unknown',
    'illnes':'illness', 
    'reaction to other uninvolved vehicle':'reaction to uninvolved vehicle',
    'passing too closely': 'passing or lane usage improper',
    }

Data['CONTRIBUTING FACTOR VEHICLE 1'] = Data['CONTRIBUTING FACTOR VEHICLE 1'].replace(Mapping)

""" Consider only 95 % Frequent MVC Contributing Factors """
CF1 = pd.DataFrame()
CF1['CONTRIBUTING FACTOR VEHICLE 1'] = Data['CONTRIBUTING FACTOR VEHICLE 1'].value_counts(normalize=True).keys()
CF1['Frequencies'] = Data['CONTRIBUTING FACTOR VEHICLE 1'].value_counts(normalize=True).values

threshold = 0
for i in range(len(CF1['CONTRIBUTING FACTOR VEHICLE 1'].unique())):
    Sum = CF1['Frequencies'][0:i+1].sum()
    if Sum > 0.95:
         threshold = i + 1
         print("Threshold that covers 95% of " + "CONTRIBUTING FACTORs".lower() +  " = " + f"{threshold}")
         break 
Focus_Factors_Type_1 = list(CF1['CONTRIBUTING FACTOR VEHICLE 1'][0:threshold].values)
print(Focus_Factors_Type_1)

# Prepare Contributing Factor 2:
""" Unify Contributing Factor string """
Data['CONTRIBUTING FACTOR VEHICLE 2'] = Data['CONTRIBUTING FACTOR VEHICLE 2'].str.lower()
Data['CONTRIBUTING FACTOR VEHICLE 2'] = Data['CONTRIBUTING FACTOR VEHICLE 2'].str.strip()

""" Fixing recording issus of Contributing Factor that has more than 50 MVC occurrences """
Frequent_MVC_Factors = (Data['CONTRIBUTING FACTOR VEHICLE 2'].value_counts().keys()[Data['CONTRIBUTING FACTOR VEHICLE 2'].value_counts().values > 50])
print(Frequent_MVC_Factors)

Mapping = {
    np.nan: 'unknown',
    'illnes':'illness', 
    'reaction to other uninvolved vehicle':'reaction to uninvolved vehicle',
    'passing too closely': 'passing or lane usage improper',
    }

Data['CONTRIBUTING FACTOR VEHICLE 2'] = Data['CONTRIBUTING FACTOR VEHICLE 2'].replace(Mapping)

""" Consider only 95 % Frequent MVC Contributing Factors """
CF2 = pd.DataFrame()
CF2['CONTRIBUTING FACTOR VEHICLE 2'] = Data['CONTRIBUTING FACTOR VEHICLE 2'].value_counts(normalize=True).keys()
CF2['Frequencies'] = Data['CONTRIBUTING FACTOR VEHICLE 2'].value_counts(normalize=True).values

threshold = 0
for i in range(len(CF2['CONTRIBUTING FACTOR VEHICLE 2'].unique())):
    Sum = CF2['Frequencies'][0:i+1].sum()
    if Sum > 0.95:
         threshold = i + 1
         print("Threshold that covers 95% of " + "CONTRIBUTING FACTORs".lower() +  " = " + f"{threshold}")
         break 
Focus_Factors_Type_2 = list(CF2['CONTRIBUTING FACTOR VEHICLE 2'][0:threshold].values)
print(Focus_Factors_Type_2)

# Slice Focus Factors Type (covers more than 95 % of MVC occurrences)
""" Slice """
Focus_Factors_Types = list(set(list(Focus_Factors_Type_1) + list(Focus_Factors_Type_2))) 
Data = Data[Data['CONTRIBUTING FACTOR VEHICLE 1'].isin((Focus_Factors_Types)) & (Data['CONTRIBUTING FACTOR VEHICLE 2'].isin(Focus_Factors_Types))].copy()
print(Focus_Factors_Types)

""" Track Reduction in data """
reduc('Slice Focus Factors Types')

""" free memory """
del(CF1,CF2)


# =====  Prepare Zip Features:
""" Drop Unspecified Zip """
Data['ZIP CODE'].replace(to_replace='     ', value=np.nan, inplace=True)

""" Change the Zip type to float64 """ 
Data['ZIP CODE'] = pd.to_numeric(Data['ZIP CODE']) 


# =====  Extract new feutres:
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


# =====  Drop uncompleted years:
""" Drop rows from 2012 since they are not completed  """
Data = Data[Data['Year']!=2012]

""" Drop rows from 2021 since they are not completed  """
Data = Data[Data['Year']!=2021]

""" Track Reduction in data """
reduc('Drop uncompleted years')



# ======================= Data Prepration (Cleaning and Transformation): 

# =====  Adding Speed_Limits Mode Data:
""" path """
fileName = 'dot_VZV_Speed_Limits_20210507.csv'
filePath = os.path.abspath(os.path.join(os.getcwd(), fileName))

""" load """
speed_limits =  pd.read_csv(filePath)

""" Drop speed limits rows with missing values in important features """
speed_limits = speed_limits[
        speed_limits['street'].notna()  &
        speed_limits['postvz_sl'].notna()  
    ].copy()


""" Prepare street name features of both datasets for merging """
Data.loc[:,'ON STREET NAME'] = Data['ON STREET NAME'].str.lower()
Data.loc[:,'ON STREET NAME'] = Data['ON STREET NAME'].str.strip()
speed_limits.loc[:,'street'] = speed_limits['street'].str.lower()
speed_limits.loc[:,'street'] = speed_limits['street'].str.strip()

Matched_streets = Data['ON STREET NAME'][Data['ON STREET NAME'].isin(speed_limits['street'])].unique()
print('Merging Speed Limits:') 
print(f"    Number of Matched Streets = {len(Matched_streets)}")
print(f"    Number of Unmatched Streets = {len(Data['ON STREET NAME'].unique()) - len(Matched_streets)}")

""" Calculate speed limits mode"""
Street_Speed_Mode = {}
streets = speed_limits['street'].unique()
for street in streets:
    Street_Values = speed_limits[speed_limits['street']==street]['postvz_sl']
    Street_Mode = stats.mode(Street_Values)[0][0]
    Street_Speed_Mode[street]= Street_Mode

""" Add speed limits mode to Data """
Data = Data[Data['ON STREET NAME'].isin(streets)].copy()
Data['SPEED LIMIT MODE'] = Data['ON STREET NAME'].apply(lambda street: Street_Speed_Mode[street])

""" Track Reduction in data """
reduc('Adding Speed_Limits')

""" Free memory """
del(speed_limits)


# =====  Adding weather data:
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
weather['PRCP'].value_counts().values
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
weather['Snow depth'.upper()].value_counts()


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

""" Track Reduction in data """
reduc('Adding Weather')

""" View and Rename Weather features """
Data['Average wind speed'.upper()] = Data['AWND'].copy()
Data['Maximum temperature'.upper()] = Data['TMAX'].copy()
Data['Minimum temperature'.upper()] = Data['TMIN'].copy()
Data = Data.drop(columns=['AWND','TMAX','TMIN'])

""" free memory """
del(weather)



# ======================= Save final Data:
fileName = 'MVC_SL_W_Final.csv'
filePath = os.path.abspath(os.path.join(os.getcwd(), fileName))
Data.to_csv(filePath)


# ======================= Data Exploration:
# ======== Summary Statistics:
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


# ======== Some interesting counts:
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


# ======== Box and whisker plots
""" For MVC Data """
Box_lst = ['ZIP CODE', 'LATITUDE', 'LONGITUDE', 'NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED', 'Response', 'Year', 'Month', 'Hour','Minute']
Data[Box_lst].plot(
                kind='box', 
                subplots=True, 
                sharex=False, 
                sharey=False, 
                fontsize=10, 
                layout=(4,3), 
                figsize=(10,9),
                title='Box-Plot for MVC data'
                )
plt.show()

""" For speed limits and weather data """
Box_lst = ['SPEED LIMIT MODE', 'PRECIPITATION', 'SNOW FALL','SNOW DEPTH', 'FOG, SMOKE OR HAZE', 'AVERAGE WIND SPEED','MAXIMUM TEMPERATURE', 'MINIMUM TEMPERATURE']
Data[Box_lst].plot(
                kind='box', 
                subplots=True, 
                sharex=False, 
                sharey=False, 
                fontsize=10, 
                layout=(3,3), 
                figsize=(10,7),
                title='Box-Plot for Speed Limits and Weather data'
                )
plt.show()


# ======== Respone over Years - Plot  
grouping = Data.groupby(['Response','Year']).count()['CRASH DATE']

fig, axs = plt.subplots(1, 2,figsize=(10, 5))
fig.suptitle('Development of Respone over Years')

axs.flat[0].set_title('NYC MVC Without injures or kills')
axs.flat[0].set_ylabel('Number of MVC')
grouping[0].plot(ax=axs.flat[0], color= 'tab:gray')    

axs.flat[1].set_title('NYC MVC With injures or kills')
grouping[1].plot(ax=axs.flat[1], color= 'tab:gray')    
axs.flat[1].set_ylabel('Number of MVC')

plt.show


# ======== Correlation Matrix Plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(Data.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0,len(Data.select_dtypes(exclude=object).columns),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(['Zip','LAT.', 'LONG.', 'Injured','killed', 'Response', 'Year', 'Month', 'Hour','Minute','S.L.M.', 'PRE.','S.F.','S.D.','F.S.H','A.W.S.','MAX.T.','MIN.T.'], rotation=90)
ax.set_yticklabels(['Zip','LAT.', 'LONG.', 'Injured','killed', 'Response', 'Year', 'Month', 'Hour','Minute','S.L.M.', 'PRE.','S.F.','S.D.','F.S.H','A.W.S.','MAX.T.','MIN.T.'], rotation=0)
plt.grid(False)
plt.title('Correlation Matrix')
plt.show()


# ======== Jitter-plots:
""" Init filtered data for Jitter plotS """ 
month = 1
hour = 13

""" Jitter plot 01 : No-Response """
data_Jit = Data[Data['Response']==0] 
data_Jit = data_Jit[data_Jit['Month']==month]
data_Jit = data_Jit[data_Jit['Hour']==hour].reset_index(drop=True)
plt.figure(figsize=(7, 4))
sns.stripplot(data_Jit['Minute'].values, jitter=True, edgecolor='none', alpha=.50 ,color='k')
plt.title('NYC MVC without injuries or kills\nMonth = '+ str(month) + '\nHour = ' + str(hour) + '-'+ str(hour+1))
plt.show()

""" Jitter plot 02 : With-Response """
data_Jit = Data[Data['Response']==1] 
data_Jit = data_Jit[data_Jit['Month']==1]
data_Jit = data_Jit[data_Jit['Hour']==13]
plt.figure(figsize=(7, 4))
sns.stripplot(data_Jit['Minute'].values, jitter=True, edgecolor='none', alpha=.50 ,color='k')
plt.title('NYC MVC with injuries or kills\nMonth = '+ str(month) + '\nHour = ' + str(hour) + '-'+ str(hour+1))
plt.show()


# ========  Histogram-Plots: 
""" Init filtered data for histogram plot """ 
month = 1

""" histogram 01: No-Respone """
data_Hist = Data[Data['Response']==0]
data_Hist = data_Hist[data_Hist['Month'] == month]
plt.figure(figsize=(7, 5))
plt.hist(data_Hist['LATITUDE'],bins= 50) 
plt.title("NYC MVC\nwithout injuries or kills\nMonth = " + str(month))
plt.xlabel("Latitude")
plt.ylabel("Number of Observations")
plt.show()                      

""" histogram 02: With-Respone """
data_Hist = Data[Data['Response']==1]
data_Hist = data_Hist[data_Hist['Month'] == month]
plt.figure(figsize=(7, 5))
plt.hist(data_Hist['LATITUDE'], bins=50) 
plt.title("NYC MVC\nwith injuries or kills\nMonth = " + str(month))
plt.xlabel("Latitude")
plt.ylabel("Number of Observations")
plt.show()


# ======== Map-plot:
""" Init filtered data for map plot """ 
select_month    = 1
start_year      = 2018
end_year        = 2019

data_Map = Data[
            (Data['Month'] == select_month)&
            (Data['Year'] >= start_year)&
            (Data['Year']  < end_year)
            ].reset_index(drop=True)

""" Create a NYC Map instances """
MapNYC = folium.Map(
            location = [40.730610, -73.935242], 
            tiles = 'Stamen Toner',
            zoom_start = 12)

""" Add Marker for the City Hall to Map"""
folium.Marker(
    location = [40.712772, -74.006058],
    popup = 'City Hall',
    icon = folium.Icon( 
                color='blue',
                icon='university',
                prefix='fa')).add_to(MapNYC)


""" Start adding points """
for i, row in data_Map.iterrows():
    if(row['Response']==1):
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



# ======== Bokeh-Plot:
# Define a general Bokeh-plot function, for all Contributing Factors and Vehicle Types: 
def Bokeh_plot(plotMe):
    # Function Descreption:  
    """
    Bokeh_plot(plotMe): 
        # A general Bokeh-plot function, for all Contributing Factors and Vehicle Types.
        # Takes plotMe:String, with one of the corresponding possible values: 
            - 'VEHICLE TYPE CODE 1'
            - 'VEHICLE TYPE CODE 2'
            - 'CONTRIBUTING FACTOR VEHICLE 1'
            - 'CONTRIBUTING FACTOR VEHICLE 2'
        # Returns:
            Bokeh plot or Error for invalid PlotMe value
    """

    # Check parameter values: 
    if not( (plotMe == 'VEHICLE TYPE CODE 1') | 
            (plotMe == 'VEHICLE TYPE CODE 2') |
            (plotMe == 'CONTRIBUTING FACTOR VEHICLE 1') | 
            (plotMe == 'CONTRIBUTING FACTOR VEHICLE 2')
        ):
        raise TypeError(
            "Not allowed parameter value for 'plotMe' in function 'Bokeh_plot'\n" +
            "The allowed parameter values are:\n" +
            "   - 'VEHICLE TYPE CODE 1'\n" +
            "   - 'VEHICLE TYPE CODE 2'\n" +
            "   - 'CONTRIBUTING FACTOR VEHICLE 1'\n" +
            "   - 'CONTRIBUTING FACTOR VEHICLE 2'\n"
        )

    # Define parameter corresponding Focus list:
    Focus = []
    if ((plotMe == 'VEHICLE TYPE CODE 1') | (plotMe == 'VEHICLE TYPE CODE 2')):
        Focus = Focus_Vehicle_Types
    elif ((plotMe == 'CONTRIBUTING FACTOR VEHICLE 1') | (plotMe == 'CONTRIBUTING FACTOR VEHICLE 2')):
        Focus = Focus_Factors_Types

    # Define parameter corresponding Figure height and width:
    plot_height,plot_width = 0,0 
    if ((plotMe == 'VEHICLE TYPE CODE 1') | (plotMe == 'VEHICLE TYPE CODE 2')):
        plot_height=400
        plot_width=800
    elif ((plotMe == 'CONTRIBUTING FACTOR VEHICLE 1') | (plotMe == 'CONTRIBUTING FACTOR VEHICLE 2')):
        plot_height=550
        plot_width=800

    # Pivot Data (Table) for Bokeh:
    Table = pd.pivot_table(Data, 
                        index = 'Hour', 
                        columns = plotMe,
                        values = 'CRASH DATE',
                        aggfunc = 'count')

    # Normalize: (div by sum)
    Table = Table.div(Table.sum(axis=0), axis=1)

    # Add Hour column (We need Hour it for Bokeh)
    Table['Hours']=Table.index

    # Convert data to bokeh data 
    source = ColumnDataSource(Table)

    # Create an Empty Bokeh Figure.
    """ first, define x_range. It should be FactorRange of str(x_axis_values) """
    x_range = list(map(str, Table['Hours'].values))  
    x_range = FactorRange(factors=x_range)

    """ then, create the figure """
    p = figure(x_range = x_range, 
            plot_height = plot_height,
            plot_width = plot_width,
            title='Hourly distribution of ' + plotMe.lower(),
            x_axis_label='Hour', 
            y_axis_label='Frequency'
            )
    
    # Loop to create a barplot for each label: 
    """ first, Define colors (one color for each label): """
    colors = palettes.Category20[len(Focus)]
    
    """ then,
    Define an empty list to store legend items. 
    The list contains tuples of label and the corresponding barplot list. 
    Syntax:[(label, [p.vbar]), ....]   
    This will be used later to extract legends using Legend function.
    """
    legend_items = []

    """ start looping """
    for i, label in enumerate(Focus):
        """ 
        p.vbar is a barplot of hour vs fraction. 
        For para see https://docs.bokeh.org/en/latest/docs/reference/plotting.html#bokeh.plotting.Figure.vbar  
        """
        vertical_bars  = p.vbar(x='Hours',  # x_axis (column name from Table), see Table['Hours']  
                        top=label,        # y_axis (column name from Table), see Table 
                        source=source,      # Table in Bokeh format 
                        width=0.9,          # width of each bar in vbar 
                        color=colors[i],    # color each label from the colors list
                        muted=True,         # Start the plot muted 
                        muted_alpha=0.005,  # Shadow of each barplot 
                        fill_alpha=1,       # how much to fill each bar in the barplot 
                        line_alpha=1)       # how much to fill the border of each bar in the barplot
        legend_items.append((label, [vertical_bars])) # store to legend_items list
        
    # Start the interactive figure p
    """ First, Extract legends, legends has the label name and info from the cor. barplot's info """
    legend = Legend(items=legend_items)

    """ Then, define legends' Place. """
    p.add_layout(legend, 'left')

    """ Define the click policy """
    p.legend.click_policy = 'mute'

    """ show """
    show(p)


# Bokeh-plot: Vehicle Type Code 1:
Bokeh_plot('VEHICLE TYPE CODE 1')

# Bokeh-plot: Vehicle Type Code 2:
Bokeh_plot('VEHICLE TYPE CODE 2')

# Bokeh-plot: Contributing Factor Vehicle 1:
Bokeh_plot('CONTRIBUTING FACTOR VEHICLE 1')

# Bokeh-plot: Contributing Factor Vehicle 2:
Bokeh_plot('CONTRIBUTING FACTOR VEHICLE 2')
