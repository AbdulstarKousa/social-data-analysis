# Libraries
from IPython.display import display
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

    # Data Handling
import numpy as np 
import pandas as pd 
import calendar
import os 

    # Plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import folium


# Import dataset using os
fileName = 'Motor_Vehicle_Collisions_-_Crashes.csv'
filePath = os.path.abspath(os.path.join(os.getcwd(), fileName))

Data =  pd.read_csv(filePath)

# ======================= Data Prepration (Cleaning and Transformation): 

""" Drop COLLISION_ID since it's not informative """
Data = Data.drop(columns=['COLLISION_ID'])

""" Drop 'LOCATION' since 'LATITUDE', 'LONGITUDE' """
Data = Data.drop(columns=['LOCATION'])

""" Drop 'CROSS STREET NAME' and 'OFF STREET NAME' since we have 'ON STREET NAME' """
Data = Data.drop(columns=['CROSS STREET NAME', 'OFF STREET NAME'])

""" Drop PEDESTRIANS, CYCLIST and MOTORIST features since we have PERSONS features """
Data = Data.drop(columns = ['NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED', 
                            'NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED', 
                            'NUMBER OF MOTORIST INJURED','NUMBER OF MOTORIST KILLED'])

""" Consider only Collisions with two vehicle involve and Drop other unrelated features """
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

""" Classify Vehicle type """
Data['VEHICLE TYPE CODE 1'] = Data['VEHICLE TYPE CODE 1'].str.lower()
Data['VEHICLE TYPE CODE 2'] = Data['VEHICLE TYPE CODE 2'].str.lower()
Data['VEHICLE TYPE CODE 1'] = Data['VEHICLE TYPE CODE 1'].str.strip()
Data['VEHICLE TYPE CODE 2'] = Data['VEHICLE TYPE CODE 2'].str.strip()

Data.groupby(['VEHICLE TYPE CODE 1'])['VEHICLE TYPE CODE 1'].count().sort_values(ascending=False).head(60)
Data.groupby(['VEHICLE TYPE CODE 2'])['VEHICLE TYPE CODE 2'].count().sort_values(ascending=False).head(60)
#Data['VEHICLE TYPE CODE 1'].unique() # Line to see unique values of Vehicle types 

"""As we see the vehicle types are very badly typed by the officers at the time of data collecting  
- meaning when the actual crimes happen - so I will "manually" try to address and combine typos in the category"""

mappings = {
    '2 dr sedan': 'sedan', '4 dr sedan': 'sedan',
    'school bus': 'bus', 'station wagon/sport utility vehicle': 'suv',
    'motorscooter': 'motorcycle', 'scooter': 'motorcycle', 'motorbike': 'motorcycle', 
    'ambul': 'ambulance', 'bicycle': 'bike', 'fdny': 'firetruck', 'fire': 'firetruck', 
    'firet': 'firetruck', 'e-sco': 'e-scooter', 'small com veh(4 tires)': 'commercial vehicle',
    'large com veh(6 or more tires)': 'commercial vehicle', 'pick-up truck': 'truck',
    'box truck': 'truck', 'tractor truck diesel': 'tanker', 'tow truck / wrecker': 'truck',
    'armored truck': 'truck', 'beverage truck': 'truck', 'pedicab': 'bike', 'flat bed': 'truck',
    'unkno': 'unknown', 'unk': 'unknown', 'stake or rack': 'truck', 'scoot': 'scooter', 'convertible': 'passenger vehicle',
    'carry all': 'passenger vehicle', 'pk': 'truck', 'tractor truck gasoline': 'tanker', 'ambu': 'ambulance',
    'tank': 'tanker', 'e-bik': 'bike', 'usps': 'truck', '3-door': 'passenger vehicle', 'refrigerated van': 'van', 
    'garbage or refuse': 'garbage truck', 'dump': 'garbage truck', 'e-bike': 'bike', 'schoo': 'bus', 
    'concrete mixer': 'truck', 'moped': 'motorcycle', 'sport utility / station wagon': 'suv', 
    'sedan': 'passenger vehicle', 'tract': 'tractor', 'snow plow': 'truck', 'comme': 'commercial vehicle',
    'fdny fire': 'firetruck', 'fdny truck': 'firetruck', 'fire truck': 'firetruck'
}

Data['VEHICLE TYPE CODE 1'] = Data['VEHICLE TYPE CODE 1'].replace(mappings)

    #Grouping by top values to see the top vehicle types
Data.groupby(['VEHICLE TYPE CODE 1'])['VEHICLE TYPE CODE 1'].count().sort_values(ascending=False).head(60)

# Make a `vehicle_types` list of the top 23 types of vehicles, drop the rest 
vehicle_types = list(Data.groupby(['VEHICLE TYPE CODE 1'])['VEHICLE TYPE CODE 1'].count().sort_values(ascending=False).head(24).index)
Data['VEHICLE TYPE CODE 2'] = Data['VEHICLE TYPE CODE 2'].replace(mappings)
Data = Data[Data['VEHICLE TYPE CODE 1'].isin(vehicle_types)]
# Data.groupby(['VEHICLE TYPE CODE 2'])['VEHICLE TYPE CODE 2'].count().sort_values(ascending=False).head(60)


""" Classify Contributing Factor type """
Data['CONTRIBUTING FACTOR VEHICLE 1'] = Data['CONTRIBUTING FACTOR VEHICLE 1'].str.lower()
Data['CONTRIBUTING FACTOR VEHICLE 2'] = Data['CONTRIBUTING FACTOR VEHICLE 2'].str.lower()


mappings_factors = {
    'illnes': 'illness'
}

Data['CONTRIBUTING FACTOR VEHICLE 1'] = Data['CONTRIBUTING FACTOR VEHICLE 1'].replace(mappings_factors)
Data['CONTRIBUTING FACTOR VEHICLE 2'] = Data['CONTRIBUTING FACTOR VEHICLE 2'].replace(mappings_factors)


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

""" Drop rows from 2021 since they are not completed  """
Data = Data[Data['Year']!=2021]

""" Drop rows from 2012 since they are not completed  """
Data = Data[Data['Year']!=2012]

Data['DATE'] = pd.to_datetime(Data['CRASH DATE']).dt.date

#Introducing weather data

fileName2 = 'weather.csv'
filePath2 = os.path.abspath(os.path.join(os.getcwd(), fileName2))

weather =  pd.read_csv(filePath2)

weather.drop(['STATION', 'NAME'], axis=1, inplace=True)
weather['DATE'] = pd.to_datetime(weather['DATE']).dt.date
weather = weather.fillna(0)

Final_data = pd.merge(Data, weather, on='DATE', how='left')
Final_data.drop(['DATE'], axis=1, inplace=True)
Final_data.to_csv('dataset_with_weather.csv')