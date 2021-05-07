# ======================= Link to data:
""" 
https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95 
"""


# ======================= Load Libraries: 
# IPython
from IPython.display import display
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
# Data Handeling
import numpy as np 
import pandas as pd 
import calendar
import os 
# Plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



# ======================= Load data:
dataDir = r"C:\Users\boody\OneDrive - Danmarks Tekniske Universitet\01 MSc\2st Semester\01_SD\Local\04 Project\Motor_Vehicle_Collisions.csv"
Data = pd.read_csv(dataDir)


# ======================= Explore data 

# Overview: 

""" Whole data """
Data.head(n=5)              

""" Time """
Data.loc[:,['CRASH DATE','CRASH TIME']] 

""" Place """
Data.loc[:,['LATITUDE','LONGITUDE','BOROUGH']] 

""" VEHICLE TYPE """
Data.loc[:,['VEHICLE TYPE CODE 1','VEHICLE TYPE CODE 2']]                       

""" CONTRIBUTING FACTOR """
Data.loc[:,['CONTRIBUTING FACTOR VEHICLE 1','CONTRIBUTING FACTOR VEHICLE 2']]   

""" PERSONS """
Data.loc[:,['NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED']]            

""" PEDESTRIANS """
Data.loc[:,['NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED']]    

""" CYCLIST """
Data.loc[:,['NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED']]            

""" MOTORIST """
Data.loc[:,['NUMBER OF MOTORIST INJURED','NUMBER OF MOTORIST KILLED']]          


# Count
Data['BOROUGH'].value_counts()
Data['VEHICLE TYPE CODE 1'].value_counts()
Data['VEHICLE TYPE CODE 2'].value_counts()
Data['CONTRIBUTING FACTOR VEHICLE 1'].value_counts()
Data['CONTRIBUTING FACTOR VEHICLE 2'].value_counts()
Data['NUMBER OF PERSONS INJURED'].value_counts()
Data['NUMBER OF PERSONS KILLED'].value_counts()
Data['NUMBER OF CYCLIST INJURED'].value_counts()
Data['NUMBER OF CYCLIST KILLED'].value_counts()
Data['NUMBER OF MOTORIST INJURED'].value_counts()
Data['NUMBER OF MOTORIST KILLED'].value_counts()


# Important: Main Idea
"""
First, 
We can categories the vehicle types into 4 categories, 
which could be for example: Small, Medium, Big, and Huge. 

Then 
we can also make a new column, 'kills_or_injures_occurred', 
which is a binary future that says 0 if there is no injures or killed person and 1 other wise. 

Finally, 
we have a similar task to the one we have been doing with the crime dataset. 
Where we can use the Temp-Spatial (+ Weather and Speed-Limit features which will be download later) to predict if there is kills or injures occurred (0 or 1). 
"""


# ======================= Data Prepration:
# Delete messing values: 
"""
Data = Data.dropna()
"""

# Select relevant columns:
"""
usecols = ['CRASH DATE','CRASH TIME','BOROUGH',
           'LATITUDE','LONGITUDE',
           'VEHICLE TYPE CODE 1','VEHICLE TYPE CODE 2',
           'CONTRIBUTING FACTOR VEHICLE 1','CONTRIBUTING FACTOR VEHICLE 2',
           'NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED',
           'NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED',
           'NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED',
           'NUMBER OF MOTORIST INJURED','NUMBER OF MOTORIST KILLED']
Data = Data[usecols]
"""



