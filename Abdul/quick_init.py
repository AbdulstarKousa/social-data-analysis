####################################
# quick_init()
####################################

# library
import numpy as np
import pandas as pd

from IPython.display import display
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()

import folium
import calendar
import os 


# Load Data
""" Data : !=2018 """
fileName = 'Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv'
filePath = os.path.abspath(os.path.join(os.getcwd(), '..' ,'Datasets', fileName))
Data = pd.read_csv(filePath)
Data['Year'] = pd.to_datetime(Data['Date']).dt.year
Data = Data[Data['Year'] != 2018]

""" data : Data in fc """
focuscrimes = set(['WEAPON LAWS', 'PROSTITUTION', 'DRIVING UNDER THE INFLUENCE', 'ROBBERY', 'BURGLARY', 'ASSAULT', 'DRUNKENNESS', 'DRUG/NARCOTIC', 'TRESPASS', 'LARCENY/THEFT', 'VANDALISM', 'VEHICLE THEFT', 'STOLEN PROPERTY', 'DISORDERLY CONDUCT'])
focuscrimes_lst = list(focuscrimes)
focuscrimes_lst.sort()
data = Data[Data['Category'].isin(focuscrimes_lst)]
