# ======== libraries
import numpy as np
import pandas as pd
import os 
from IPython.display import display
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
    

# ======== Load Data
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
data = Data[Data['Category'].isin(focuscrimes_lst)].copy()


# ======== prepare data
""" slice: [2010 - 2018[ """
data = data.loc[(Data['Year'] >= 2010)]
data['Hour'] = pd.to_datetime(Data['Time']).dt.hour

""" Normalize: div by sum """
group_count = data.groupby(['Category','Hour'])['PdId'].count()
group_count = group_count.astype('float64') # float64 to avoid float to int cast when div by sum later in the for loop 
for i, crime in enumerate(focuscrimes_lst): 
    sum_crime = group_count[crime].sum()
    group_count[crime] = group_count[crime]/sum_crime
    
""" To table: """
plt_data = group_count.unstack().T
plt_data['hour'] = plt_data.index + 1


# ======== Bokeh
""" Libraries:  """
from bokeh.models import ColumnDataSource # to trans data to bokeh data 
from bokeh.models import FactorRange #@to-do: need to look into it
from bokeh.plotting import figure # to start a figure inctace 
from bokeh.io import show # to show bokeh figures 
from bokeh.io import output_notebook # to run the output in the notebook not html page
output_notebook() # set notebook output active


""" start plotting: """
source = ColumnDataSource(plt_data) # trans to bokeh data
x = [str(i) for i in plt_data.index] # trans the x's to string (x_range expect list of strings)
p = figure(x_range = FactorRange(factors=x)) #@to-do: need to look into it, more arguments needed 

bar ={} # to store vbars
for i,crime in enumerate(focuscrimes_lst): # start a loop to make vbars:
    bar[crime] = p.vbar(x='hour',  
                    top=crime, 
                    width=0.9,   
                    source= source) # figure.vbar : to make vertical bars
                                    #@to-do: need to look into it, more arguments needed
                                    # legend_label=crime,  muted_alpha=0, muted = 0)


p.legend.click_policy="mute" # assigns the click policy (you can try to use ''hide')
                             #@to-do: need to make it as Sune fig in week 08
show(p) # displays your plot