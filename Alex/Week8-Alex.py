import numpy as np
import pandas as pd
import os 
from IPython.display import display
from IPython import get_ipython
import pdb
get_ipython().run_line_magic('matplotlib', 'inline')

from bokeh.models import ColumnDataSource # to trans data to bokeh data 
from bokeh.models import FactorRange #@to-do: need to look into it
from bokeh.plotting import figure # to start a figure inctace 
from bokeh.io import show # to show bokeh figures 
from bokeh.io import output_notebook # to run the output in the notebook not html page
from bokeh.palettes import Category20
from bokeh.models import Legend
output_notebook() # set notebook output active
    

fileName = 'Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv'
filePath = os.path.abspath(os.path.join(os.getcwd(), '..' ,'Datasets', fileName))

focuscrimes = set(['WEAPON LAWS', 'PROSTITUTION', 'DRIVING UNDER THE INFLUENCE',
                   'ROBBERY', 'BURGLARY', 'ASSAULT', 'DRUNKENNESS', 'DRUG/NARCOTIC',
                   'TRESPASS', 'LARCENY/THEFT', 'VANDALISM', 'VEHICLE THEFT',
                   'STOLEN PROPERTY', 'DISORDERLY CONDUCT'])
focuscrimes_lst = list(focuscrimes)
focuscrimes_lst.sort()

df_raw =  pd.read_csv(filePath, usecols=['Category', 'Date', 'Time', 'PdDistrict', 'PdId'])

df_raw['Year'] = pd.to_datetime(df_raw['Date']).dt.year
df = df_raw[(df_raw['Year'] != 2018) & (df_raw['Year']>=2010)]
df = df[df['Category'].isin(focuscrimes_lst)].copy()

df['Hour'] = pd.to_datetime(df['Time']).dt.hour

df = df.dropna()
df = df.reset_index(drop=True)

category_count = df.groupby(['Category'])['PdId'].count().astype('float64').copy() # float64 to avoid float to int cast when div by sum later in the for loop
per_hour_count = df.groupby(['Category','Hour'])['PdId'].count().astype('float64').copy()  

plt_data = per_hour_count.unstack().T
for i,crime in enumerate(focuscrimes_lst):
    plt_data[crime] = plt_data[crime]/category_count[crime]

plt_data['hour'] = plt_data.index + 1

source = ColumnDataSource(plt_data) # trans to bokeh data
x = [str(i) for i in plt_data.index] # trans the x's to string (x_range expect list of strings)

p = figure(plot_width=800,
           x_range=FactorRange(factors=x),
           toolbar_location=None,
           title='Crimes per Hour')

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