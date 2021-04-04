# ---
# # Assignment 2.
# ## 02806 Social data analysis and visualization
# ## Apr. 2021
# ---

# ---
# ## Part 3: Data visualization
# ---

# ---
# ## Importing needed libraries
# ---

"""for Data Handling """
import numpy as np
import pandas as pd
import os

""" for Displaying Figures on the Notebook """
from IPython.display import display
#get_ipython().run_line_magic('matplotlib', 'inline')

""" for Interactive Visualization with Bokeh Library """
import bokeh 
from bokeh.models import ColumnDataSource, FactorRange   # to trans data to bokeh data 
from bokeh.plotting import figure  # to start a figure inctace 
from bokeh.palettes import Category20
from bokeh.models import Legend
from bokeh.io import show, output_notebook   # to show bokeh figures and run the output in the notebook not html page 
output_notebook()
 
# ---
# ## Loading and Preprocessing Crime Data
# ---

""" Step1: Add Focus Crimes """
# Import hard-coded set of focus crimes handed from before
focuscrimes = set(['WEAPON LAWS', 'PROSTITUTION', 'DRIVING UNDER THE INFLUENCE','ROBBERY', 'BURGLARY', 'ASSAULT', 'DRUNKENNESS', 'DRUG/NARCOTIC','TRESPASS', 'LARCENY/THEFT', 'VANDALISM', 'VEHICLE THEFT','STOLEN PROPERTY', 'DISORDERLY CONDUCT'])
focuscrimes_lst = list(focuscrimes)
focuscrimes_lst.sort()


""" Step2: Import the Crimes Dataset using os and pandas """
# Import dataset using os
fileName = 'Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv'
filePath = os.path.abspath(os.path.join(os.getcwd(), '..' ,'Datasets', fileName))

# Create raw dataframe from file using specific columns only
df_raw =  pd.read_csv(filePath, usecols=['Category', 'Date', 'Time', 'PdDistrict', 'PdId'])


""" Step3: Preprocess the dataset """
# Add the column of Year to the raw dataframe
df_raw['Year'] = pd.to_datetime(df_raw['Date']).dt.year

# Filter the years so that they are between 2010 up to 2017 into a new dataframes
# (without 2018 since it's not fully covered in the dataset)
df = df_raw[(df_raw['Year'] != 2018) & (df_raw['Year']>=2010)]

# Filter the dataframe to keep only the rows of the focus crimes
df = df[df['Category'].isin(focuscrimes_lst)].copy()

# Add the column of Hour to the dataframe
df['Hour'] = pd.to_datetime(df['Time']).dt.hour

# Clear the final dataset out of nan
df = df.dropna()
df = df.reset_index(drop=True)


""" Step4: Extract Focus Crimes average counts per hour"""
# Get the total crime counts per category
# (float64 to avoid float to int cast when div by sum later in the for loop)
category_count = df.groupby(['Category'])['PdId'].count().astype('float64').copy() # float64 to avoid float to int cast when div by sum later in the for loop

# Get the total instances per hour for each different crime category
per_hour_count = df.groupby(['Category','Hour'])['PdId'].count().astype('float64').copy()  

# Get the normalized crime rate per hour for each different category
plt_data = per_hour_count.unstack().T
for i,crime in enumerate(focuscrimes_lst):
    plt_data[crime] = plt_data[crime]/category_count[crime]

# ---
# ## Interactive Plotting with Bokeh
# ---

# %%
""" Data Preparation for Bokeh (see below for detailed explanation) """
# Transform the data into bokeh readable version
source = ColumnDataSource(plt_data)

# Transform the x-axis values (hours) to string
x = [str(i) for i in plt_data.index] 


""" Set up the figure attributes (see below for detailed explanation) """
p = figure(plot_width=970,
           plot_height=500,
           x_range=FactorRange(factors=x),
           toolbar_location=None,
           tools='',
           title='Crimes per Hour',
           x_axis_label='Hours of the Day',
           y_axis_label='Relative Frequency')


""" Generate the Interactive Bar Plot (see below for detailed explanation) """
# Prepare the color set for the plot
colors = Category20[len(focuscrimes_lst)]

# Initialize the structures for vbars and legend items
bar = {}
legend_items = []

# Create vbars and legend entries for each of the focus crimes
for i,crime in enumerate(focuscrimes_lst):
    bar[crime] = p.vbar(x='Hour',  
                        top=crime,
                        source= source,
                        fill_color=colors[i],
                        line_color=colors[i],
                        alpha=0.62,
                        muted_alpha=0.03,
                        muted=True,
                        width=0.75)
    legend_items.append((crime, [bar[crime]]))
    
# Create a direct legend object in order to make it appear outside the figure and at the wanted position
legend = Legend(items=legend_items, location="top_left")
p.add_layout(legend, 'left')

# Assign a click policy to the legend entries
p.legend.click_policy="mute"

# Show the figure
show(p)

# ---
# ### Summary of Bokeh Library use
# #### Part1: Importing Needed Libraries
# Below is the list of imported Bokeh functions along with their descriptions:
# 1. `bokeh.models.ColumnDataSource`: Is used to transform python dictionary or pandas data into bokeh readable data format, more details [here](https://docs.bokeh.org/en/latest/docs/reference/models/sources.html#bokeh.models.sources.ColumnDataSource).
# ---
# 2. `bokeh.models.FactorRange`: Turns a categorical variable (i.e. in our case the whole-hour strings) into a range, more details [here](https://docs.bokeh.org/en/latest/docs/reference/models/ranges.html#bokeh.models.ranges.FactorRange).
# ---
# 3. `bokeh.plotting.figure`: Creates a new Bokeh figure instance for plotting inside, more details [here](https://docs.bokeh.org/en/latest/docs/reference/plotting.html).
# ---
# 4. `bokeh.palettes.Category20`: A [D3js](https://github.com/d3/d3-3.x-api-reference/blob/master/Ordinal-Scales.md#categorical-colors) categorical color palette containing up to a total of 20 colors, more details [here](https://docs.bokeh.org/en/latest/docs/reference/palettes.html#d3-palettes).
# ---
# 5. `bokeh.models.Legend`: Creates a new separate Bokeh figure legend render instance, more details [here](https://docs.bokeh.org/en/latest/docs/reference/models/annotations.html#bokeh.models.annotations.Legend).
# ---
# 6. `bokeh.io.show`: Shows a Bokeh figure, more details [here](https://docs.bokeh.org/en/latest/docs/reference/io.html#bokeh.io.show).
# ---
# 7. `bokeh.io.output_notebook`: Edits the default output state for generating figures in notebook cells when `show()` is called, more details [here](https://docs.bokeh.org/en/latest/docs/reference/io.html#bokeh.io.output_notebook).
# 
# ---
# #### Part 2: Interactive Plotting with Bokeh
# Susbteps followed in the **Data Preparation** step:
# 1. Using the `ColumnDataSource()` what would be the main data source was transformed into Bokeh readable version.
# 2. The x-axis values, which are the whole-hours 0,1,...,23, were switched to strings '0','1',...,'23' as the `FactorRange()` accepts categorical input factors.
# ---
# Parameters used for the **Figure Setup**:
# 1. `plot_width`: sets the plot width.
# 2. `plot_height`: sets the plot height.
# 3. `x_range`: sets the data bounds for x-axis, more details [here](https://docs.bokeh.org/en/latest/docs/user_guide/plotting.html#setting-ranges).
# 4. `toolbar_location`: 
# 5. `tools`: 
# 6. `title`: sets the title of the figure.
# 7. `x_axis_label`: sets the label for the x-axis.
# 8. `y_axis_label`: sets the label for the y-axis.
# ---
# Substeps followed in the **Bar Plot Generation** step:
# 1. The color range for each one of the bar plots is specified. Using `Category20()` and specifying the length of the focus crimes list for picking the appropriate number of colours (in our case 14 out of the total 20).
# 2. Initialize the `bar` dictionary for the vertical bar charts and the `legend_items` list for the legend entries to be filled using the for loop below.
# 3. Using a for loop fill in `bar` and `legend_items` for each of the different focus crime categories. The vertical bar charts are created using the function `bokeh.plotting.figure().vbar()`, using the following parameters:
#     1. `x`: specifies the x-axis parameter inputs, in our case since the input dataset is given by a *Pandas Dataframe* we just specify the name of the column used which is 'Hour'.
#     2. `top`: specifies the y-axis (vertical) parameter inputs, in a similar manner the name of the column for each of the crimes is given at each iteration.
#     3. `source`: specifies the source of the dataset out of which the data will be dragged (this has been initialized before using `ColumnDataSource()`).
#     4. `fill_color`: specifies the fill colour of the vertical bars.
#     5. `line_color`: specifies the outer line colour of each of the vertical bars.
#     6. `alpha`: sets the opacity of the bars, a value in \[0,1\].
#     7. `muted_alpha`: seths the opacity of the bars that are muted in the figure, a value in \[0,1\].
#     8. `muted`: specifies whether the initial state of all vertical bars will be muted.
#     9. `width`: specifies the width of the bars relative to the x-axis parameter points.
# 4. A Legend object is directly created using the `legend_items` filled in from the for loop above and using the `bokeh.plotting.figure().add_layout()` command the legend is positioned outside the figure plot area.
# 5. A click policy is assigned to the legend entries using `bokeh.plotting.figure().legend.click_policy` in our case turning them to switches for controlling the muted state of the plots per crime.
# 6. Finally the figure is shown using `bokeh.io.show()`.




