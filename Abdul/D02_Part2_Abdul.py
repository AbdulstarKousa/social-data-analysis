# %% [markdown]
"""
# [Reading Materials](https://www.sciencemag.org/news/2016/09/can-predictive-policing-prevent-crime-it-happens)
# Libraries and Dataset:
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the data as pandas DataFrame:
data_dir = r"C:\Users\boody\OneDrive - Danmarks Tekniske Universitet\01 MSc\2st Semester\01_SD\Lectures\01\Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv"
Data = pd.read_csv(data_dir) 

# Take a look at the data head 
Data.head(n=2)



# %% [markdown]
"""
# Report the total number of crimes in the dataset:
"""
# %%
Data['Date'].count()



# %% [markdown]
"""
# List the various categories of crime:
"""
# %%
Data['Category'].unique()



# %% [markdown]
"""
# How many crime category are there?
"""
# %%
Data['Category'].nunique()



# %% [markdown]
"""
# List the number of crimes in each category:
"""
# %%
Data['Category'].value_counts(ascending=False)



# %% [markdown]
"""
# Create a histogram over crime occurrences:
"""
# %%
Data['Category'].value_counts(ascending=False).plot(kind='bar', figsize =(8,8))



# %% [markdown]
"""
# Count the number of crimes per year for the years 2003-2017 (since we don't have full data for 2018). What's the average number of crimes per year?
"""
# %%
Data['Year'] = pd.to_datetime(Data['Date']).dt.year
Data = Data[Data['Year'] != 2018]

# %% [markdown]
"""
### Number of crimes per years from 2003 to 2018 (not included):
"""
# %%
Data['Year'].value_counts(ascending=True)

# %% [markdown]
"""
### Average number of crimes from 2003 up to 2018 (not included):
"""
# %%
Data['Year'].value_counts(ascending=True).mean()



# %% [markdown]
"""
### Police chief Suneman is interested in the temporal development of only a subset of categories, the so-called focus crimes. Those categories are listed below (for convenient copy-paste action).
"""
# %%
focuscrimes = set(['WEAPON LAWS', 'PROSTITUTION', 'DRIVING UNDER THE INFLUENCE', 'ROBBERY', 'BURGLARY', 'ASSAULT', 'DRUNKENNESS', 'DRUG/NARCOTIC', 'TRESPASS', 'LARCENY/THEFT', 'VANDALISM', 'VEHICLE THEFT', 'STOLEN PROPERTY', 'DISORDERLY CONDUCT'])
data = Data[Data['Category'].isin(list(focuscrimes))]

# %% [markdown]
"""
# Now create bar-charts displaying the year-by-year development of each of these categories across the years 2003-2017.
"""
# %% [markdown]
"""
### Example of single plot for WEAPON LAWS
"""
# %%
data.groupby(['Category','Year']).count()['PdId'] ['WEAPON LAWS'] .plot(kind='bar', figsize=(9,5))




# %% [markdown]
"""
### Plots for year-by-year development for all Focus Crimes
"""
# %%
focuscrimes_lst = [ 'WEAPON LAWS', 'DRUNKENNESS',
                    'TRESPASS','PROSTITUTION',
                    'DRIVING UNDER THE INFLUENCE','BURGLARY',
                    'ROBBERY','DRUG/NARCOTIC',
                    'LARCENY/THEFT','DISORDERLY CONDUCT',
                    'VANDALISM', 'VEHICLE THEFT',
                    'ASSAULT', 'STOLEN PROPERTY']
fig, axs = plt.subplots(7, 2,figsize=(15, 20), sharex=True)
fig.suptitle('Development of Focus Crimes over Years', fontsize=20)
for i,ax in enumerate(axs.flat):
    ax.set(title=focuscrimes_lst[i])
    data.groupby(['Category','Year']).count()['PdId'] [focuscrimes_lst[i]] .plot(kind='bar',ax=ax)



# %% [markdown]
"""
# Comment on at least three interesting trends in your plot.
Also, here's a fun fact: 
The drop in car thefts is due to new technology called 'engine immobilizer systems' 
get the full story [here](https://www.nytimes.com/2014/08/12/upshot/heres-why-stealing-cars-went-out-of-fashion.html):

* VECHICLE THEFT:
    Huge decrease in occurrences after 2005 when the engine immobilizer systam
    was introduced. Take a look [here](https://www.nytimes.com/2014/08/12/upshot/heres-why-stealing-cars-went-out-of-fashion.html).
   
* ASSAULT:

* STOLEN PROPERTY:
"""



# %% [markdown]
"""
####################################
# Week 02, Part 2.1:
####################################
"""

# %% [markdown]
"""
# Weekly patterns. 
Basically, we'll forget about the yearly variation 
and just count up what happens during each weekday.
Here's a [Link](https://raw.githubusercontent.com/suneman/socialdata2021/master/files/weekdays.png) of what my version looks like. 
Some things make sense - for example drunkenness and the weekend. 
But there are some aspects that were surprising to me. 
Check out prostitution and mid-week behavior, for example!?
"""



# %% [markdown]
"""
**Example: WEAPON LAWS**
"""
# %%
(
data.groupby(['Category','DayOfWeek']).count()
['PdId'] 
['WEAPON LAWS']
.plot(kind='bar',color= 'tab:gray' , figsize=(5,5))
)



# %% [markdown]
"""
**First** Transfer 'DayOfWeek' from Categorical to ordered Categorical variable 
"""
# %%
cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
data['DayOfWeek'] = pd.Categorical(data['DayOfWeek'], categories=cats, ordered=True)

# %% [markdown]
"""
**Then** order the focuscrimes_lst to match the given figures order
"""
# %%
focuscrimes_lst.sort()

# %% [markdown]
"""
**Finally** start plotting 
"""
# %%
fig, axs = plt.subplots(7, 2,figsize=(15, 20), sharex=True)
fig.suptitle('Development of Focus Crimes over Days of the week', fontsize=20)
for i,ax in enumerate(axs.flat):
    ax.set(title=focuscrimes_lst[i])
    (
    data.groupby(['Category','DayOfWeek']).count()
    ['PdId']
    [focuscrimes_lst[i]]
    .plot(kind='bar',ax=ax, color= 'tab:gray')
    )



# %% [markdown]
"""
# Development of Focus Crimes over Months
We can also check if some months
are worse by counting up number
of crimes in Jan, Feb, ..., Dec. 
Did you see any surprises there?
"""
# %% [markdown]
"""
**First** Add a Month column to data 
"""
# %%
data['Month'] = pd.to_datetime(data['Date']).dt.month

# %% [markdown]
"""
**Then** Transfer 'Month' to ordered Categorical variable 
"""
# %%
import calendar
data['Month'] = data['Month'].apply(lambda x: calendar.month_abbr[x])
cats = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',  'Jul', 'Aug', 'Sep', 'Oct', 'Nov','Dec',]
data['Month'] = pd.Categorical(data['Month'], categories=cats, ordered=True)

# %%[markdown]
"""
**Finally** start plotting: 
"""
# %%
fig, axs = plt.subplots(7, 2,figsize=(15, 20), sharex=True)
fig.suptitle('Development of Focus Crimes over Months', fontsize=20)
for i,ax in enumerate(axs.flat):
    ax.set(title=focuscrimes_lst[i])
    (
    data.groupby(['Category','Month']).count()
    ['PdId']
    [focuscrimes_lst[i]]
    .plot(kind='bar',ax=ax, color= 'tab:gray')
    )

# %%[markdown]
"""
### check if some months are worse.  
write here  
write here  
write here  
write here  
"""
# %%[MarkDown]
"""
### Did you see any surprises there?
write here  
write here  
write here  
write here  
"""



# %%[markdown]
"""
# The 24 hour cycle.
We'll can also forget about weekday and simply count up
the number of each crime-type that occurs 
in the entire dataset from midnight to 1am,
1am - 2am ... and so on. 
"""

# %%
data['Hour'] = pd.to_datetime(data['Time']).dt.hour
data['Hour'] = data['Hour'].apply(lambda x: str(x)+"-"+str(x+1))
cats = ['0-1','1-2','2-3','3-4','4-5','5-6','6-7','7-8','8-9','9-10','10-11','11-12','12-13',
        '13-14','14-15','15-16','16-17','17-18','18-19','19-20','20-21','21-22','22-23','23-24']
data['Hour'] = pd.Categorical(data['Hour'], categories=cats, ordered=True)

fig, axs = plt.subplots(7, 2,figsize=(15, 20), sharex=True)
fig.suptitle('Development of Focus Crimes over The 24 hour cycle', fontsize=20)
for i,ax in enumerate(axs.flat):
    ax.set(title=focuscrimes_lst[i])
    (
    data.groupby(['Category','Hour']).count()
    ['PdId']
    [focuscrimes_lst[i]]
    .plot(kind='bar',ax=ax, color= 'tab:gray')
    )

# %%[MarkDown]
"""
### Give me a couple of comments on what you see
write here  
write here  
write here  
write here  
"""



# %%[markdown]
"""
# This Question takes a long running time (uncomment below if needed)
# Hours of the week.
But by looking at just 24 hours,
we may be missing some important
trends that can be modulated by week-day,
so let's also check out the 168 hours of the week.
So let's see the number of each crime-type
Monday night from midninght to 1am,
Monday night from 1am-2am -
all the way to Sunday night from 11pm to midnight
"""

'''
# %%
def day_to_num(x):
    if(x=='Monday'): return 1
    elif(x=='Tuesday'): return 2
    elif(x=='Wednesday'): return 3
    elif(x=='Thursday'): return 4
    elif(x=='Friday'): return 5
    elif(x=='Saturday'): return 6
    elif(x=='Sunday'): return 7
    else: return -1

data['DayOfWeek_Numeric'] =data['DayOfWeek'].apply(lambda x : day_to_num(x))
data['Hour']      = pd.to_datetime(data['Time']).dt.hour
data['HourOfWeek'] = data.apply(lambda row: row.Hour + ((row.DayOfWeek_Numeric-1)*24), axis=1)

# %%[markdown]
"""
### Example:
"""
# %%
(
data.groupby(['Category','HourOfWeek']).count()
['PdId'] 
['WEAPON LAWS']
.plot(figsize=(5,5))
)


# %%
# %%[markdown]
"""
# Hours of the week.
"""
# %%
fig, axs = plt.subplots(7, 2,figsize=(15, 20), sharex=True)
fig.suptitle('Development of Focus Crimes over Week hours', fontsize=20)
for i,ax in enumerate(axs.flat):
    ax.set(title=focuscrimes_lst[i])
    (
    data.groupby(['Category','HourOfWeek']).count()
    ['PdId']
    [focuscrimes_lst[i]]
    .plot(ax=ax)
    )
'''







# %% [markdown]
"""
####################################
# Week 02, Part 2.2:
####################################
"""

# %% [markdown]
"""
### Clear some variables
"""
# %%
del(ax);del(axs);del(cats);del(data_dir);del(fig);del(focuscrimes);del(i)



# %%

# First, simply list the names of SF's 10 police districts.
[x for x in data['PdDistrict'].unique() if (isinstance(x, str)) ]

# Which has the most crimes? 
Data['PdDistrict'].value_counts(ascending=False).index[0]

# Which has the most focus crimes?
data['PdDistrict'].value_counts(ascending=False).index[0]

# First, 
# we need to calculate the relative probabilities of seeing
# each type of crime in the dataset as a whole.
# That's simply a normalized version of this plot. 
# https://raw.githubusercontent.com/suneman/socialdata2021/master/files/categoryhist.png
# Let's call it P(crime).

plt.figure(figsize=(8, 5))
x_bar = Data['Category'].value_counts(ascending=True).index
y_bar = Data['Category'].value_counts(ascending=True).values/Data['Date'].count()
plt.bar(x_bar,y_bar,)
plt.xticks(rotation='vertical')
plt.title("$P(crime)$")


# Next, 
# we calculate that same probability distribution but for each PD district,
# let's call that P(crime|district).
PD_district = [x for x in data['PdDistrict'].unique() if (isinstance(x, str)) ]
PD_district.sort()



""" Example : BAYVIEW """ 
group   = Data.groupby(['PdDistrict','Category'])['PdId'].count()
ser_sub  = group['BAYVIEW'] ### here where you iterate ###
ser_sub  = ser_sub.divide(ser_sub.sum())
ser_sub.plot(kind='bar',color= 'tab:gray' , figsize=(8,5))



""" Plot """
group = Data.groupby(['PdDistrict','Category'])['PdId'].count()

fig, axs = plt.subplots(5, 2,figsize=(15, 20), sharex=True)
fig.suptitle('$P(crime|district)$', fontsize=20)

for i,ax in enumerate(axs.flat):    
    ser_sub  = group[PD_district[i]]   
    ser_sub  = ser_sub.divide(ser_sub.sum())

    ax.set(title= "$P(crime|"+PD_district[i]+"$)")
    ser_sub.plot(ax=ax, kind='bar',color= 'tab:gray')



# Now
# we look at the ratio P(crime|district)/P(crime).
# That ratio is equal to
# 1 if the crime occurs at the same level within a district
# as in the city as a whole.
# If it's greater than one,
# it means that the crime occurs more frequently within that district.
# If it's smaller than one,
# it means that the crime is rarer within the district
# in question than in the city as a whole.

# For each district 
# plot these ratios for 
# the 14 focus crimes. 
# see plots on the homepage for reference


focus_crimes   = data['Category'].value_counts(ascending=True).sort_index()
p_focus_crimes = focus_crimes.divide(data['PdId'].count())


""" Example : BAYVIEW """ 
group   = data.groupby(['PdDistrict','Category'])['PdId'].count()
ser_sub  = group['BAYVIEW'] ### here where you iterate ###
ser_sub  = ser_sub.divide(ser_sub.sum())
ser_sub  = ser_sub.divide(p_focus_crimes)
ser_sub.plot(kind='bar',color= 'tab:gray' , figsize=(8,5))
 



""" Plot """
group = data.groupby(['PdDistrict','Category'])['PdId'].count()

fig, axs = plt.subplots(5, 2,figsize=(15, 20), sharex=True)
fig.suptitle('$P(focus_crime|district)$', fontsize=20)

for i,ax in enumerate(axs.flat):    
    ser_sub  = group[PD_district[i]]   
    ser_sub  = ser_sub.divide(ser_sub.sum())
    ser_sub  = ser_sub.divide(p_focus_crimes)

    ax.set(title= "$P(focus_{crime}|"+PD_district[i]+"$)")
    ser_sub.plot(ax=ax, kind='bar',color= 'tab:blue')



# Comment on the top crimes in Tenderloin,
# Mission, and Richmond. 
# Does this fit with the impression you get of
# these neighborhoods on Wikipedia?
"""
Write here
Write here

Write here
Write here
"""


# Comment. 
# Notice how much awesome datascience 
# (i.e. learning about interesting real-world crime patterns)
# we can get out by simply counting and plotting 
# (and looking at ratios). Pretty great, right?