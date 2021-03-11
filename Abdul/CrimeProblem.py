####################################
# Week 01:
####################################

# [Reading Materials](https://www.sciencemag.org/news/2016/09/can-predictive-policing-prevent-crime-it-happens)

# import main libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import folium       # for map

# Loading the data as pandas DataFrame:
""" Note if you working from Git then consider setting Local to False below """  
Local = False # False if from git
if Local : # Access data locally from Abdul machine 
    data_dir = r"C:\Users\boody\OneDrive - Danmarks Tekniske Universitet\01 MSc\2st Semester\01_SD\GitHub_Group\social-data-analysis\Datasets\Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv"
    Data = pd.read_csv(data_dir) 
else:   # Access data from Git Repo 
        # Note! If you change the path of this script on git repo you need to change the data_dir below
    import os
    fileName = "Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv"
    data_dir = os.path.abspath(os.path.join(os.getcwd(),'..','Datasets', fileName))
    Data = pd.read_csv(data_dir) 


# Take a look at the data head 
Data.head(n=2)
 
# Report the total number of crimes in the dataset:
Data['Date'].count()

# List the various categories of crime:
Data['Category'].unique()

# How many crime category are there?
Data['Category'].nunique()

# List the number of crimes in each category:
Data['Category'].value_counts(ascending=False)

# Create a histogram over crime occurrences:
Data['Category'].value_counts(ascending=False).plot(kind='bar', figsize =(8,8))

# Count the number of crimes per year for the years 2003-2017 (since we don't have full data for 2018). What's the average number of crimes per year?
Data['Year'] = pd.to_datetime(Data['Date']).dt.year
Data = Data[Data['Year'] != 2018]

# Number of crimes per years from 2003 to 2018 (not included):
Data['Year'].value_counts(ascending=True)

# Average number of crimes from 2003 up to 2018 (not included):
Data['Year'].value_counts(ascending=True).mean()

# Police chief Suneman is interested in the temporal development of only a subset of categories,
# the so-called focus crimes. Those categories are listed below (for convenient copy-paste action).
focuscrimes = set(['WEAPON LAWS', 'PROSTITUTION', 'DRIVING UNDER THE INFLUENCE', 'ROBBERY', 'BURGLARY', 'ASSAULT', 'DRUNKENNESS', 'DRUG/NARCOTIC', 'TRESPASS', 'LARCENY/THEFT', 'VANDALISM', 'VEHICLE THEFT', 'STOLEN PROPERTY', 'DISORDERLY CONDUCT'])
data = Data[Data['Category'].isin(list(focuscrimes))]

# Now create bar-charts displaying the year-by-year development of each of these categories across the years 2003-2017.
""" Example of single plot for WEAPON LAWS"""
data.groupby(['Category','Year']).count()['PdId'] ['WEAPON LAWS'] .plot(kind='bar', figsize=(9,5))

""" Plots for year-by-year development for all Focus Crimes"""
focuscrimes_lst = [ 'WEAPON LAWS', 'DRUNKENNESS',
                    'TRESPASS','PROSTITUTION',
                    'DRIVING UNDER THE INFLUENCE','BURGLARY',
                    'ROBBERY','DRUG/NARCOTIC',
                    'LARCENY/THEFT','DISORDERLY CONDUCT',
                    'VANDALISM', 'VEHICLE THEFT',
                    'ASSAULT', 'STOLEN PROPERTY']
fig, axs = plt.subplots(7, 2,figsize=(15, 20), sharex=True)
fig.suptitle('Development of Focus Crimes over Years', fontsize=20)
group_count = data.groupby(['Category','Year'])['PdId'].count()
for i,ax in enumerate(axs.flat):
    ax.set(title=focuscrimes_lst[i])
    group_count[focuscrimes_lst[i]] .plot(kind='bar',ax=ax)


# Comment on at least three interesting trends in your plot.
# Also, here's a fun fact: 
# The drop in car thefts is due to new technology called 'engine immobilizer systems' 
# get the full story [here](https://www.nytimes.com/2014/08/12/upshot/heres-why-stealing-cars-went-out-of-fashion.html):
"""
* VECHICLE THEFT:
    Huge decrease in occurrences after 2005 when the engine immobilizer systam
    was introduced. Take a look [here](https://www.nytimes.com/2014/08/12/upshot/heres-why-stealing-cars-went-out-of-fashion.html).
   
* ASSAULT:

* STOLEN PROPERTY:
"""



####################################
# Week 02, Part 2.1:
####################################


# Weekly patterns. 
# Basically, we'll forget about the yearly variation 
# and just count up what happens during each weekday.
# Here's a [Link](https://raw.githubusercontent.com/suneman/socialdata2021/master/files/weekdays.png) of what my version looks like. 
# Some things make sense - for example drunkenness and the weekend. 
# But there are some aspects that were surprising to me. 
# Check out prostitution and mid-week behavior, for example!?



"""Example: WEAPON LAWS**"""
# %%
(
data.groupby(['Category','DayOfWeek'])['PdId'].count() 
['WEAPON LAWS']
.plot(kind='bar',color= 'tab:gray' , figsize=(5,5))
)



""" **First** Transfer 'DayOfWeek' from Categorical to ordered Categorical variable """
cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
data['DayOfWeek'] = pd.Categorical(data['DayOfWeek'], categories=cats, ordered=True)

""" **Then** order the focuscrimes_lst to match the given figures order """
focuscrimes_lst.sort()

""" **Finally** start plotting """
fig, axs = plt.subplots(7, 2,figsize=(15, 20), sharex=True)
fig.suptitle('Development of Focus Crimes over Days of the week', fontsize=20)
group_count = data.groupby(['Category','DayOfWeek'])['PdId'].count()
for i,ax in enumerate(axs.flat):
    ax.set(title=focuscrimes_lst[i])
    group_count[focuscrimes_lst[i]].plot(kind='bar',ax=ax, color= 'tab:gray')


# Development of Focus Crimes over Months
# We can also check if some months
# are worse by counting up number
# of crimes in Jan, Feb, ..., Dec. 
# Did you see any surprises there?

""" **First** Add a Month column to data """
data['Month_Numeric'] = pd.to_datetime(data['Date']).dt.month

""" **Then** Transfer 'Month_Numeric' to ordered Categorical variable """
import calendar
data['Month'] = data['Month_Numeric'].apply(lambda x: calendar.month_abbr[x])
cats = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',  'Jul', 'Aug', 'Sep', 'Oct', 'Nov','Dec',]
data['Month'] = pd.Categorical(data['Month'], categories=cats, ordered=True)

""" **Finally** start plotting: """
fig, axs = plt.subplots(7, 2,figsize=(15, 20), sharex=True)
fig.suptitle('Development of Focus Crimes over Months', fontsize=20)
group_count = data.groupby(['Category','Month'])['PdId'].count()
for i,ax in enumerate(axs.flat):
    ax.set(title=focuscrimes_lst[i])
    group_count[focuscrimes_lst[i]].plot(kind='bar',ax=ax, color= 'tab:gray')
    

# check if some months are worse.  
"""
write here  
write here  
write here  
write here  
"""

# Did you see any surprises there?
"""
write here  
write here  
write here  
write here  
"""

# The 24 hour cycle.
# We'll can also forget about weekday and simply count up
# the number of each crime-type that occurs 
# in the entire dataset from midnight to 1am,
# 1am - 2am ... and so on. 

data['Hour'] = pd.to_datetime(data['Time']).dt.hour
data['Hour'] = data['Hour'].apply(lambda x: str(x)+"-"+str(x+1))
cats = ['0-1','1-2','2-3','3-4','4-5','5-6','6-7','7-8','8-9','9-10','10-11','11-12','12-13',
        '13-14','14-15','15-16','16-17','17-18','18-19','19-20','20-21','21-22','22-23','23-24']
data['Hour'] = pd.Categorical(data['Hour'], categories=cats, ordered=True)

fig, axs = plt.subplots(7, 2,figsize=(15, 20), sharex=True)
fig.suptitle('Development of Focus Crimes over The 24 hour cycle', fontsize=20)
group_count = data.groupby(['Category','Hour'])['PdId'].count()
for i,ax in enumerate(axs.flat):
    ax.set(title=focuscrimes_lst[i])
    group_count[focuscrimes_lst[i]].plot(kind='bar',ax=ax, color= 'tab:gray')
    
# Give me a couple of comments on what you see
"""
write here  
write here  
write here  
write here  
"""

# Hours of the week.
# But by looking at just 24 hours,
# we may be missing some important
# trends that can be modulated by week-day,
# so let's also check out the 168 hours of the week.
# So let's see the number of each crime-type
# Monday night from midninght to 1am,
# Monday night from 1am-2am -
# all the way to Sunday night from 11pm to midnight

def day_to_num(x):
    if(x=='Monday'): return 1
    elif(x=='Tuesday'): return 2
    elif(x=='Wednesday'): return 3
    elif(x=='Thursday'): return 4
    elif(x=='Friday'): return 5
    elif(x=='Saturday'): return 6
    elif(x=='Sunday'): return 7
    else: return -1

data['DayOfWeek_Numeric']   = data['DayOfWeek'].apply(lambda x : day_to_num(x)).astype('int64')
data['Hour']                = pd.to_datetime(data['Time']).dt.hour
h_v   = data['Hour'].values 
d_v   = data['DayOfWeek_Numeric'].values
data['HourOfWeek'] =  (h_v + ((d_v-1)*24) )


""" Example:"""
(
data.groupby(['Category','HourOfWeek'])['PdId'].count()
['WEAPON LAWS']
.plot(figsize=(5,5))
)

# Hours of the week.
# %%
fig, axs = plt.subplots(7, 2,figsize=(15, 20), sharex=True)
fig.suptitle('Development of Focus Crimes over Week hours', fontsize=20)
group_count = data.groupby(['Category','HourOfWeek'])['PdId'].count()
for i,ax in enumerate(axs.flat):
    ax.set(title=focuscrimes_lst[i])
    group_count[focuscrimes_lst[i]].plot(ax=ax)


# %% [markdown]
####################################
# Week 02, Part 2.2:
####################################


# First, simply list the names of SF's 10 police districts.
PD_district_lst = [x for x in data['PdDistrict'].unique() if (isinstance(x, str)) ]
PD_district_lst.sort()
PD_district_lst

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

""" Example : BAYVIEW """ 
grouping   = Data.groupby(['PdDistrict','Category'])['PdId'].count()
sub_group  = grouping['BAYVIEW'] ### here where you iterate ###
sub_group_nor  = sub_group.divide(sub_group.sum())
sub_group_nor.plot(kind='bar',color= 'tab:gray' , figsize=(8,5))



""" Plot """
grouping = Data.groupby(['PdDistrict','Category'])['PdId'].count()

fig, axs = plt.subplots(5, 2,figsize=(15, 20), sharex=True)
fig.suptitle('$P(crime|district)$', fontsize=20)

for i,ax in enumerate(axs.flat):    
    sub_group  = grouping[PD_district_lst[i]]   
    sub_group_nor  = sub_group.divide(sub_group.sum())

    ax.set(title= "$P(crime|"+PD_district_lst[i]+"$)")
    sub_group_nor.plot(ax=ax, kind='bar',color= 'tab:gray')



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


focus_crimes_nr     = data['Category'].value_counts(ascending=True).sort_index()
p_focus_crimes      = focus_crimes_nr.divide(data['PdId'].count())


""" Example : BAYVIEW """ 
grouping        = data.groupby(['PdDistrict','Category'])['PdId'].count()
sub_group       = grouping['BAYVIEW'] ### here where you iterate ###
sub_group_nor   = sub_group.divide(sub_group.sum())
ser_sub_ra      = sub_group_nor.divide(p_focus_crimes)
ser_sub_ra.plot(kind='bar',color= 'tab:blue' , figsize=(8,5))
 



""" Plot """
grouping = data.groupby(['PdDistrict','Category'])['PdId'].count()

fig, axs = plt.subplots(5, 2,figsize=(15, 20), sharex=True)
fig.suptitle('$P(focus_{crime}|district)$', fontsize=20)

for i,ax in enumerate(axs.flat):    
    sub_group       = grouping[PD_district_lst[i]]   
    sub_group_nor   = sub_group.divide(sub_group.sum())
    ser_sub_ra      = sub_group_nor.divide(p_focus_crimes)

    ax.set(title= "$P(focus_{crime}|"+PD_district_lst[i]+"$)")
    ser_sub_ra.plot(ax=ax, kind='bar',color= 'tab:blue')



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






####################################
# Week 03, Part 3:
####################################

# Let's make a jitter-plot 
# (that is, code up something like Figure 2-1 from DAOST from scratch),
# but based on SF Police data. 
# My hunch from inspecting the file is that the police-folks
# might be a little bit lazy in noting down the exact time down to the second.

# So choose a crime-type and a suitable time interval 
# (somewhere between a month and 6 months depending on the crime-type) 
# and create a jitter plot of the arrest times during a single hour 
# (like 13-14, for example). 
# So let time run on the x-axis and create vertical jitter.


"""
Crime = ROBBERY # imperically: gives one of best figures from Focus Crimes
Month = from Jan to Jul. 
Hour  = 13 - 14  
Time -> Minutes 

Data  = focus crimes & != 2018
"""
data_Jit = data[data['Category']=='ROBBERY'] 
data_Jit = data_Jit[data_Jit['Month_Numeric']<=6]
data_Jit = data_Jit[data_Jit['Hour']==13].reset_index(drop=True)
data_Jit['Time'] = pd.to_datetime(data_Jit['Time']).dt.minute

import seaborn as sns # for jitter plot
plt.figure(figsize=(7, 7))
sns.stripplot(data_Jit['Time'].values, jitter=True, edgecolor='none', alpha=.50 ,color='k').set_title('Robbery\nJan to Jul\n13:00-14:00')
plt.show()


# Last time, we did lots bar-plots.
# Today, we'll play around with histograms 
# (creating two crime-data based versions 
#  of the plot-type shown in DAOST Figure 2-2). 
# I think the GPS data could be fun to see this way.

# This time, pick two crime-types 
# with different geographical patterns
# and a suitable time-interval for each 
# (you want between 1000 and 10000 points in your histogram)

# Then take the latitude part of the GPS coordinates
# for each crime and bin the latitudes so that 
# you have around 50 bins across the city of SF. 
# You can use your favorite method for binning. 
# I like numpy.histogram. 
# This function gives you the counts 
# and then you do your own plotting.



"""
Histogram 01
Crime = ROBBERY
Month = Jan 

Data  = focus crimes & != 2018
"""
data ['Y']
data_hist1 = data[data['Category'] == 'ROBBERY']
data_hist1 = data_hist1[data_hist1['Month_Numeric']==1]
data_hist1.shape[0]
data_hist1['Y']

plt.figure(figsize=(6, 6))
plt.hist(data_hist1['Y'], bins=50) # uses numpy.histogram
plt.title("Robbery in January")
plt.xlabel("ROBBERY")
plt.ylabel("Number of Observations")
plt.show()                                                       


"""
Histogram 01
Crime = VANDALISM 
Month = Jan 

Data  = focus crimes & != 2018
"""
data_hist2 = data[data['Category'] == 'VANDALISM']
data_hist2 = data_hist2[data_hist2['Month_Numeric']==1]
data_hist2.shape[0]
data_hist2['Y']

plt.figure(figsize=(6, 6))
plt.hist(data_hist2['Y'], bins=50) # uses numpy.histogram
plt.title("Vandalisim in January")
plt.xlabel("Vandalisim")
plt.ylabel("Number of Observations")
plt.show()                                                       




####################################
# Week 04 part 01
####################################

"""
see more here: https://www.kaggle.com/daveianhickey/how-to-folium-for-maps-heatmaps-time-data
"""


# > First start by plotting a map of San Francisco with a nice tight zoom.
#  Simply use the command `folium.Map([lat, lon], zoom_start=13)`,
#  where you'll have to look up San Francisco's longitude and latitude.

""" Create New Map Instance"""
mapSF1 = folium.Map(
    location = [37.7749, -122.4194],
    tiles = 'Stamen Toner',
    zoom_start = 13)
# Display Map
mapSF1



# > Next, use the the coordinates for SF City Hall `37.77919, -122.41914`
#  to indicate its location on the map with a nice, pop-up enabled maker.
#  (In the screenshot below, I used the black & white Stamen tiles, because they look cool).
#  ![example](https://raw.githubusercontent.com/suneman/socialdataanalysis2020/master/files/city_hall_2020.png)

""" Add Marker for the City Hall to Map"""
folium.Marker([37.77919, -122.41914],
              popup='City Hall',
              icon=folium.Icon( color='blue',
                                icon='university',
                                prefix='fa')).add_to(mapSF1)
""" Display Map"""
mapSF1




# > Now, let's plot some more data (no need for popups this time). Select a couple of months of data for `'DRUG/NARCOTIC'` and draw a little dot for each arrest for those two months. You could, for example, choose June-July 2016, but you can choose anything you like - the main concern is to not have too many points as this uses a lot of memory and makes Folium behave non-optimally.
#  We can call this a kind of visualization a *point scatter plot*.

""" INPUT VARIABLES """
crime = 'DRUG/NARCOTIC'     # pick a crime category 
start_date = '2016-06-01'   # format: yyyy-mm-dd
end_date = '2016-07-01'     # format: yyyy-mm-dd

""" Create a filtered dataframe based on new preferences """ 
Data['Date'] = pd.to_datetime(Data['Date'])
DataMap = Data[(Data['Date'] >= start_date) &
               (Data['Date'] < end_date) &
               (Data['Category']==crime)]

""" Create New Map instance """
mapSF2 = folium.Map(
    location = [37.7749, -122.4194],
    tiles = 'Stamen Toner',
    zoom_start = 12)

""" Add Makers to the map (based on preferences from above) """
for i, row in DataMap.iterrows():
    folium.CircleMarker([row['Y'], row['X']],
                        radius=1,
                        popup=row['Date'].date(),
                        color='red').add_to(mapSF2)
""" Display Map """ 
mapSF2




####################################
# Week 06 Part 3: KNN
####################################

# How does K-nearest-neighbors work? 
# Explain in your own words: What is the curse of dimensionality? 
# Use figure 12-6 in DSFS as part of your explanation.
"""
KNN is to Predict new point based on the points closest to it.
Curse of Dimensionality means that points in high-dimensional spaces tend not to be close to one another at all.
Figure 12-6 shows that: 
"In low-dimensional data sets, the closest points tend to be much closer than average.
when you have a lot of dimensions, it’s likely that the
closest points aren’t much closer than average, which means that two points being
close doesn’t mean very much"
[DSFS CH13]
"""

# Exercise: K-nearest-neighbors map.
# We know from last week's exercises that the focus crimes 
# PROSTITUTION, DRUG/NARCOTIC and DRIVING UNDER THE INFLUENCE 
# tend to be concentrated in certain neighborhoods, 
# so we focus on those crime types since they will make the most sense a KNN - map.

# Begin by using folium (see Week4) to plot all incidents of the three crime types on their own map. 
# This will give you an idea of how the various crimes are distributed across the city.

from IPython.display import display

""" load map data """
MapData = data[data['Category'].isin(['PROSTITUTION', 'DRUG/NARCOTIC', 'DRIVING UNDER THE INFLUENCE' ])]

"""
since Folium is not very memory efficient. 
a subset of the data, in Jan-Jul 2016, will be considered. 
[ref: Peter(TA)]
"""
MapData['Date'] = pd.to_datetime(MapData['Date'])
MapData     = MapData[(MapData['Date'] >= '2016-01-01') &
                      (MapData['Date'] <  '2016-07-01')]


""" create SF Map instance """
MapSF = folium.Map(location = [37.7749, -122.4194],tiles = 'Stamen Toner',zoom_start = 12)

""" Add Makers """
for _, row in MapData.iterrows():
    if(row['Category']=='PROSTITUTION'):
        folium.CircleMarker([row['Y'], row['X']],
                            radius=1,
                            popup="PROSTITUTION",
                            color='red').add_to(MapSF)

    elif(row['Category']=='DRUG/NARCOTIC'):
        folium.CircleMarker([row['Y'], row['X']],
                    radius=1,
                    popup="DRUG/NARCOTIC",
                    color='blue').add_to(MapSF)
    else:
        folium.CircleMarker([row['Y'], row['X']],
                    radius=1,
                    popup="DRIVING UNDER THE INFLUENCE",
                    color='yellow').add_to(MapSF)

""" Display Map Results: """
display(MapSF)


# Next, it's time to set up your model based on the actual data. 
# I recommend that you try out sklearn's KNeighborsClassifier. 
# For an intro, 
# start with this tutorial and follow the link 
# https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html
# to get a sense of the usage.

""" Split MapData in train and test data """
MapData_X = MapData[['Y','X']].values
MapData_y = MapData['Category'].values
""" A random permutation, to split the data randomly """
indices = np.random.permutation(len(MapData_X))
MapData_X_train = MapData_X[indices[:-20]]
MapData_y_train = MapData_y[indices[:-20]]
MapData_X_test  = MapData_X[indices[-20:]]
MapData_y_test  = MapData_y[indices[-20:]]


""" Create and fit a nearest-neighbor classifier """
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(MapData_X_train, MapData_y_train)

""" to left are predicted values and to right are the test values: """
list(zip(knn.predict(MapData_X_test),MapData_y_test))

""" calculate accuracy  """
from sklearn.metrics import accuracy_score
accuracy_score(knn.predict(MapData_X_test), MapData_y_test) 



# You don't have to think a lot about testing/training and accuracy 
# for this exercise. 
# We're mostly interested in creating a map 
# that's not too problematic. 
# But do calculate the number of observations of each crime-type 
# respectively. 
# You'll find that the levels of each crime varies 
# (lots of drug arrests, 
# an intermediate amount of prostitiution registered, 
# and very little drunk driving in the dataset). 
# Since the algorithm classifies each point according to 
# it's neighbors, what could a consequence of this imbalance 
# in the number of examples from each class mean for your map?
MapData['Category'].value_counts()

"""
Class Imbalance: 
    means that the probability of classifying any new random point as the class with more examples might becomes higher.
    Since the algorithm classifies each point according to it's neighbors.
    In other words, if a model always predict, for a new random points, 
    the class with most examples, this model will have somehow high accuracy!
    Not because this model is good but rather due class imbalance.
    a suggestion where to choose k=1 in this situation.  
    [ref: https://www.quora.com/Why-does-knn-get-effected-by-the-class-imbalance]
    
"""



# You can make the dataset 'balanced' by grabbing an equal number of examples from each crime category.
"""
here we will use the Down-sample Majority Class method 
see link: https://elitedatascience.com/imbalanced-classes
"""

""" Separate majority and minority classes """
MapData_X_y = MapData[['Category','Y','X']]
MapData_X_y_majority   = MapData_X_y[MapData_X_y['Category']=='DRUG/NARCOTIC']
MapData_X_y_minority_1 = MapData_X_y[MapData_X_y['Category']=='PROSTITUTION']
MapData_X_y_minority_2 = MapData_X_y[MapData_X_y['Category']=='DRIVING UNDER THE INFLUENCE']
 
""" Downsample class """
from sklearn.utils import resample
MapData_X_y_majority = resample(MapData_X_y_majority, 
                                replace=False,    # sample without replacement
                                n_samples=193,     # to match minority_2 class
                                random_state=123) # reproducible results
 
MapData_X_y_minority_1 = resample(MapData_X_y_minority_1, 
                                replace=False,    # sample without replacement
                                n_samples=193,     # to match minority_2 class
                                random_state=123) # reproducible results


""" Combine minority class with downsampled majority class """
MapData_X_y_downsampled = pd.concat([MapData_X_y_majority, MapData_X_y_minority_1, MapData_X_y_minority_2])

""" Shuffle DataFrame rows """
MapData_X_y_downsampled.sample(frac=1,random_state=123).reset_index(drop=True)

""" Check and Display new class counts """
MapData_X_y_downsampled['Category'].value_counts()



# How do you expect that will change the KNN result?
"""
This will increase the probability of classifying a new random point 
with one of the classes with less actual examples.
Thus decrease precision and recall of knn 
"""

# In which situations is the balanced map useful -
# When is the map where data is in proportion to occurrences useful?
# Choose which map you will work on in the following.
"""
Class balance is usefull when we want to 
emphasize the minority class in favor of the majority class.
i.g. detecting a disease based on biological inputs collected from patients

For the following we choose to work with the balanced map (downsampled data)
"""


# Now create an approximately square grid of point 
# that runs over SF. 
# You get to decide the grid-size, 
# but I recommend somewhere between 50 x 50 and 100 x 100  points. 
# I recommend using folium for this task.

""" Calculate the edges of the grid """
Y_min = MapData_X_y_downsampled['Y'].min()
Y_max = MapData_X_y_downsampled['Y'].max()
X_min = MapData_X_y_downsampled['X'].min()
X_max = MapData_X_y_downsampled['X'].max()

""" Calculate the grid matrix """
grid_size = 50
grid = []

Y_grid= np.linspace(Y_min, Y_max, num=grid_size)
X_grid= np.linspace(X_min, X_max, num=grid_size)

for lat in Y_grid: 
     for lon in X_grid:
        grid.append([lat, lon])


# Visualize your model by coloring the grid, 
# coloring each grid point according to it's category. 
# Create a plot of this kind for models where each point 
# is colored according to the majority of its 
# 5, 10, and 30 nearest neighbors. 
# Describe what happens to the map 
# as you increase the number of neighbors, K.
"""
we choose to work with the balanced map (downsampled data)
"""

""" Split balanced data in features matrix and respone vector """
MapData_X = MapData_X_y_downsampled[['Y','X']].values
MapData_y = MapData_X_y_downsampled['Category'].values

""" A random permutation, to split the data randomly """
np.random.seed(123)  # To fix random seed 
indices = np.random.permutation(len(MapData_y))

""" Split balanced data in 2/3 train and 1/3 test data """
one_Third = int(np.ceil(len(indices)/3))
MapData_X_train = MapData_X[indices[:-one_Third]]
MapData_y_train = MapData_y[indices[:-one_Third]]
MapData_X_test  = MapData_X[indices[-one_Third:]]
MapData_y_test  = MapData_y[indices[-one_Third:]]


""" Create SF Map instances """
MapSF1 = folium.Map(location = [37.7749, -122.4194],tiles = 'Stamen Toner',zoom_start = 12)
MapSF2 = folium.Map(location = [37.7749, -122.4194],tiles = 'Stamen Toner',zoom_start = 12)
MapSF3 = folium.Map(location = [37.7749, -122.4194],tiles = 'Stamen Toner',zoom_start = 12)

Maps = [MapSF1,MapSF2,MapSF3]
Models = []
for i,k in enumerate([5,10,30]):
    """ Create and fit a nearest-neighbor classifier """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(MapData_X_train, MapData_y_train)
    Models.append(knn)
    
    """ Start plotting """
    for g in grid:
        """ predict """
        prd = knn.predict([g])[0]
        
        """ plot according to the class """
        if (prd== 'DRUG/NARCOTIC'):
            folium.CircleMarker(g,
                                radius=1,
                                popup='DRUG/NARCOTIC',
                                color='blue',
                                opacity=0.5).add_to(Maps[i])

        elif (prd== 'PROSTITUTION'):
            folium.CircleMarker(g,
                                radius=1,
                                popup='PROSTITUTION',
                                color='red',
                                opacity=0.5).add_to(Maps[i])

        else :
            folium.CircleMarker(g,
                                radius=1,
                                popup='DRIVING UNDER THE INFLUENCE',
                                color='yellow',
                                opacity=0.5).add_to(Maps[i])

""" display maps """
# Map k = 5
display(Maps[0])

# Map K = 10
display(Maps[1]) 

# Map k = 30
display(Maps[2])

# Describe what happens to the map as you increase the number of neighbors, K.
""" calculate accuracy """
for k,m in zip([5,10,30],Models):
    print("With K = {} neighbors:\n     accuracy = {}\n".format(k, accuracy_score(m.predict(MapData_X_test), MapData_y_test)))

"""
It is seen from the the output of the code above that:
Accuracy decrases as k increases. 
This is due to the fact that in the balanced data as k increases 
the minority classes emphasize more in favor of the majority class 
as expected!
"""