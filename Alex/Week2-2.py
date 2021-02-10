import os
import pandas as pd
import matplotlib.pyplot as plt

# Pick the filename of the dataset to import in pandas
fileName = 'Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv'
filePath = os.path.abspath(os.path.join(os.getcwd(), '..', 'Datasets', fileName))
df = pd.read_csv(filePath)

focuscrimes = set(['WEAPON LAWS', 'PROSTITUTION',
                   'DRIVING UNDER THE INFLUENCE',
                   'ROBBERY', 'BURGLARY', 'ASSAULT',
                   'DRUNKENNESS', 'DRUG/NARCOTIC',
                   'TRESPASS', 'LARCENY/THEFT',
                   'VANDALISM', 'VEHICLE THEFT',
                   'STOLEN PROPERTY', 'DISORDERLY CONDUCT'])


# YEAR-BY-YEAR DEVELOPMENT=====================================================
df['Year'] = pd.to_datetime(df['Date']).dt.to_period('Y')
df2 = df[df['Year'] != '2018'].copy() # drop rows corresponding to 2018


# Plots for year-by-year development for all Focus Crimes
fig, axarr = plt.subplots(7, 2, figsize=(17, 20), sharex=True)
plt.suptitle('Year-by-year development of specific focus crimes', fontsize=30)

for i,crime in enumerate(focuscrimes):
    focusCrimeDf = pd.DataFrame(df2.groupby(['Category', 'Year']).count()
                             .filter(like = crime, axis=0)['PdId']).unstack(level=0)
    x1 = focusCrimeDf.plot(kind='bar', ax=axarr[i//2,i%2], title=crime, rot=0, grid=True, legend=False)
    x2 = focusCrimeDf.plot(kind='bar', ax=axarr[i//2,i%2], title=crime, rot=0, grid=True, legend=False)
 
    
    
# WEEKLY DAY-BY-DAY DEVELOPMENT================================================
df2['DayOfWeek'] = pd.Categorical(df2['DayOfWeek'],
                                  categories=['Monday','Tuesday','Wednesday',
                                              'Thursday','Friday','Saturday', 
                                              'Sunday'], ordered=True)

fig, axarr = plt.subplots(7, 2, figsize=(17, 20), sharex=True)
plt.suptitle('Weekly day-by-day development of specific focus crimes', fontsize=30)

for i,crime in enumerate(focuscrimes):
    focusCrimeDf = pd.DataFrame(df2.groupby(['Category', 'DayOfWeek']).count()
                             .filter(like = crime, axis=0)['PdId']).unstack(level=0)
    x1 = focusCrimeDf.plot(kind='bar', ax=axarr[i//2,i%2], title=crime, rot=0, grid=True, legend=False)
    x2 = focusCrimeDf.plot(kind='bar', ax=axarr[i//2,i%2], title=crime, rot=0, grid=True, legend=False)

    
# MONTH-BY-MONTH DEVELOPMENT===================================================
df2['Month'] = pd.Categorical(pd.to_datetime(df2['Date']).dt.month_name(),
                              categories=['January', 'February', 'March', 'April', 'May',
                                          'June', 'July', 'August', 'September', 'October',
                                          'November', 'December'], ordered=True)

fig, axarr = plt.subplots(7, 2, figsize=(17, 20), sharex=True)
plt.suptitle('Month-by-month development of specific focus crimes', fontsize=30)
for i,crime in enumerate(focuscrimes):
    focusCrimeDf = pd.DataFrame(df2.groupby(['Category', 'Month']).count()
                             .filter(like = crime, axis=0)['PdId']).unstack(level=0)
    x1 = focusCrimeDf.plot(kind='bar', ax=axarr[i//2,i%2], title=crime, rot=90, grid=True, legend=False)
    x2 = focusCrimeDf.plot(kind='bar', ax=axarr[i//2,i%2], title=crime, rot=90, grid=True, legend=False)

    