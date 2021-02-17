import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb

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


df = df[['PdId','Category','Date','Time','X']].copy()
df['Category'] = df['Category'].astype(str)
df = df[df['Category'].isin(focuscrimes)]
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

pdb.set_trace()

# Take a Range of Dates / Pick Category / Keep Latitude (X)
#startDate = '2010-01-01'; endDate = '2012-01-01'; crimeType = 'VEHICLE THEFT';
startDate = '2013-01-01'; endDate = '2015-01-01'; crimeType = 'DRUG/NARCOTIC';


df2 = (df.loc[(df['Date'] > startDate)
             & (df['Date'] < endDate) 
             & (df['Category'] == crimeType), 'X']
         .reset_index()['X'])


plt.figure()
stats.probplot(df2, plot=plt)
plt.title('Probability Plot for Latitude \'X\' of Crime: {}'.format(crimeType)
          +'\nAcross Days for the range {} - {}'.format(startDate, endDate)
          +'\nTotal Incidents in this period: {}'.format(df2.shape[0]))
plt.grid()
plt.show()


## CRIMES PER DAY - BOX PLOTS
df3 = df[['Category', 'Date']].copy()
df3['Date'] = df3['Date'].astype(int)/10**9
df3.boxplot(column='Date', by='Category', rot=90)

'''
fig, axarr = plt.subplots(1, 14, figsize=(40, 10), sharex=True)
plt.suptitle('Box Plots: Focus Crimes Daily Incidents')
for i,crime in enumerate(tqdm(focuscrimes)):
    df3 = pd.DataFrame(df.groupby(['Category', 'Date']).count()
                            .filter(like = crime, axis=0)).reset_index()[['Date','PdId']]
    axarr[i].set_title(crime)
    axarr[i].boxplot(df3['PdId'])
'''

## CRIMES PER TIME-OF-DAY - BOX PLOTS
df4 = df[['Category','Time']].copy()
fTime2Sec = lambda x: int(x.split(':')[0])*3600 + int(x.split(':')[1])*60
df4['Time'] = df4['Time'].apply(fTime2Sec)
df4.boxplot(column='Time', by='Category', rot=90)