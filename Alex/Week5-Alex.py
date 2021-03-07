import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========INITIALIZATIONS======================
# Pick the filename of the dataset to import in pandas
fileName = 'Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv'
filePath = os.path.abspath(os.path.join(os.getcwd(), '..','Datasets', fileName))

# Raw Dataframe as created directly from csv
df_raw = pd.read_csv(filePath)

# Edit and store the Dataframe
df = df_raw.copy()
df['Date'] = pd.to_datetime(df['Date'])

focuscrimes = set(['WEAPON LAWS', 'PROSTITUTION',
                   'DRIVING UNDER THE INFLUENCE',
                   'ROBBERY', 'BURGLARY', 'ASSAULT',
                   'DRUNKENNESS', 'DRUG/NARCOTIC',
                   'TRESPASS', 'LARCENY/THEFT',
                   'VANDALISM', 'VEHICLE THEFT',
                   'STOLEN PROPERTY', 'DISORDERLY CONDUCT'])

# =======PART 3==========
# Incidents per Category
crimesPerCategory = (df
                  .groupby(by='Category')
                  .PdId.count()
                  .sort_values(ascending=False)
                  .reset_index())

# Plot with Linear Axes
plt.figure(figsize=(12, 8))
plt.bar(crimesPerCategory['Category'], crimesPerCategory['PdId'])
plt.xticks(fontsize=14, rotation=90)
plt.grid(True)
plt.title('Occurances per different Crime Category in SF')
plt.show()

# Plot with Log y-Axis
plt.figure(figsize=(12, 8))
plt.bar(crimesPerCategory['Category'], crimesPerCategory['PdId'])
plt.yscale('log')
plt.xticks(fontsize=14, rotation=90)
plt.grid(True)
plt.title('Occurances per different Crime Category in SF (log y-axis)')
plt.show()

# Spatial Bin Plotting
nbins = 150 # to give a 100x100 array

# Filter for Lattitude/Longitude Outliers
y_lim = (37.71, 37.83) # SF latitude limits
x_lim = (-122.52, -122.36) # SF longitude limits

df=df[(df['X']<x_lim[1])
      &(df['X']>x_lim[0])
      &(df['Y']<y_lim[1])
      &(df['Y']>x_lim[0])] # filter for outliers

# X=Longitude / Y=Latitude
Xmin = df['X'].min(); Xmax = df['X'].max()
Ymin = df['Y'].min(); Ymax = df['Y'].max()

Xedges = np.linspace(Xmin,Xmax,nbins)
Yedges = np.linspace(Ymin,Ymax,nbins)
 
H, xedges, yedges = np.histogram2d(df['X'].values, df['Y'].values, nbins)
H=H.T

plt.figure(figsize=(10,10))
plt.imshow(H, cmap='hot',vmax=3000, origin='lower')
plt.colorbar()
plt.xlabel('Longitude - X')
plt.ylabel('Lattitude - Y')
plt.title('Spatial Crime Distribution - Split in X/Y with {} bins'.format(nbins))
plt.show()

# Spatial Bin Occurrence Histogram
Hvalues, Hcounts = np.unique(H, return_counts=True)

# Linear Axes
plt.figure(figsize=(12, 8))
plt.plot(Hvalues, Hcounts)
plt.xticks(fontsize=14, rotation=90)
plt.grid(True)
plt.xlabel('Unique Occurence Number per bin')
plt.ylabel('Counts of the Unique Occurence Number')
plt.title('Distribution of crime occurrence counts for different spatial bins')
plt.show()

# Loglog Axes
plt.figure(figsize=(12, 8))
plt.plot(Hvalues, Hcounts)
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=14, rotation=90)
plt.grid(True)
plt.xlabel('Unique Occurence Number per bin')
plt.ylabel('Counts of the Unique Occurence Number')
plt.title('Distribution of crime occurrence counts for different spatial bins (loglog axes)')
plt.show()

