import os
import urllib.request
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pdb

# Declare the directories and file names
datasetsDir = os.path.abspath(os.path.join(os.getcwd(), '..', 'Datasets'))
nFiles = 4

dataFiles = []
for i in range(nFiles):
    dataFiles.append(os.path.join(datasetsDir, 'wk2_data'+str(i+1)+'.tsv'))


# Check if Data1,..,Data4 exist otherwise download
if not os.path.isfile(dataFiles[0]):
    print('wk2_data1.tsv not found in Datasets, downloading now...')
    url1 = 'https://raw.githubusercontent.com/suneman/socialdata2021/master/files/data1.tsv'
    urllib.request.urlretrieve(url1, dataFiles[0])

if not os.path.isfile(dataFiles[1]):
    print('wk2_data2.tsv not found in Datasets, downloading now...')
    url2 = 'https://raw.githubusercontent.com/suneman/socialdata2021/master/files/data2.tsv'
    urllib.request.urlretrieve(url2, dataFiles[1])
    
if not os.path.isfile(dataFiles[2]):
    print('wk2_data3.tsv not found in Datasets, downloading now...')
    url3 = 'https://raw.githubusercontent.com/suneman/socialdata2021/master/files/data3.tsv'
    urllib.request.urlretrieve(url3, dataFiles[2])
    
if not os.path.isfile(dataFiles[3]):
    print('wk2_data4.tsv not found in Datasets, downloading now...')
    url4 = 'https://raw.githubusercontent.com/suneman/socialdata2021/master/files/data4.tsv'
    urllib.request.urlretrieve(url4, dataFiles[3])

data = []

# Import the datasets
for i in range(nFiles):
    data.append(np.genfromtxt(dataFiles[i], delimiter='\t'))

# Calculate Mean
mean = list(map(lambda x: np.mean(x, axis=0), data))
for i in range(nFiles):
    print('Mean for x-y for data{} is: {:10.2f} and {:10.2f} respectively.'
          .format(i+1, mean[i][0], mean[i][1]))

# Calculate Variance
variance = list(map(lambda x: np.var(x, axis=0), data))
for i in range(nFiles):
    print('Variance for x-y for data{} is: {:10.3f} and {:10.3f} respectively.'
          .format(i+1, variance[i][0], variance[i][1]))
    
# Calculate Pearson Correlation Coeff
corr = list(map(lambda x: np.corrcoef(x[:,0],x[:,1]), data))
for i in range(nFiles):
    print('Correlation between x-y for data{} is: {:10.3f}'
          .format(i+1, corr[i][0,1]))

# Fit Linear Regression
lr = list(map(lambda x: stats.linregress(x[:,0],x[:,1]), data))

# Plot x-y data and their Linear Regression Result
for i in range(nFiles):
    x = data[i][:,0]
    y = data[i][:,1]
    ylr = lr[i].intercept + lr[i].slope*x
    
    plt.subplot(int('22'+str(i+1)))
    plt.scatter(x,y)
    plt.plot(x,ylr, c= "red", linestyle=':')
    plt.xlim((2, 20))
    plt.ylim((2, 14))
    plt.grid()
plt.show()
