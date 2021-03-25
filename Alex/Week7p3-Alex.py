import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
import pdb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# INITIALIZATIONS=============================================================
# Datasets Related
fileName = 'Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv' # crimes filename
wfileName = 'weather_data.csv' # weather filename
crime1 = 'BURGLARY' # 'VEHICLE THEFT'
crime2 = 'FRAUD' # 'FORGERY/COUNTERFEITIN'

# ML / Random Forest Related
cvf = 10               # number of folds for k-fold cross validation
nTrees = 900           # number of trees
criterion = 'entropy'  # criterion
max_depth = 11         # max tree depth
min_leaf_size = 2      # minimum number of samples possible at a leaf node
random_state = 0       # random state
# ============================================================================


# Generate filepaths
filePath = os.path.abspath(os.path.join(os.getcwd(), '..','Datasets', fileName))
wfilePath = os.path.abspath(os.path.join(os.getcwd(), '..','Datasets', wfileName))

# Import and edit Crime Dataset-----------------------------------------------
df =  pd.read_csv(filePath, usecols=["Category", "Date", "Time", "PdDistrict"])
df = df[df["Category"].isin([crime1, crime2])]
df['Month'] = pd.to_datetime(df['Date']).dt.month
df['Hour'] = pd.to_datetime(df['Time']).dt.hour
df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
df["datetime"] = df.apply(lambda x: pd.to_datetime(x.Date + " " + x.Time).round("H").tz_localize("ETC/GMT-7"), axis = 1) 

# Turn PdDistricts to integers
PdDistrict_list = sorted(df['PdDistrict'].unique())
df['PdDistrict'] = df['PdDistrict'].apply(lambda x: PdDistrict_list.index(x))

# Turn selected Categories to integers
Category_list = sorted(df['Category'].unique())
df['Category'] = df['Category'].apply(lambda x: Category_list.index(x))


# Import Weather Dataset------------------------------------------------------
wdf = pd.read_csv(wfilePath, parse_dates=["date"],
                date_parser=lambda x: pd.to_datetime(x).tz_convert(None).tz_localize("Etc/GMT+3").tz_convert("Etc/GMT-7"),
                usecols=['date', 'weather']) 
wdf = wdf.rename(columns={'date':'datetime'})

# Turn weather to integers
weather_list = sorted(wdf['weather'].unique())
wdf['weather'] = wdf['weather'].apply(lambda x: weather_list.index(x))


# Merge Crimes with Weather Dataframes----------------------------------------
dfMerge = df.merge(wdf, on='datetime')
dfMerge = dfMerge.dropna()


# Count different crime categories
Category_counts = dfMerge.Category.value_counts().reset_index().to_numpy()[:,1]
nCategories = Category_counts.size

# Full list of incidents per crime
y_raw = dfMerge['Category'].to_numpy().reshape(-1,1)
x_raw = dfMerge.drop(['Category','Date','Time','datetime'], axis=1).to_numpy()


# Acquire dataset which is equally distributed between the 2 crime categories 
minCategoryCount = np.min(Category_counts) # num of instances of the category having least instances
catIdx=np.zeros((minCategoryCount,nCategories),dtype=int)
for i,cat in enumerate(range(nCategories)):
    catIdx[:,i] = np.random.choice(np.where(y_raw==cat)[0], minCategoryCount, replace=False)

# Store ML-ready datasets
x = x_raw[np.concatenate(catIdx),:]
y = y_raw[np.concatenate(catIdx),:]


# K-Fold Cross Validation with Random Forrest---------------------------------
kfold = KFold(n_splits=cvf, shuffle=True)

# KFold CV intermediary variables
accuracies = [None]*cvf
targets_true = np.array([])
targets_pred = np.array([])

fold_count = 0
for train_index, test_index in tqdm(kfold.split(x)):
    
    x_train, x_test = x[train_index,:], x[test_index,:]
    y_train, y_test = y[train_index,:], y[test_index,:]

    y_train = y_train.ravel()
    y_test = y_test.ravel()
    
    # z-Score Standardization
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    
    rfClassifier = RandomForestClassifier(n_estimators=nTrees,
                                        criterion=criterion,
                                        max_depth=max_depth,
                                        min_samples_leaf=min_leaf_size,
                                        random_state=random_state)
    rfClassifier.fit(x_train,y_train)
    y_pred = rfClassifier.predict(x_test)
    
    confMatrixFold = confusion_matrix(y_test, y_pred)
    confMatrixFoldpct = confMatrixFold/np.sum(confMatrixFold,axis=1).reshape(-1,1)
    #when applying .ravel() to confusion matrix the order is: TN, FP, FN, TP
    
    accuracies[fold_count] = np.trace(confMatrixFoldpct)/nCategories
    
    # Store predictions and true values
    targets_true = np.concatenate((targets_true, y_test), axis=0)
    targets_pred = np.concatenate((targets_pred, y_pred), axis=0)
    fold_count +=1
        
        
confMatrixKF = confusion_matrix(targets_true, targets_pred)
confMatrixKFpct = confMatrixKF/np.sum(confMatrixKF,axis=1).reshape(-1,1)
#when applying .ravel() to confusion matrix the order is: TN, FP, FN, TP

cm = ['{0:0.0f}'.format(val) for val in confMatrixKF.flatten()]
cmpct = ['{0:.2%}'.format(val) for val in confMatrixKFpct.flatten()]
labels = [f'{val1}({val2})' for val1, val2 in zip(cm,cmpct)]
labels = np.asarray(labels).reshape(nCategories,nCategories)
    
title = ('Crime Prediction using Random Forest Classifier'
         +'\nPerformance on 10-Fold Cross Validation'
         +f'\nCrime Categories: {crime1}(0), {crime2}(1)'
         +f'\nDataset: {minCategoryCount} randomly selected instances per category'
         +f'\nClassifier Parameters: nTrees:{nTrees}, maxDepth:{max_depth}, min_leaf_size:{min_leaf_size}'
         +f'\ncriterion:{criterion}, randomState:{random_state}')

plt.figure()
sn.heatmap(confMatrixKFpct, annot=labels, fmt='', cmap="Blues")
plt.xlabel('Predicted Crime Categories')
plt.ylabel('True Crime Categories')
plt.title(title)
plt.show()
