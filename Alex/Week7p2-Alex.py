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


# Pick the filename of the dataset to import in pandas
fileName = 'Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv'
filePath = os.path.abspath(os.path.join(os.getcwd(), '..','Datasets', fileName))

# Raw Dataframe as created directly from csv
df_raw = pd.read_csv(filePath)

# Edit and store the Dataframe
df = df_raw[['PdDistrict','Category']].copy()
df['Month'] = pd.to_datetime(df_raw['Date']).dt.month
df['Hour'] = pd.to_datetime(df_raw['Time']).dt.hour
df['DayOfWeek'] = pd.to_datetime(df_raw['Date']).dt.dayofweek
df = df.dropna()

# Turn PdDistricts to integers
PdDistrict_list = sorted(df['PdDistrict'].unique())
df['PdDistrict'] = df['PdDistrict'].apply(lambda x: PdDistrict_list.index(x))


# Select Crimes of Interest for the Classification Task
crime1 = 'BURGLARY' # 'VEHICLE THEFT'
crime2 = 'FRAUD' # 'FORGERY/COUNTERFEITIN'

dfSlice = df[(df['Category']==crime1) | (df['Category']==crime2)].copy()

# Turn selected Categories to integers
Category_counts = dfSlice.Category.value_counts().reset_index().to_numpy()
nCategories = Category_counts.shape[0]
dfSlice['Category'] = dfSlice['Category'].apply(lambda x: int(np.argwhere(Category_counts[:,0]==x)))

# Full list of incidents per crime
y_raw = dfSlice['Category'].to_numpy().reshape(-1,1)
x_raw = dfSlice.drop(['Category'], axis=1).to_numpy()

# Acquire dataset which is equally distributed between the 2 crimes 
minCategoryCount = np.min(Category_counts[:,1]) # num of instances of the category having least instances
catIdx=np.zeros((minCategoryCount,nCategories),dtype=int)
for i,cat in enumerate(range(nCategories)):
    catIdx[:,i] = np.random.choice(np.where(y_raw==cat)[0], minCategoryCount, replace=False)

# Store ML-ready dataset
x = x_raw[np.concatenate(catIdx),:]
y = y_raw[np.concatenate(catIdx),:]


# K-Fold Cross Validation with Random Forrest==================================
cvf = 10    # number of folds for k-fold CV

# Random Forrest Parameters
nTrees = 500           # number of trees
criterion = 'entropy'  # criterion
max_depth = 4          # max tree depth
random_state=0         # random state

kfold = KFold(n_splits=10, shuffle=True)

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
         +f'\nParameters: nTrees:{nTrees}, criterion:{criterion}, maxDepth:{max_depth}, randomState:{random_state}')

plt.figure()
sn.heatmap(confMatrixKFpct, annot=labels, fmt='', cmap="Blues")
plt.xlabel('Predicted Crime Categories')
plt.ylabel('True Crime Categories')
plt.title(title)
plt.show()
