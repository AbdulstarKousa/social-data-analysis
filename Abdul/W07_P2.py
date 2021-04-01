####################################
# Part 2: Random forest and weather
####################################


# Importing needed libraries:
import os 
import numpy as np 
import pandas as pd 
import calendar
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.stats import ks_2samp
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


# Loading data
""" Importing dataset using os""" 
data_dir = r"C:\Users\boody\OneDrive - Danmarks Tekniske Universitet\01 MSc\2st Semester\01_SD\GitHub_Group\social-data-analysis\Datasets\Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv"
Data = pd.read_csv(data_dir) 

""" Removing the 2018 (since we don't have full data for 2018) """
Data['Year'] = pd.to_datetime(Data['Date']).dt.year
Data = Data[Data['Year'] != 2018]     


# Test for different spatio-temporal distribution of the two selected crime categories, 
# BURGLARY and FRAUD
# Using Kolmogorov-Smirnov statistic:  
""" selected Crimes """
Selected_Crimes = ['BURGLARY','FRAUD']

""" add 'Hour' feature by transfering 'Date' to numeric 'Hour' """
Data['Hour'] = pd.to_datetime(Data['Time']).dt.hour

""" add 'Month' feature by transfering 'Date' """
Data['Month'] = pd.to_datetime(Data['Date']).dt.month

""" Kolmogorov-Smirnov on Y """
d1 = Data[Data['Category'] == Selected_Crimes[0]]['Y']
d2 = Data[Data['Category'] == Selected_Crimes[1]]['Y']
print("Test under \u03B1 = 0.5 is: ", ks_2samp(d1, d2)[1]<0.5)

""" Kolmogorov-Smirnov on Hour """
d1 = Data[Data['Category'] == Selected_Crimes[0]]['Hour']
d2 = Data[Data['Category'] == Selected_Crimes[1]]['Hour']
print("Test under \u03B1 = 0.5 is: ", ks_2samp(d1, d2)[1]<0.5)

""" Kolmogorov-Smirnov on Month  """
d1 = Data[Data['Category'] == Selected_Crimes[0]]['Month']
d2 = Data[Data['Category'] == Selected_Crimes[1]]['Month']
print("Test under \u03B1 = 0.5 is: ", ks_2samp(d1, d2)[1]<0.5)

""" Plot the Selected two crime categories for jan [2003-2017] """
for crime in Selected_Crimes:
    d = Data[Data['Category'] == crime]
    d = d[d['Month']==1]
    plt.figure(figsize=(6, 6))
    plt.hist(d['Y'], bins=50) 
    plt.title( crime + "'s Latitude (Y)" + '\nJanuary [2003-2017]''\nTotal number of observations = '+ str(len(d)))
    plt.xlabel(crime)
    plt.ylabel('Number of Observations')
    plt.show()

""" free memory """
del (d,d1,d2)



# Slice needed data 
""" slice the two selected crimes Categories """
data = Data[Data['Category'].isin(Selected_Crimes)].copy()

""" slice features: Category and the four selected features Month, DayOfWeek, Hour and PdDistrict """
data = data[['Category','Month','DayOfWeek','Hour','PdDistrict']]

""" free memory """
del (Data)



# Prepare data
numeric_transformer = StandardScaler()
categorical_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('lenc', OrdinalEncoder()),
        ('stand', StandardScaler())
        ])

preprocessor = ColumnTransformer(
        transformers=[
        ('num', numeric_transformer, selector(dtype_include="int64")),
        ('cat', categorical_transformer, selector(dtype_include="object"))
        ])



# Class balance:
"""
here we will use the Down-sample Majority Class method 
see link: https://elitedatascience.com/imbalanced-classes
"""

""" count data """
data['Category'].value_counts()

""" Separate majority and minority classes """
data_majority   = data[data['Category']==Selected_Crimes[0]]
data_minority   = data[data['Category']==Selected_Crimes[1]]
 
""" Downsample class """
seed = 123
data_majority = resample(data_majority, 
                    replace=False,                  # sample without replacement
                    n_samples=len(data_minority),   # to match minority class
                    random_state=seed)              # reproducible results

data_minority = resample(data_minority, 
                    replace=False,                  # sample without replacement
                    n_samples=len(data_minority),   # to match minority class
                    random_state=seed)              # reproducible results


""" Combine minority class with downsampled majority class """
data_downsampled = pd.concat([data_majority, data_minority])

""" Shuffle DataFrame rows and reset index """
data_downsampled = data_downsampled.sample(frac=1, random_state=seed).reset_index(drop=True)

""" Check and Display new class counts """
data_downsampled['Category'].value_counts()

""" free memory """
del(data_majority,data_minority,data)



# Split data for Learning
""" Split-out X,Y dataset """
X = data_downsampled.iloc[:,1:]
Y = data_downsampled.iloc[:,0]

""" Split-out validation dataset """
validation_size = 0.8  # 16216 sample to train on 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed, stratify=Y)



# Compare Algorithms, RandomForest vs DecisionTree (Both with sklearn Default settings): 
pipelines = []
pipelines.append(('RF', Pipeline([('preprocessor', preprocessor),('rf',RandomForestClassifier())])))
pipelines.append(('DT', Pipeline([('preprocessor', preprocessor),('dt',DecisionTreeClassifier())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10,shuffle= True, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()



# RandomForestClassifier tuning:
"""
- Explore Number of Trees    (n_estimators)      = to be investegated with CV
- Explore Minimum Node Size  (min_samples_leaf)  = to be investegated with CV
- Explore Number of Features (max_features)      = 'sqrt' (Heuristic: the square root of the number of input features)  
- Explore the quality of a split (criterion)     = entropy(Heuristic: a measure of information that indicates the disorder of the features with the target)
"""


# Random Grid Search (3x3 CV) for hyperparameters focus area:
model = Pipeline([('preprocessor', preprocessor),('rf',RandomForestClassifier(max_features = 'sqrt', criterion = 'entropy'))])

n_estimators = [100,500,1000,2000]
min_samples_leaf = [1, 50, 100, 200]
random_grid = {'rf__n_estimators': n_estimators,
               'rf__min_samples_leaf':min_samples_leaf }

s_kfold = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=seed)
grid  = RandomizedSearchCV(estimator = model, param_distributions = random_grid, scoring='accuracy', n_iter = 10, cv = s_kfold, verbose=2, n_jobs = -1)
grid_result = grid.fit(X_train, Y_train)

print('Accuracy:     %.3f' % grid_result.best_score_)
print('random_best_params_:   %s' % grid_result.best_params_)
"""
> Accuracy:     0.623
> random_best_params_:   {'rf__n_estimators': 500, 
                          'rf__min_samples_leaf': 50}
"""



# Grid Search with Cross Validation ( 5x5 CV focused around RandomizedSearchCV's best hyperparameters)
model = Pipeline([('preprocessor', preprocessor),('rf',RandomForestClassifier(max_features = 'sqrt', criterion = 'entropy'))])

n_estimators = [400, 500, 600]
min_samples_leaf = [25, 50, 75]
param_grid = {'rf__n_estimators': n_estimators,
              'rf__min_samples_leaf':min_samples_leaf }

s_kfold = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=seed)
grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring='accuracy', cv = s_kfold, n_jobs = -1, verbose = 2)
grid_result = grid.fit(X_train, Y_train)

print('Accuracy:     %.3f' % grid_result.best_score_)
print('best_params_:   %s' % grid_result.best_params_)
"""
> Accuracy:     0.623
> random_best_params_:   {'n_estimators': 500, 
                          'min_samples_leaf': 50}
"""


# prepare final model (with GridSearchCV best hyperparameters)
model = Pipeline([('preprocessor', preprocessor),('rf',RandomForestClassifier(n_estimators= 500, min_samples_leaf = 50, max_features = 'sqrt', criterion = 'entropy'))])
model.fit(X_train, Y_train)


# Report:
predictions = model.predict(X_validation)
print('Accuracy:     %.3f' % accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

"""
> Accuracy:    0.635
"""
"""
>
[[21300 11132]
 [12567 19865]]
"""
"""
>
              precision    recall  f1-score   support

    BURGLARY       0.63      0.66      0.64     32432
       FRAUD       0.64      0.61      0.63     32432

    accuracy                           0.63     64864
   macro avg       0.63      0.63      0.63     64864
weighted avg       0.63      0.63      0.63     64864
"""



# Feature selection Permutation feature importance:
from sklearn.inspection import permutation_importance

X = X_train 
y = Y_train

result = permutation_importance(model, X, y, n_repeats=10,random_state=seed, scoring='accuracy')
result.importances_mean
result.importances_std

col_names = list(X_train.columns)

for i in result.importances_mean.argsort()[::-1]:
     if result.importances_mean[i] - 2 * result.importances_std[i] > 0:
         print(f"{col_names[i]:<20}"
               f"{result.importances_mean[i]:.3f}"
               f" +/- {result.importances_std[i]:.3f}")



""" 1
>
Hour                0.121 +/- 0.002
PdDistrict          0.049 +/- 0.003
DayOfWeek           0.012 +/- 0.001
Month               0.010 +/- 0.001
"""