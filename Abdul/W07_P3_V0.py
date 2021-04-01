# Load needed libraries:
""" Data Handeling """
import numpy as np 
import pandas as pd 
import calendar

""" Display """
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

""" Statistic """
from scipy.stats import ks_2samp

""" Cross Validation """
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

""" Evaluation Metrics """ 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

""" Data Preprocessing  """
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.inspection import permutation_importance

""" Models """
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# Load Data
""" Load Weather Data """ 
weather_url = 'https://raw.githubusercontent.com/suneman/socialdata2021/master/files/weather_data.csv'
weather_data = pd.read_csv(weather_url) 

""" Load Crime Data """
crimes_path = r"C:\Users\boody\OneDrive - Danmarks Tekniske Universitet\01 MSc\2st Semester\01_SD\GitHub_Group\social-data-analysis\Datasets\Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv"
crimes_data = pd.read_csv(crimes_path, usecols=['Category', 'Date', 'Time', 'DayOfWeek','PdDistrict', 'Y'])
crimes_data = crimes_data[crimes_data["Category"].isin(['BURGLARY', 'FRAUD'])]


# Merge: Data = weatherdata + crimedata 
""" Shift Weather time-zone """
weather_data['date'] = weather_data['date'].apply(lambda x: pd.to_datetime(x).tz_convert(None).tz_localize("Etc/GMT+3").tz_convert("Etc/GMT-7")) 

""" Shift Crime time-zone """
crimes_data['datetime'] = crimes_data.apply(lambda x: pd.to_datetime(x.Date + " " + x.Time).round("H").tz_localize("ETC/GMT-7"), axis = 1)

""" Merge weather and crime Data """
weather_data = weather_data.rename(columns={'date':'datetime'})
Data = crimes_data.merge(weather_data, on='datetime', how= 'inner')

""" check if all entries has of weather info """
Data['weather'].value_counts()

""" Free memory """
del (crimes_data, weather_data)


# Rearrange Merged Data: 
""" Delete unneeded columns """
Data = Data.drop(columns=['Date','Time'])

""" Delete messing values: """  
Data = Data.dropna()

""" Rearrange columns """
l = list(Data.columns)
l.remove('datetime')
l.remove('Category')
Columns = ['Category','datetime'] + l 
Data = Data[Columns]
Data = Data.rename(columns={'datetime':'Date'})


# Overview of Merged Data:
Data.head()
Data.info()


# Add new features 
""" add 'Hour' feature by transfering 'Date' to numeric 'Hour' """
Data['Hour'] = pd.to_datetime(Data['Date']).dt.hour

""" add 'Month' feature by transfering 'Date' """
Data['Month'] = pd.to_datetime(Data['Date']).dt.month


# Test for different spatio-temporal distribution of BURGLARY and FRAUD 
# Using Kolmogorov-Smirnov statistic:  
""" selected Crimes """
Selected_Crimes = ['BURGLARY','FRAUD']

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
""" drop Date and Y """
data = Data.drop(columns=['Date', 'Y']).copy()

""" free data """
del(Data)


# Prepare data : making a preprocessor
numeric_transformer = StandardScaler()


categorical_transformer = Pipeline(
        steps = [
        ('enc', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)),
        ('stand', StandardScaler())
        ])

preprocessor = ColumnTransformer(
        transformers=[
        ('num', numeric_transformer, selector(dtype_exclude="object")),
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
validation_size = 0.6  # 11440 samples to train on 
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
> Accuracy:     0.620
> random_best_params_:   {'rf__n_estimators': 1000, 
                          'rf__min_samples_leaf': 50}
"""


# Grid Search with Cross Validation ( 5x5 CV focused around RandomizedSearchCV's best hyperparameters)
model = Pipeline([('preprocessor', preprocessor),('rf',RandomForestClassifier(max_features = 'sqrt', criterion = 'entropy'))])

n_estimators = [7250, 1000, 1250]
min_samples_leaf = [25, 50, 75]
param_grid = {'rf__n_estimators': n_estimators,
              'rf__min_samples_leaf':min_samples_leaf }

s_kfold = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=seed)
grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring='accuracy', cv = s_kfold, n_jobs = -1, verbose = 2)
grid_result = grid.fit(X_train, Y_train)

print('Accuracy:     %.3f' % grid_result.best_score_)
print('best_params_:   %s' % grid_result.best_params_)
"""
> Accuracy:     0.621
> random_best_params_:   {'n_estimators': 1250, 
                          'min_samples_leaf': 25}
"""

# prepare final model (with GridSearchCV best hyperparameters)
model = Pipeline([('preprocessor', preprocessor),('rf',RandomForestClassifier(n_estimators= 1250, min_samples_leaf = 25, max_features = 'sqrt', criterion = 'entropy'))])
model.fit(X_train, Y_train)


# Report:
predictions = model.predict(X_validation)
print('Accuracy:     %.3f' % accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

"""
> Accuracy:     0.619
"""
"""
>
[[5301 3280]
 [3261 5320]]
"""
"""
>
              precision    recall  f1-score   support

    BURGLARY       0.62      0.62      0.62      8581
       FRAUD       0.62      0.62      0.62      8581

    accuracy                           0.62     17162
   macro avg       0.62      0.62      0.62     17162
weighted avg       0.62      0.62      0.62     17162
"""


# Feature selection Permutation feature importance:
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


""" 2
>
Hour                0.144 +/- 0.004
PdDistrict          0.052 +/- 0.003
wind_direction      0.042 +/- 0.001
temperature         0.039 +/- 0.002
humidity            0.031 +/- 0.001
pressure            0.022 +/- 0.001
weather             0.017 +/- 0.001
Month               0.017 +/- 0.001
wind_speed          0.015 +/- 0.001
DayOfWeek           0.014 +/- 0.001
"""