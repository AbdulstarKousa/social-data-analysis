####################################
# Week 06 Part 01: Lightning intro to machine learning
####################################



# What do we mean by a 'feature' in a machine learning model?
"""
Features are whatever inputs we provide to our model. [ref: DSFS ch11]
It could be a variable in the given data-set or an independent variables of the model we want to estimate.
"""


# What is the main problem with overfitting?'
"""
A usual risk in machine learning. 
Which happens when producing a model that performs well on the training data but generalizes poorly to any new data. 
This involves learning noise in the data. 
Or it could involve learning to identify specific inputs rather than whatever factors are actually predictive for the desired output.
[ref: DSFS ch11]
"""


# Explain the connection between the bias-variance trade-off and overfitting/underfitting?
"""
High bias and low variance typically correspond to underfitting
    For example: 
    - The degree 0 model DSFS ch11:
       -- It has High bias 
          since for any training set it predict 
          the sample mean that it is somehow 
          close to the population mean.
       -- It has low variance since for any training set 
          it will give the somehow 
          the exact model y = sample_mean 
          which is close to the population mean. 

Low bias and High variance typically correspond to overfitting
    For example: 
    - The degree 9 model from DSFS ch11:
        -- It has low bias:
           since it fit the training set perfectly
        -- It high variance:
           since any two training sets would likely give rise to very
           different models


[ref: DSFS ch11]
"""

# The Luke is for leukemia on page 145 in 
# the reading is a great example of why accuracy is not a good 
# measure in very unbalanced problems. 
# You know about the incidents dataset we've been working with. 
# Try to come up with a similar example based on the data we've been working with today.
"""
Not sure how to answer this question: 
Example from the data set where the accuracy is large but the precision and recall are low
"""








####################################
# Week 06 Part 2: Scikit-learn
####################################

"""
You can find an overview of Scikit-learn here 
https://scikit-learn.org/stable/tutorial/index.html.
"""



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

from quick_init import *
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