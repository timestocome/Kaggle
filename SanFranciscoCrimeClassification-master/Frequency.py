#!/python


import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 5000)

from datetime import datetime
from sklearn import cross_validation

import random
import matplotlib.pyplot as plt



import LoadData




# change categories to numbers,
# scale everything for use in NN
# parse date into min, hour, month, day, year
def prepData():
    
    # load up files from disk
    training_data, kaggle_data = LoadData.load_data()    
    features_in = ['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address' 'X', 'Y']
    

    # break dates into month, day, year, day of week, hour 
    # categorize category, month, day, year, dow, hour, district
    # scale lat (y), long(x)
    training_data['Year'] = (pd.DatetimeIndex(training_data['Dates']).year) - 2000
    training_data['Month'] = (pd.DatetimeIndex(training_data['Dates']).month)
    training_data['Day'] = (pd.DatetimeIndex(training_data['Dates']).day)
    training_data['Hour'] = (pd.DatetimeIndex(training_data['Dates']).hour)
    training_data['Minute'] = (pd.DatetimeIndex(training_data['Dates']).minute)

  
    


    # cast date as unix time
    training_data['UnixTime'] = (pd.DatetimeIndex(training_data['Dates'])).astype(np.int64) / 10000000000

   
    # day of week to number
    sorted_days = ('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday')
    def dayOfWeekNumber(d):
        return sorted_days.index(d)
    training_data['DayNumber'] = training_data['DayOfWeek'].apply(dayOfWeekNumber)
    
    
    # set up an id number for each category from alphabetical list
    # add to training_data
    categories = pd.unique(training_data['Category'])
    sorted_categories = (np.sort(categories)).tolist()

    def categoryNumber(category):
        return sorted_categories.index(category)
    training_data['CategoryNumber'] = training_data['Category'].apply(categoryNumber)
    
   
    
    districts = pd.unique(training_data['PdDistrict'])
    sorted_districts = (np.sort(districts)).tolist()
    
    def districtNumber(district):
        return sorted_districts.index(district)
    training_data['DistrictNumber'] = training_data['PdDistrict'].apply(districtNumber)
    
    
    # X is longitude, Y is latitude set ones outside city to median values
    training_data.loc[training_data.X > -122.0, 'X'] = training_data.X.median()
    training_data.loc[training_data.X < -123.0, 'X'] = training_data.X.median()
    training_data.loc[training_data.Y < 37.0, 'Y'] = training_data.Y.median()
    training_data.loc[training_data.Y > 38.0, 'Y'] = training_data.Y.median()

    # center and scale X, Y
    x_median = training_data['X'].median()
    y_median = training_data['Y'].median()
    
    training_data['X'] = training_data['X'].subtract(x_median) * 1000.0
    training_data['Y'] = training_data['Y'].subtract(y_median) * 1000.0
    
    #training_data['X'] = training_data['X'].astype(int)
    #training_data['Y'] = training_data['Y'].astype(int)
    
    return (training_data)
    
    
    
############################################################################################
trainingData = prepData()


# split into training, testing, validation
features = ['X', 'Y', 'DistrictNumber', 'Year', 'Month', 'DayNumber', 'Hour', 'UnixTime']

print(min(trainingData['X']), max(trainingData['X']))

input = trainingData[features]
output = trainingData['CategoryNumber']


X_train, testX, y_train, testY = cross_validation.train_test_split(input, output, test_size=0.2)
X_test, X_validate, y_test, y_validate = cross_validation.train_test_split(testX, testY, test_size=0.5)

#print(len(X_train), len(X_test), len(X_validate))


############################################################################################
# frequency of each crime

trainingData = trainingData.sort_values('UnixTime')

# crime as % of total crimes
# crime as % of year, month, dayOfWeek, hour
# crime as % of location incidents

totalCrimes = len(trainingData)
crimesByCategory = trainingData['CategoryNumber'].value_counts()

crimesByMonth = trainingData['Month'].value_counts()
crimesByMonthCategory = trainingData.groupby(['Month', 'CategoryNumber']).count()

crimesByYear = trainingData['Year'].value_counts()
crimesByYearCategory = trainingData.groupby(['Year', 'CategoryNumber']).count()

crimesByHour = trainingData['Hour'].value_counts()
crimesByHourCategory = trainingData.groupby(['Hour', 'CategoryNumber']).count()

crimesByDay = trainingData['DayNumber'].value_counts()
crimesByDayByCategory = trainingData.groupby(['DayNumber', 'CategoryNumber']).count()

trainingData['X'] = trainingData['X'].astype(int)
trainingData['Y'] = trainingData['Y'].astype(int)
crimesByLocation = trainingData.groupby(['CategoryNumber', 'X', 'Y']).count()



print("Crimes by categoryNumber", crimesByCategory/totalCrimes)

"""
print("Crimes by month", crimesByMonth/totalCrimes)
print("Crimes by year", crimesByYear/totalCrimes)
print("Crimes by hour", crimesByHour/totalCrimes)
print("Crimes by Day of week", crimesByDay/totalCrimes)
print("Crimes by month by category", crimesByMonthCategory/totalCrimes)
print("Crimes by year by category", crimesByYearCategory/totalCrimes)
print("Crimes by hour by category", crimesByHourCategory/totalCrimes)
print("Crimes by day of week by category", crimesByDayByCategory/totalCrimes)
"""
"""
Crimes by categoryNumber in training data
16    0.199192
21    0.143707
20    0.105124
1     0.087553
7     0.061467
36    0.061251
35    0.050937
37    0.048077
4     0.041860
32    0.035777
19    0.029599
25    0.026194
13    0.018996
12    0.012082
27    0.011372
38    0.009743
23    0.008523
34    0.008343
30    0.005171
28    0.004997
5     0.004920
8     0.004874
24    0.003574
15    0.002666
6     0.002583
26    0.002216
17    0.002167
0     0.001723
18    0.001395
9     0.001328
31    0.000579
11    0.000559
2     0.000462
3     0.000329
10    0.000292
29    0.000169
14    0.000166
22    0.000025
33    0.000007
"""