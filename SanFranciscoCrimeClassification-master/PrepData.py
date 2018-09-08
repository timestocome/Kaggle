#!/python


import numpy as np
import pandas as pd

from datetime import datetime
from sklearn import cross_validation

import matplotlib.pyplot as plt
import random

import LoadData



# change categories to numbers,
# scale location
# parse date into min, hour, month, day, year
def cleanData():
    
    # load up files from disk
    training_data, kaggle_data = LoadData.load_data()    
    features_in = ['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address' 'X', 'Y']
    

    # break dates into month, day, year, day of week, hour 
    # categorize category, month, day, year, dow, hour, district
    # scale lat (y), long(x)
    training_data['Year'] = pd.DatetimeIndex(training_data['Dates']).year - 2000
    training_data['Month'] = pd.DatetimeIndex(training_data['Dates']).month
    training_data['Day'] = pd.DatetimeIndex(training_data['Dates']).day
    training_data['Hour'] = pd.DatetimeIndex(training_data['Dates']).hour
    training_data['Minute'] = pd.DatetimeIndex(training_data['Dates']).minute

    kaggle_data['Year'] = pd.DatetimeIndex(kaggle_data['Dates']).year - 2000
    kaggle_data['Month'] = pd.DatetimeIndex(kaggle_data['Dates']).month
    kaggle_data['Day'] = pd.DatetimeIndex(kaggle_data['Dates']).day
    kaggle_data['Hour'] = pd.DatetimeIndex(kaggle_data['Dates']).hour
    kaggle_data['Minute'] = pd.DatetimeIndex(kaggle_data['Dates']).minute


    # cast date as unix time
    training_data['UnixTime'] = (pd.DatetimeIndex(training_data['Dates'])).astype(np.int64) / 10000000000
    kaggle_data['UnixTime'] = (pd.DatetimeIndex(kaggle_data['Dates'])).astype(np.int64) / 10000000000

   
    # day of week to number
    sorted_days = ('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday')
    def dayOfWeekNumber(d):
        return sorted_days.index(d)
    training_data['DayNumber'] = training_data['DayOfWeek'].apply(dayOfWeekNumber)
    kaggle_data['DayNumber'] = kaggle_data['DayOfWeek'].apply(dayOfWeekNumber)
    
    
    # set up an id number for each category from alphabetical list
    # add to training_data
    categories = pd.unique(training_data['Category'])
    sorted_categories = (np.sort(categories)).tolist()

    def categoryNumber(category):
        return sorted_categories.index(category)
    training_data['CategoryNumber'] = training_data['Category'].apply(categoryNumber)
    # no categories for validation data, that's what we're trying to figure out

    
    # scale lat and long
    def scaleLat(lat):
        return lat - 37.0
    training_data['ScaledLatitude'] = training_data['Y'].apply(scaleLat) 
    kaggle_data['ScaledLatitude'] = kaggle_data['Y'].apply(scaleLat)
    
    def scaleLong(long):
        return long + 122.0
    training_data['ScaledLongitude'] = training_data['X'].apply(scaleLong)
    kaggle_data['ScaledLongitude'] = kaggle_data['X'].apply(scaleLong)
    
    
    districts = pd.unique(training_data['PdDistrict'])
    sorted_districts = (np.sort(districts)).tolist()
    
    def districtNumber(district):
        return sorted_districts.index(district)
    training_data['DistrictNumber'] = training_data['PdDistrict'].apply(districtNumber)
    kaggle_data['DistrictNumber'] = kaggle_data['PdDistrict'].apply(districtNumber)
    
    
    # add output array for validation set just for convience 
    kaggle_data['CategoryNumber'] = 0


    # split inputs from outputs
    features = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'DayNumber', 'DistrictNumber', 'ScaledLatitude', 'ScaledLongitude']
    training_x = training_data[features]
    training_y = training_data['CategoryNumber']
    kaggle_x = kaggle_data[features]
    kaggle_y = kaggle_data['CategoryNumber']
    

    # create a testing and validation set from the training_data
    x_train, x_split, y_train, y_split = cross_validation.train_test_split(training_x, training_y, test_size=0.2)
    x_test, x_validate, y_test, y_validate = cross_validation.train_test_split(x_split, y_split, test_size=0.5)
    
    training_set = zip(x_train, y_train)
    kaggle_set = zip(kaggle_x, kaggle_y)
    validation_set = zip(x_validate, y_validate)
    test_set = zip(x_test, y_test)
    
    print("length train", len(y_train))
    print("length test", len(y_test))
    print("length validate", len(y_validate))
    print("length kaggle", len(kaggle_y))
   

    return training_set, validation_set, test_set, kaggle_set
    
