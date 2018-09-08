#!/python


import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 5000)

from datetime import datetime
from sklearn import cross_validation

import matplotlib.pyplot as plt
import random

import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import seaborn as sns


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
    training_data['Year'] = (pd.DatetimeIndex(training_data['Dates']).year) 
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




    
    return (training_data)
    
    
    
############################################################################################
def stats():
    
    train = prepData()
    
    print(len(train))
    print(train.columns.values)
    print(train.dtypes)
    
    byCategory = train.groupby('CategoryNumber')
    
    # basics 
    #print(byCategory.describe())
    
    # correlation matrix ? strong ties between features
    #print(byCategory.corr())
    
   
    
    
    """
    histogramFeatures = ['DayNumber', 'DistrictNumber', 'Hour', 'Month', 'Year']
    for f in histogramFeatures:
        train[f].hist(by=train['Category'], sharex=True, sharey=False)
        plt.title(f)
        plt.show()
    """    
    
    """
    # feature to feature relationships
    scatter_matrix(t, alpha=0.2, figsize=(6,6), diagonal='kde')
    plt.show()     
    """
    
#stats()    