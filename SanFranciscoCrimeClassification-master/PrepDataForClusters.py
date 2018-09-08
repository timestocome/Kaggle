#!/python

import numpy as np
import pandas as pd
import pickle

from datetime import datetime
from sklearn import cross_validation
from sklearn.cluster import Birch, KMeans



############################################################################
# read in csv files
import LoadData

# load up files from disk
training_data, kaggle_data = LoadData.load_data()    

############################################################################
"""
# cluster locations
def clusterLocations():

    t_location = training_data[['X', 'Y']]
    k_location = kaggle_data[['X', 'Y']]
    
    clf = KMeans(n_clusters=640)    # 23,104 unique addresses in training set
    clf.fit(t_location)
   
    
    training_data['Location'] = clf.predict(t_location)
    kaggle_data['Location'] = clf.predict(k_location)
"""    
    
##############################################################################
# intersections vs real addresses

def intersections(a):

    # if contains 'Block'
    # if contains '/'
    if 'Block' in a:
        return 0
    elif '/' in a:
        return 1
    else:
        return 2




#############################################################################################################
# change categories to numbers,
# scale everything for use in NN
# parse date into min, hour, month, day, year
def prepData():
   
    # get an idea of how many locations are in db
    #unique_addresses = pd.unique(training_data['Address'])
    #print(len(unique_addresses))
    
    #clusterLocations()
    training_data['Intersection'] = training_data['Address'].apply(intersections)
    kaggle_data['Intersection'] = kaggle_data['Address'].apply(intersections)
    
    all_data = training_data.append(kaggle_data)
    unique_addresses = pd.unique(all_data.Address)
    uniqueAddress = unique_addresses.tolist()
    
    features_in = ['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address' 'X', 'Y', 'Intersection']

    
    
    def address_to_number(a):
        return uniqueAddress.index(a)
    training_data['A'] = training_data.Address.apply(address_to_number)
    kaggle_data['A'] = kaggle_data.Address.apply(address_to_number)
    
    
    
    # break dates into month, day, year, day of week, hour 
    # categorize category, month, day, year, dow, hour, district
    # scale lat (y), long(x)
    training_data['Year'] = (pd.DatetimeIndex(training_data['Dates']).year - 2000) 
    training_data['Month'] = (pd.DatetimeIndex(training_data['Dates']).month) 
    training_data['Day'] = (pd.DatetimeIndex(training_data['Dates']).day) 
    training_data['Hour'] = (pd.DatetimeIndex(training_data['Dates']).hour) 
    training_data['Minute'] = (pd.DatetimeIndex(training_data['Dates']).minute) 

    kaggle_data['Year'] = (pd.DatetimeIndex(kaggle_data['Dates']).year - 2000) 
    kaggle_data['Month'] = (pd.DatetimeIndex(kaggle_data['Dates']).month) 
    kaggle_data['Day'] = (pd.DatetimeIndex(kaggle_data['Dates']).day) 
    kaggle_data['Hour'] = (pd.DatetimeIndex(kaggle_data['Dates']).hour)
    kaggle_data['Minute'] = (pd.DatetimeIndex(kaggle_data['Dates']).minute) 
    


    # cast date as unix time
    training_data['UnixTime'] = (pd.DatetimeIndex(training_data['Dates'])).astype(np.int64) / 10000000000
    kaggle_data['UnixTime'] = (pd.DatetimeIndex(kaggle_data['Dates'])).astype(np.int64) / 10000000000

   
    # day of week to number
    sorted_days = ('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday')
    def dayOfWeekNumber(d):
        return sorted_days.index(d)
    training_data['DayNumber'] = (training_data['DayOfWeek'].apply(dayOfWeekNumber)) 
    kaggle_data['DayNumber'] = (kaggle_data['DayOfWeek'].apply(dayOfWeekNumber)) 
    
    
    
    
    # set up an id number for each category from alphabetical list
    # add to training_data
    categories = pd.unique(training_data['Category'])
    sorted_categories = (np.sort(categories)).tolist()

    def categoryNumber(category):
        return sorted_categories.index(category)
    training_data['CategoryNumber'] = training_data['Category'].apply(categoryNumber)
    
    # no categories for validation data, that's what we're trying to figure out
    # add output array for validation set just for convience 
    kaggle_data['CategoryNumber'] = 0
    
    
    
    
    districts = pd.unique(training_data['PdDistrict'])
    sorted_districts = (np.sort(districts)).tolist()
    
    def districtNumber(district):
        return sorted_districts.index(district)
    training_data['DistrictNumber'] = (training_data['PdDistrict'].apply(districtNumber))
    kaggle_data['DistrictNumber'] = (kaggle_data['PdDistrict'].apply(districtNumber)) 
    

  
    
 
# save the dataframes to disk as pickle files    
def save_cluster_data():
    # save as pickle files
    pickle_out = open('trainingClusters.pkl', 'wb')
    pickle.dump(training_data, pickle_out)
    pickle_out.close()

    pickle_out = open('validationClusters.pkl', 'wb')
    pickle.dump(kaggle_data, pickle_out)
    pickle_out.close()
    
    
    
def load_cluster_data():   
 
   # read in pickle files
    file = 'trainingClusters.pkl'
    with open(file, 'rb') as f:
        training_data = pickle.load(f)
        f.close()

    file = 'validationClusters.pkl'
    with open(file, 'rb') as f:
        validation_data = pickle.load(f)
        f.close()
        
        
        
    #shuffle data
    training_data = training_data.iloc[np.random.permutation(len(training_data))]     
    
    return training_data, kaggle_data

    """
    # split inputs from outputs
    features = ['Year', 'Month', 'Day', 'Hour', 'DayNumber', 'DistrictNumber', 'X', 'Y', 'Intersection']
    training_x = training_data[features]
    training_y = training_data['CategoryNumber']
    
    kaggle_x = validation_data[features]
    kaggle_y = validation_data['CategoryNumber']
    
   

    # create a testing and validation set from the training_data
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(training_x, training_y, test_size=0.1)
    
   
    
    # package them up
    training_set = (x_train, y_train)
    test_set = (x_test, y_test)
    kaggle_set = (kaggle_x, kaggle_y) 
    
    
   
    return training_set, test_set, kaggle_set
    """    
###############################################################################################    
#prepData() 
#save_cluster_data()
#training_set,  kaggle_set = load_cluster_data()
    
