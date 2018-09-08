#!/python



# starting point for this attempt
# http://efavdb.com/predicting-san-francisco-crimes/

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from scipy.stats import binned_statistic

 
 
 
def convertData(data):

    
    # intersections vs real addresses
    def intersections(a):
        if 'Block' in a:
            return 0
        else:
            return 1
     
    data['Corners'] = data['Address'].apply(intersections)
    
    # convert intersections to one hot
    intersection = pd.get_dummies(data.Corners)

    # Convert crime labels to numbers
    le_crime = preprocessing.LabelEncoder()
    crime = le_crime.fit_transform(data.Category)
    
    # Get binarized weekdays, districts, and hours.
    days = pd.get_dummies(data.DayOfWeek)
    
    district = pd.get_dummies(data.PdDistrict)
    
    hour = data.Dates.dt.hour
    hour = pd.get_dummies(hour) 
    
    months = data.Dates.dt.month 
    month = pd.get_dummies(months)
    
    years = data.Dates.dt.year
    years = years - 2003
    year = pd.get_dummies(years)
 
   
     # X is longitude, Y is latitude set ones outside city to median values
    data.loc[data.X > -122.0, 'X'] = data.X.median()
    data.loc[data.X < -123.0, 'X'] = data.X.median()
    data.loc[data.Y < 37.0, 'Y'] = data.Y.median()
    data.loc[data.Y > 38.0, 'Y'] = data.Y.median()
    
    min_x = min(data.X)
    max_x = max(data.X)
    dx = max_x - min_x

    min_y = min(data.Y)
    max_y = max(data.Y)
    dy = max_y - min_y
    
    def binX(x):
       return int((x - min_x) / dx * 10)
       
    data['XCat'] = data['X'].apply(binX)    
       
    def binY(y):
        return int((y - min_y) / dy * 10)   
    
    data['YCat'] = data['Y'].apply(binY)
    
    xLocation = pd.get_dummies(data['XCat'])
    yLocation = pd.get_dummies(data['YCat'])
    
        
    
    
    # Build new array
    new_data = pd.concat([hour, days, month, year, district, xLocation, yLocation, intersection], axis=1)
    new_data['crime'] = crime
 
    columns =  ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12',
                'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 
                'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 
                'm0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11',
                'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12',
                'pd0', 'pd1', 'pd2', 'pd3', 'pd4', 'pd5', 'pd6', 'pd7', 'pd8', 'pd9',
                  'x0', 'x1','x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                   'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10',
                   'i0', 'i1', 'crime' ]
    new_data.columns = columns
 
   
        
    return new_data
 
 
 
 
def get_data():

    # Load Data with pandas, and parse the first column into datetime
    train = pd.read_csv('train.csv', parse_dates = ['Dates'])

    test = pd.read_csv('test.csv', parse_dates = ['Dates'])
    test['Category'] = 'ARSON'      # place holder for converting
   
    # split data into categories and inputs and outputs    
    train_category = convertData(train)
    kaggle_category = convertData(test)

    training_y = train_category['crime']
    training_x = train_category[ ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12',
                'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 
                'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 
                'm0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11',
                'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12',
                'pd0', 'pd1', 'pd2', 'pd3', 'pd4', 'pd5', 'pd6', 'pd7', 'pd8', 'pd9',
                  'x0', 'x1','x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                   'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10',
                   'i0', 'i1']]

    kaggle_y = kaggle_category['crime']
    kaggle_x = kaggle_category[ ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12',
                'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 
                'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 
                'm0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11',
                'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12',
                'pd0', 'pd1', 'pd2', 'pd3', 'pd4', 'pd5', 'pd6', 'pd7', 'pd8', 'pd9',
                  'x0', 'x1','x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                   'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10',
                   'i0', 'i1']]


    # split into train, test, validate
    # create a testing and validation set from the training_data
    train_x, x_split, y_train, y_split = train_test_split(training_x, training_y, test_size=0.2)
    x_test, x_validate, y_test, y_validate = train_test_split(x_split, y_split, test_size=0.5)

 
    # package them up
    training_set = (train_x, y_train)
    validation_set = (x_validate, y_validate)
    test_set = (x_test, y_test)
    kaggle_set = (kaggle_x, kaggle_y) 
    
    
    
   
    return training_set, validation_set, test_set, kaggle_set
    
    
    
