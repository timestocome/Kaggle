#!/python

import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 5000)

from datetime import datetime
from sklearn import cross_validation

import random
import six
import math

import matplotlib.pyplot as plt
from matplotlib import colors

from pandas.tools.plotting import scatter_matrix




import LoadData
import PrepDataforStatistics



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
    
    
    
###################################################################################################################
# map data
####################################################################################################################
# http://stackoverflow.com/questions/14329691/covert-latitude-longitude-point-to-a-pixels-x-y-on-mercator-projection
def mercator_projection(latitude, longitude):
    scale = 100.0
   
    x = (longitude + 180.0) * (scale / 360.0)
    latitude_radians = latitude * np.pi/180.0
   
    y3 = np.log(np.tan(np.pi/4.0 + latitude_radians/2.0))
    y = (scale /2.0) - (scale * y3 / (2.0 * np.pi))
   
    return (x, y)
    
    

    
# matplotlib - chokes on large data sets    
def map_data():

    sunday = train[train.DayOfWeek == 'Sunday']
    monday = train[train.DayOfWeek == 'Monday']
    tuesday = train[train.DayOfWeek == 'Tuesday']
    wednesday = train[train.DayOfWeek == 'Wednesday']
    thursday = train[train.DayOfWeek == 'Thursday']
    friday = train[train.DayOfWeek == 'Friday']
    saturday = train[train.DayOfWeek == 'Saturday']


    # map.plot fails over about 2000 points - grab a small sample
    if len(sunday) > 2000:    
        sunday = sunday.sample(n=2000, replace=False)
    px0, py0 = mercator_projection(sunday['Y'], sunday['X'])
   
    if len(monday) > 2000:
        monday = monday.sample(n=2000, replace=False)
    px1, py1 = mercator_projection(monday['Y'], monday['X'])   
   
    if len(tuesday) > 2000:    
        tuesday = tuesday.sample(n=2000, replace=False)
    px2, py2 = mercator_projection(tuesday['Y'], tuesday['X'])
   
    if len(wednesday) > 2000:
        wednesday = wednesday.sample(n=2000, replace=False)
    px3, py3 = mercator_projection(wednesday['Y'], wednesday['X'])   
    
    if len(thursday) > 2000:    
        thursday = thursday.sample(n=2000, replace=False)
    px4, py4 = mercator_projection(thursday['Y'], thursday['X'])
   
    if len(friday) > 2000:
        friday = friday.sample(n=2000, replace=False)
    px5, py5 = mercator_projection(friday['Y'], friday['X'])   
   
    if len(saturday) > 2000:    
        saturday = saturday.sample(n=2000, replace=False)
    px6, py6 = mercator_projection(saturday['Y'], saturday['X'])
   
   
   
    figure = plt.figure(figsize=(15, 15))
    figure.patch.set_facecolor('white')
    
    plt.scatter(px0, py0, s=5.0, c='black', alpha=1.0)
    plt.scatter(px1, py1, s=4.5, c='turquoise', alpha=0.9)
    plt.scatter(px2, py2, s=4.0, c='red', alpha=0.8)
    plt.scatter(px3, py3, s=3.5, c='darkviolet', alpha=0.7)
    plt.scatter(px4, py4, s=3.0, c='orange', alpha=0.6)
    plt.scatter(px5, py5, s=2.5, c='lavender', alpha=0.5)
    plt.scatter(px6, py6, s=2.0, c='lightslategray', alpha=0.4)
    
    plt.axis('equal')
    
    x_min = min(px0)
    x_max = max(px0)
    y_min = min(py0)
    y_max = max(py0)
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis('off')
    
    plt.title("By day of week")
    
    plt.show()
    
    
# list categories
#print("Category", pd.unique(training_data.Category))

"""
# plot a map for a category
category = forgery
map_data(category, "Forgery")    
"""
    
    


################################################################################################
# misc plots
############################################################################################
def histogram():
    
    histogramFeatures = ['DayNumber', 'DistrictNumber', 'Hour', 'Month', 'Year']
    numDays = 7
    numDistricts = len(train.DistrictNumber.unique())
    numHours = 24
    numMonths = 12
    numYears = len(train.Year.unique())
    for f in histogramFeatures:
        train[f].hist(by=train['CategoryNumber'], rwidth=0.9, align='mid', histtype='bar')
        plt.title(f)
        plt.show()
    
        
def sequencePlot():    
        
    features = ['CategoryNumber']    
    categoryData = train.groupby(['CategoryNumber', 'DayNumber'])
    d = categoryData['CategoryNumber'].aggregate(np.sum)
    
  
    for i in range(1,38):
        x = np.array([0, 1, 2, 3, 4, 5, 6])
        y = np.array(d[i])

        plt.xticks( [0, 1, 2, 3, 4, 5, 6], ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
        plt.plot(x, y) 
        plt.title(categoryNames[i])
        plt.show()
       
    
    
def featureToFeature():    
  
    scatter_matrix(t, alpha=0.2, figsize=(6,6), diagonal='kde')
    plt.show()     
    

    
def scaledPolar():

    categoriesList = train.Category.values
    categoriesList = np.unique(categoriesList)
    features = ['X', 'Y']

    counts = []
    locations = []
    categories = []
    rs = []
    thetas = []

    colorsList = list(six.iteritems(colors.cnames))
    colorNames = [c[0] for c in colorsList]

    for c in categoriesList:
        d = train[train.Category == c]
        X = d[features]
        y = d[['CategoryNumber']]
    
   
        categories.append(c)
        locations.append(X.mean())
        counts.append(X.count())
    
        r = (X.X.mean()**2 + X.Y.mean()**2)**.5
        r = (r - 128.0) * 10.0
        rs.append(r)
    
        theta = math.degrees(math.atan(X.Y.mean()/X.X.mean()))
        theta = (theta + 17.0) * 10.0
        thetas.append(theta)
     
        ax = plt.subplot(111, projection='polar')
        c = plt.scatter(thetas, rs, c=colorNames, s=counts, cmap=plt.cm.hsv)
        c.title = categories
        c.set_alpha(0.75)
    
        plt.show()
 
 ################################################################################
 # number of crimes per time slice 
 ###############################################################################
# Time series plots

def yearlyPlots():
    # yearly
    

    for c in categoriesList:
        d = data[data.Category == c]
    
        year = d[['Year', 'Month']]
        yearTotals = year.groupby('Year').agg('count')
    
        plt.plot(yearTotals['Month'])
        plt.title("Yearly: %s" % c)
        plt.show()



def monthlyPlots():
 # monthly
    for c in categoriesList:
        d = data[data.Category == c]
    
        month = d[['Month', 'Day']]
        monthTotals = month.groupby('Month').agg('count')
    
    
        plt.plot(monthTotals['Day'])
        plt.title("Monthly: %s" % c)
        plt.show()



def dayOfMonthPlots():
    # day of month
    for c in categoriesList:
        d = data[data.Category == c]
    
        month = d[['Day', 'Hour']]
        monthTotals = month.groupby('Day').agg('count')
    
        plt.plot(monthTotals['Hour'])
        plt.title("Day of month: %s" % c)
        plt.show()   


def hourOfDayPlots():
    # hour of day
    for c in categoriesList:
        d = data[data.Category == c]
    
        month = d[['Day', 'Hour']]
        monthTotals = month.groupby('Hour').agg('count')
    
        plt.plot(monthTotals['Day'])
        plt.title("Hour of day: %s" % c)
        plt.show()    


def dayOfWeekPlots():
    # day of week
    for c in categoriesList:
        d = data[data.Category == c]
    
        month = d[['DayNumber', 'Day']]
        monthTotals = month.groupby('DayNumber').agg('count')
    
    
        plt.plot(monthTotals['Day'])
        plt.title("Day of week: %s" % c)
        plt.show()    


def minuteOfHourPlots():
    # minute of hour
    for c in categoriesList:
        d = data[data.Category == c]
    
        month = d[['Minute', 'Day']]
        monthTotals = month.groupby('Minute').agg('count')
    
        plt.plot(monthTotals['Day'])
        plt.title("Minute of hour: %s" % c)
        plt.show()

       
###############################################################################
# day of week by hour heat maps for a category 
################################################################################    

import pylab as pl 


train = prepData()
categoryNames = sorted(train.Category.unique())
print (categoryNames)

crimes = train[train.Category=='ARSON']

# build heat map
numDays = 7
numHours = 24
H = np.zeros((numDays, numHours))

for index, row in crimes.iterrows():
    H[row['DayNumber'], row['Hour']] += 1


pl.pcolor(H, cmap='OrRd')
pl.colorbar()
pl.title("Arson")

pl.show()



##############################################################################    

""" 
train = prepData()
categoryNames = sorted(train.Category.unique())


train = train.drop('Dates', 1)
#train = train.drop('Category', 1)
#train = train.drop('DayOfWeek', 1)
train = train.drop('PdDistrict', 1)
train = train.drop('Address', 1)
train = train.drop('Resolution', 1)
train = train.drop('Descript', 1)
    
#print(len(train))
#print(train.columns.values)
#print(train.dtypes)
  

# histogram()        
# featureToFeature()
# sequencePlot()
# map_data()
# scaledPolar()
"""

"""
features = ['UnixTime', 'Year', 'Month', 'Day', 'Hour', 'DayNumber']

data = PrepDataforStatistics.prepData()
categoriesList = data.Category.values
categoriesList = np.unique(categoriesList)

yearlyPlots()
monthlyPlots()
dayOfMonthPlots()
hourOfDayPlots()
dayOfWeekNumber()
minuteOfHourPlots()
"""