#!/python

import numpy as np
import pandas as pd
import pickle

from datetime import datetime
from sklearn import cross_validation

from scipy import stats
import matplotlib.pyplot as plt

import LoadData



# load up files from disk
training_data, kaggle_data = LoadData.load_data()    
  

#############################################################################################################
# get an idea of how many locations are in db
#unique_addresses = pd.unique(training_data['Address'])
#print(len(unique_addresses))
def intersections(a):
    if 'Block' in a: return 0
    else:  return 1
    
training_data['Intersection'] = training_data['Address'].apply(intersections)

    
features_in = ['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address' 'X', 'Y', 'Location', 'Intersection']

    
    
       
# break dates into month, day, year, day of week, hour 
# categorize category, month, day, year, dow, hour, district
# scale lat (y), long(x)
training_data['Year'] = (pd.DatetimeIndex(training_data['Dates']).year - 2000) 
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
training_data['DayNumber'] = (training_data['DayOfWeek'].apply(dayOfWeekNumber)) 

      
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
training_data['DistrictNumber'] = (training_data['PdDistrict'].apply(districtNumber))


    #shuffle data
#training_data = training_data.iloc[np.random.permutation(len(training_data))]     
    



####################################################################################
#print(training_data.head())
features = ['X', 'Y', 'Intersection', 'UnixTime', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayNumber', 'DistrictNumber']


totalCrimes = len(training_data)
crimesByCategoryCount = training_data['CategoryNumber'].value_counts()

############################################################################################################
# separate data
############################################################################################################
warrants = training_data[training_data.Category == 'WARRANTS']
otherOffenses = training_data[training_data.Category == 'OTHER OFFENSES']
larcenyTheft = training_data[training_data.Category == 'LARCENY/THEFT']
vehicleTheft = training_data[training_data.Category == 'VEHICLE THEFT']
vandalism = training_data[training_data.Category == 'VANDALISM']
noncriminal = training_data[training_data.Category == 'NON-CRIMINAL']
robbery = training_data[training_data.Category == 'ROBBERY']
assault = training_data[training_data.Category == 'ASSAULT']
weapons = training_data[training_data.Category == 'WEAPON LAWS']
burglary = training_data[training_data.Category == 'BURGLARY']
suspicious = training_data[training_data.Category == 'SUSPICIOUS OCC']
drunkeness = training_data[training_data.Category == 'DRUNKENNESS']
forgery = training_data[training_data.Category == 'FORGERY/COUNTERFEITING']
drug = training_data[training_data.Category == 'DRUG/NARCOTIC']
stolenProperty = training_data[training_data.Category == 'STOLEN PROPERTY']
secondary = training_data[training_data.Category == 'SECONDARY CODES']
trespass = training_data[training_data.Category == 'TRESPASS']
missing = training_data[training_data.Category == 'MISSING PERSON']
fraud = training_data[training_data.Category == 'FRAUD']
kidnapping = training_data[training_data.Category == 'KIDNAPPING']
runaway = training_data[training_data.Category == 'RUNAWAY']
dui = training_data[training_data.Category == 'DRIVING UNDER THE INFLUENCE']
sexOffenseForcible = training_data[training_data.Category == 'SEX OFFENSES FORCIBLE']
prostitution = training_data[training_data.Category == 'PROSTITUTION']
disorderly = training_data[training_data.Category == 'DISORDERLY CONDUCT']
arson = training_data[training_data.Category == 'ARSON']
family = training_data[training_data.Category == 'FAMILY OFFENSES']
liquor = training_data[training_data.Category == 'LIQUOR LAWS']
bribery = training_data[training_data.Category == 'BRIBERY']
embezzlement = training_data[training_data.Category == 'EMBEZZLEMENT']
suicide = training_data[training_data.Category == 'SUICIDE']
loitering = training_data[training_data.Category == 'LOITERING']
sexOffenseNonForcible = training_data[training_data.Category == 'SEX OFFENSES NON FORCIBLE']
extortion = training_data[training_data.Category == 'EXTORTION']
gambling = training_data[training_data.Category == 'GAMBLING']
checks = training_data[training_data.Category == 'BAD CHECKS']
trea = training_data[training_data.Category == 'TREA']
recoveredVehicle = training_data[training_data.Category == 'RECOVERED VEHICLE']
pornography = training_data[training_data.Category == 'PORNOGRAPHY/OBSCENE MAT']




x = training_data.UnixTime
y = training_data.CategoryNumber * 2.0
   
figure = plt.figure(figsize=(20, 10))
figure.patch.set_facecolor('white')
plt.plot(x, y, 'ro')
x_min = min(x)
x_max = max(x)
y_min = min(y)
y_max = max(y)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.axis('off')
plt.title('Frequency, Crime category 0 at bottom, 38 is top row, x-axis time stamp')
    
plt.show()
  


