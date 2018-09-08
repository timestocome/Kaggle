#!/python

import numpy as np
import pandas as pd
import pickle

from datetime import datetime

import LoadData


# starter code
# http://blog.districtdatalabs.com/time-maps-visualizing-discrete-events-across-many-timescales




# load up files from disk
training_data, kaggle_data = LoadData.load_data()    
  

#############################################################################################################
features_in = ['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address' 'X', 'Y', 'Location', 'Intersection']

    
       
# set up an id number for each category from alphabetical list
# add to training_data
categories = pd.unique(training_data['Category'])
sorted_categories = (np.sort(categories)).tolist()

def categoryNumber(category):
    return sorted_categories.index(category)
training_data['CategoryNumber'] = training_data['Category'].apply(categoryNumber)
    
  


# cast date as unix time
training_data['UnixTime'] = (pd.DatetimeIndex(training_data['Dates'])).astype(np.int64) // 10 **9
training_data['UnixTime'] = training_data['UnixTime'] 




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


	

##############################################################################
# Time plot, fast and steady bottom left
# slow steady top right
# slowing down top left
# speeding up bottom right
# fast steady bottom left 

import matplotlib.pylab as plt




# calculate time differences:
min_time_stamp = min(suicide['UnixTime'])
sorted_by_time = suicide.sort_values(['Dates'], ) 
times_of_crimes = np.array(sorted_by_time['UnixTime'])
diffs = np.array([times_of_crimes[i] - times_of_crimes[i-1] for i in range(1,len(times_of_crimes))])
print(max(diffs), min(diffs))

seconds_in_year = 60 * 60 * 24 * 365
diffs_less_than_one_year =  diffs[diffs < seconds_in_year]
max_days = max(diffs) // (60 * 60 * 24)
max_diffs = max(diffs)

xcoords = diffs_less_than_one_year[:-1] # all differences except the last
ycoords = diffs_less_than_one_year[1:] # all differences except the first


plt.plot(xcoords, ycoords, 'b.') # make scatter plot with blue dots
plt.title('Time between suicide (days) 0 to %d' % (max_days))
plt.xlabel('Time before next')
plt.ylabel('Time after last')
#plt.savefig('timemap.png')
plt.show()


import scipy.ndimage as ndi

Nside=256 # this is the number of bins along x and y for the histogram
width=8 # the width of the Gaussian function along x and y when applying the blur operation

H = np.zeros((Nside,Nside)) # a 'histogram' matrix that counts the number of points in each grid-square

max_diff = np.max(diffs) # maximum time difference

x_heat = (Nside-1)*xcoords/max_diff # the xy coordinates scaled to the size of the matrix
y_heat = (Nside-1)*ycoords/max_diff # subtract 1 since Python starts counting at 0, unlike Fortran and R

for i in range(len(xcoords)): # loop over all points to calculate the population of each bin
    H[x_heat[i], y_heat[i]] += 1 # Increase count by 1
    #here, the integer part of x/y_heat[i] is automatically taken

H = ndi.gaussian_filter(H,width) # apply Gaussian blur
H = np.transpose(H) # so that the orientation is the same as the scatter plot

plt.imshow(H, origin='lower') # display H as an image
plt.show()

