#!/python


import numpy
import pandas as pd

import matplotlib.pyplot as plt
import random

import LoadData
    
training_data, validation_data = LoadData.load_data()    
    
print("Training examples", len(training_data))
print("Validation examples", len(validation_data))   
    
print(training_data.columns)  
features = ['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address' 'X', 'Y']


###################################################################################################################
# map data
####################################################################################################################
# http://stackoverflow.com/questions/14329691/covert-latitude-longitude-point-to-a-pixels-x-y-on-mercator-projection
def mercator_projection(latitude, longitude):
    scale = 100.0
   
    x = (longitude + 180.0) * (scale / 360.0)
    latitude_radians = latitude * numpy.pi/180.0
   
    y3 = numpy.log(numpy.tan(numpy.pi/4.0 + latitude_radians/2.0))
    y = (scale /2.0) - (scale * y3 / (2.0 * numpy.pi))
   
    return (x, y)
    
    

    
# matplotlib - chokes on large data sets    
def map_data(category, title):

    numberEntries = len(category)
    
    # map.plot fails over about 2000 points - grab a small sample
    if numberEntries > 2000:    
        category = category.sample(n=2000, replace=False)
    
    px, py = mercator_projection(category['Y'], category['X'])
   
    figure = plt.figure(figsize=(5, 5))
    figure.patch.set_facecolor('white')
    plt.scatter(px, py, s=0.5)
    plt.axis('equal')
    x_min = min(px)
    x_max = max(px)
    y_min = min(py)
    y_max = max(py)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis('off')
    plt.title(title)
    
    return plt
    #plt.show()
    
    

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




sunday = training_data[training_data.DayOfWeek == 'Sunday']
monday = training_data[training_data.DayOfWeek == 'Monday']
tuesday = training_data[training_data.DayOfWeek == 'Tuesday']
wednesday = training_data[training_data.DayOfWeek == 'Wednesday']
thursday = training_data[training_data.DayOfWeek == 'Thursday']
friday = training_data[training_data.DayOfWeek == 'Friday']
saturday = training_data[training_data.DayOfWeek == 'Saturday']





#####################################################################################################################
# static maps by category or by day of week
#####################################################################################################################

# list categories
#print("Category", pd.unique(training_data.Category))

"""
# plot a map for a category
category = forgery
map_data(category, "Forgery")    
"""


# plot map day of week
f, ax = plt.subplots(7, sharex=True)
ax[0] = map_data(sunday, "Sunday")
ax[1] = map_data(monday, "Monday")
ax[2] = map_data(tuesday, "Tuesday")
ax[3] = map_data(wednesday, "Wednesday")
ax[4] = map_data(thursday, "Thursday")
ax[5] = map_data(friday, "Friday")
ax[6] = map_data(saturday, "Saturday")
plt.show()







