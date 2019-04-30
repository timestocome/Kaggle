
# http://github.com/timestocome





# Pandas is included in Anaconda or you can install it separately 

# Python Data Analysis Library
# http://pandas.pydata.org/

# In 2008, pandas development began at AQR Capital Management. 
# By the end of 2009 it had been open sourced, and is actively 
# supported today by Lambda Foundry, a company focused on building 
# high-performance and high-productivity tools for Finance. 
# BSD License


# Pandas documentation is extensive but not always helpful, StackoverFlow is better
# Panda's Cheat Sheet
# https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf 





# Kaggle Data Analysis Competitions, Data, Tutorials
# https://www.kaggle.com/c/titanic

# Simple demo of reading in csv files, combining, cleaning, and writing out cleaned data 
# using the Titanic data set from Kaggle




import pandas as pd

# Pandas truncates horz and vert middle of large tables, this forces full printing
# pd.set_option('display.max_rows', 1000)
# pd.set_option('display.max_columns', 20)



# misc libraries
import re 										# regex for text parsing
import math 									# math for math 
from collections import Counter 				# used to count values for use in dictionaries etc 



######################################################################################################
# read in csv files and put them into dataframes 
# Pandas does everything in dataframes ( 2D ) or series ( 1D )


# Pandas can read huge files and just about every file type 
# Pandas io tools
# CSV, Excel, HDF, SQL, MSGPack, GBQ, STATA, SAS, Clipboard, Pickel

# http://pandas-docs.github.io/pandas-docs-travis/io.html#hdf5-pytables  High Performance read/write
######################################################################################################
print("####################################################################################################")

def readData():
    
    # read in training and test files in dataframes
    # use the first column ("PassengerId") as the index column
    training_data = pd.read_csv("train.csv", index_col=0)
    testing_data = pd.read_csv("test.csv", index_col=0)
    
    # combine files for some parsing
    testing_data['Survived'] = -1               # add a column to the dataframe

    frames = [testing_data, training_data]      
    all_data = pd.concat(frames)

    return training_data, testing_data, all_data 

train, test, data = readData()


print("First few rows of dataframe")
print(train.head(n=5))                           # print top rows, can also print last rows with tail 





########################################################################################################
# check data 
########################################################################################################
print("####################################################################################################")
# get data type for each column, need to change object to numbers
print("Data types of the columns")
print(data.dtypes)                                

print("####################################################################################################")
print("A row of data")
print(data.ix[42])

print("####################################################################################################")
print("A Column of data")
print(data['Fare'])


print("####################################################################################################")
print("Count rows")
print(len(data))


print("####################################################################################################")
# how many rows contain nulls in each column  
print("NaN/Null values?")
print(data.isnull().sum())                        


print("####################################################################################################")
# Pandas contains all the basic statistical tools you'd usually use
# http://pandas.pydata.org/pandas-docs/stable/api.html#api-dataframe-stats 
print("Describe Data")
print(data.describe())                            # basic stats mean, std, min, max...


print("####################################################################################################")
# list all columns 
features = list(data.columns.values)
print("Column names")
print('Columns', features)


#########################################################################################################
# clean data 
#########################################################################################################
print("####################################################################################################")
###### rename some columns to make them clearer 
new_names = {'Parch' : 'ParentsAndOrChildren',
            'Pclass' : 'Class',
            'SibSp' : 'SiblingsAndOrSpouse'}
data.rename(columns = new_names, inplace=True)
print ("New Column Names")
features = list(data.columns.values)
print(features)


print("####################################################################################################")
###### remove low information columns 

# cabin has 1014 null values out of 1309 samples in the data 
data = data.drop('Cabin', axis=1)  # 0 for rows, 1 for columns


# drop ticket. There is some data about the cabin number/deck and some cabins/decks had a better but it's not a lot of info 
data = data.drop('Ticket', axis=1)


print ("Dropped Cabin and Ticket columns")
features = list(data.columns.values)
print(features)


###### convert text to categories ['Embarked', 'Name', 'Sex', 'Ticket']
print("####################################################################################################")
# Embarked has 3 locations, create 3 columns to store ( one-hot vector )
# could store this all in one vector in the column but it's easier to feed into neural nets, gradient boosted trees
# etc if we pull it out and give each value it's own column.
# this also makes it easier to drop a column of values later, not all locations might be relevent 
embarked_values = data['Embarked'].value_counts()
print("Embarked values and counts")
print(embarked_values)

# 3 possible locations for embarking, S makes up ~70% so fill our 2 nulls with S 
data['Embarked'] = data['Embarked'].fillna('S')
print("Embarked values and counts after filling both NA values with S")
embarked_values = data['Embarked'].value_counts()
print(embarked_values)

# convert the 3 locations to 3 columns in the dataframe and drop the old column
# add 3 new columns, fill them all with zeroes
data['Embarked_S'] = 0
data['Embarked_C'] = 0
data['Embarked_Q'] = 0

# put a one in the column that the passenger/crew embarked at
data.loc[data['Embarked'] == 'S', 'Embarked_S'] = 1
data.loc[data['Embarked'] == 'C', 'Embarked_C'] = 1
data.loc[data['Embarked'] == 'Q', 'Embarked_Q'] = 1

print("Check embarking location properly changed to new columns")
print(data.head(5))             # check everything worked

# delete the original 'Embarked' column 
data = data.drop('Embarked', axis=1)



print("####################################################################################################")
# Sex, convert from male/female to 1/0
print("Convert Sex column from male/female to 1/0")
data.loc[data["Sex"] == "male", "Sex"] = 1
data.loc[data["Sex"] == "female", "Sex"] = 0

# convert sex from object type to int
data['Sex'] = data['Sex'].astype('int32')

print(data.head(5))




print("####################################################################################################")
# Name, pull title out of the name, ditch the rest of the name 
# function to regex parse titles ( space, capital letter, lower case letter(s), period
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:    return(title_search.group(1))
    else:   return 'No title'

# apply function to every row in the 'Name' column and store result in new 'Title' column
data["Title"] = data["Name"].apply(get_title)

titles = data['Title'].value_counts()
print("Titles:")
print(titles)



# convert title column to an int to use as a category
# or better still, convert to a set of one hot vectors as done with Embarked data
def convert_title(title):
	if title == 'Mr': return 1
	elif title == 'Miss': return 2
	elif title == 'Mrs': return 3
	elif title == 'Master': return 4
	elif title == 'Dr': return 5
	elif title == 'Rev': return 6
	elif title == 'Col': return 7
	elif title == 'Major': return 8
	elif title == 'Mlle': return 9
	elif title == 'Ms': return 10
	elif title == 'Sir': return 11
	elif title == 'Don': return 12
	elif title == 'Capt': return 13
	elif title == 'Lady': return 14
	elif title == 'Dona': return 15
	elif title == 'Jonkheer': return 16
	elif title == 'Countess': return 17
	elif title == 'Mme': return 18
	else: return 19
	
data["Title"] = data["Title"].apply(convert_title)


# check the counts for each title int match our count of the text values
titles = data['Title'].value_counts()
data['Title'] = data['Title'].astype('category')

print("Titles:")
print(titles)


print("####################################################################################################")
# Age has 263 nulls, let's try using the median age by title to get closer than just a median for all passengers/crew
means_by_age = data.groupby("Title").median()       # median age by title
means_by_age = means_by_age["Age"]                  # create a lookup table with the age by title

# create a new dataframe with all the rows missing ages, then put a title number in each 
missing_ages = data.loc[data.Age.isnull(), 'Title']	                       
data.loc[data.Age.isnull(), 'Age'] = means_by_age.loc[missing_ages].values # replace title with calculated age



print("####################################################################################################")
# one missing fare - use the median fare by class to fill it in
fares_by_class = data.groupby("Class").median()     # median fare by class
fares_by_class = fares_by_class['Fare']             # create lookup table with class - median fare by class
print("Average fare by class", fares_by_class)

# same as ages but this time we're iterating over each row just to show how to do so
# http://pandas.pydata.org/pandas-docs/stable/indexing.html#different-choices-for-indexing
# see pandas notes for loc, iloc, ix 
for index, row in data.iterrows():
		
		if math.isnan(data.loc[index, "Fare"]): # if data.loc[index, "Fare"] == 0:
			c = row["Class"]
			f = fares_by_class[c]
			data.loc[index, "Fare"] = f


print("####################################################################################################")
print("Min/Max Fares: ", data.Fare.min(), data.Fare.max())

# bin data to smooth outliers
# 3 Classes - try 6 bins min 0, max 512, meds: 8.05, 15.05, 60.0
# 0-8, 8-11.5, 11.5-15, 15-37.5, 37.5-60, 60-512
# Note: edges <> so go a bit high/low to cover ==
bins = [-1, 8, 11.5, 15, 37.5, 60, 513]
labels = ['Lower 3rd', 'Upper 3rd', 'Lower 2nd', 'Upper 2nd', 'Lower 1st', 'Upper 1st']
data['fare_categories'] = pd.cut(data['Fare'], bins, labels=labels)



print("####################################################################################################")
print("No built in scaling but can scale to -1..1 ")
data['scaled_fares'] = (data['Fare'] - data['Fare'].mean()) / (data['Fare'].max() - data['Fare'].min())


###########################################################################################################
# use family names?
###########################################################################################################
print("####################################################################################################")
# create new columns by parsing out family name and counting family members

def get_family_id(row):
	return row["Name"].split(",")[0]

family_id = data.apply(get_family_id, axis=1)
print("Family names:")
print(family_id)

# use counter to create a dictionary with family names and head count
family_member_count = Counter(family_id)
print("Family members count:")
print(family_member_count)

# add surnames to dataframe
data["FamilyID"] = data.apply(get_family_id, axis=1)


print("####################################################################################################")

# remove duplicate family names
names = set(family_id)
family_names = list(names)
family_names.sort()
print("Family names", family_names)


# add number of family members on board to dataframe
data['FamilyMemberCount'] = data['FamilyID'].map(family_member_count)
print("Family members onboard with passenger")
print(data['FamilyMemberCount'])



###########################################################################################################
# check data
###########################################################################################################
print("####################################################################################################")
print("Verify data:")
# mark Class, Title as categorical data
data['Class'] = data['Class'].astype('category')

print("####################################################################################################")
print("Basic stats")
print(data.describe())

print("####################################################################################################")
print("Data types")
print(data.dtypes)

print("####################################################################################################")
print("Correlations with survival")
print(data.corr()['Survived'])

###########################################################################################################
# split training from hold out (test)
###########################################################################################################
print("####################################################################################################")
print("Original Train %d, Test %d" % (len(train), len(test)))

new_train = data[data['Survived'] > -1]
new_test = data[data['Survived'] == -1]

print(len(new_train), len(new_test))        # double check we split it correctly





###########################################################################################################
# split features from target
###########################################################################################################
features = list(data.columns.values)

target = ['Survived']                   # survivial is what we're attempting to predict
features.remove('Survived')             # remove target from features list

print("target", target)
print("features", features)

print("####################################################################################################")
print("Final Features:")
print("features", features)


##########################################################################################################
# write cleaned data back to disk as a data frame 
##########################################################################################################

# save to disk
new_train.to_csv("clean_training_data.csv")
new_test.to_csv("clean_testing_data.csv")



# quick check saved data okay
training_data = pd.read_csv("clean_training_data.csv", index_col=0)
testing_data = pd.read_csv("clean_testing_data.csv", index_col=0)


print("####################################################################################################")
print("Print number of samples saved in training data", len(training_data))
print("Print column list in saved training data", list(training_data.columns.values))



