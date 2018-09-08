#!/python


import numpy as np
import pandas as pd



#################################################################################################################
# create one hot submission to kaggle from predicted category number


# read in predictions, add index as 'Id', and predicted category number as 'Prediction'
submission_data = pd.read_csv("kaggle_predictions.txt", names=['Prediction'])
submission_data['Prediction'] = submission_data['Prediction'].astype(int)
submission_data.index.name = 'Id'


# check later to be sure we've removed all the dummy rows 
print("length before dummy rows added", len(submission_data))

print("?", submission_data.groupby('Prediction').size())


# I'm sure there is a better way to do this, but I haven't found it yet.
# Ugh. No hits for TREA in the predictions and that messes up one hot conversion
dummy_data = pd.DataFrame([33], columns=['Prediction'])     # add a TREA prediction
dummy_data1 = pd.DataFrame([22], columns=['Prediction'])
submission_data = submission_data.append(dummy_data)
submission_data = submission_data.append(dummy_data1)


# convert prediction to one hot array and add column headings
s = pd.get_dummies(submission_data['Prediction'])


# get header/column information
columns = ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION', 'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES', 'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS']

print("num columns", len(columns))

# remove dummy_data and check length of array matches what it was before dummy data added
s = s.ix[:len(s)-3]
print("length after dummy rows removed", len(s))


# give columns their proper titles
s.columns = columns


# print the submissiont to a text file.
s.to_csv("mySubmission.csv", index=True)

