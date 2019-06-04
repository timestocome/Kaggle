

# https://github.com/timestocome
# https://www.kaggle.com/c/LANL-Earthquake-Prediction

# try a stacking a linear regression model using the output from the 
# SGBoost model to see if can tighten up predictions


################################################################################
# libraries
################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##################################################################################
# print settings
##################################################################################
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 1000)

np.set_printoptions(precision=10)



###################################################################################
# read in data created by cross_validate ( random forest)
# and xgboost
###################################################################################
print('Reading data.............................')


# read in raw data
rf = pd.read_csv('rf_train_predictions.csv')
xgb = pd.read_csv('xgboost_train_predictions.csv')
data = pd.read_csv('stats_data.csv', index_col=0)

print(data.head())

data['rf'] = rf
data['xgb'] = xgb



# remove last row, dropped during stats calculations
print('length data', len(data))
print('length rf', len(rf))
print('length xgb', len(xgb))
data = data[0:-2]


# trim things
data = data[['time_to_failure', 'xgb', 'rf']]


# check things
print(data.head(30))
print(data.tail(30))


# split features and targets apart
features = ['xgb', 'rf']
targets = ['time_to_failure']




# split into training and holdout data
# because it's a time series pull the last 20% or so instead of a random selection
x = data[features][0:7500]
y = data['time_to_failure'][0:7500]

x_holdout = data[features][7500:len(data)]
y_holdout = data['time_to_failure'][7500:len(data)]



print(data.columns.values)
print('********************************************************************************')
print('starting features')
print(features)
print('--------------------------------------------------------------------------------')
print('Samples %d' %(len(x)))




#########################################################################################
# import sklearn libraries for model
#########################################################################################
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score



print('Splitting data')
# split data up into features, targets, train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)




# create model
clf = LinearRegression()



# train model using cross validation 
print('training....')

cv_results = cross_validate(clf, x_train, y_train, cv=5, return_train_score=True)
print('-----------------cv_results--------------------------')
print('Train: %f, std %f' % (np.mean(cv_results['train_score']) * 100., np.std(cv_results['train_score'])))
print('Test: %f std %f' % (np.mean(cv_results['test_score']) * 100., np.std(cv_results['test_score'])))




###################################################################################################
# re-fit classifier on all training data except holdout to check feature values
####################################################################################################
clf = clf.fit(x_train, y_train)

# check regression accuracy on holdout data
predict_test = clf.predict(x_holdout) 
y_test = y_holdout.values 

print('=========================================================================')
mae = (np.abs(predict_test - y_test )).mean()
print('mae error on hold out data', mae)
print('actual max/min', y_test.max(), y_test.min())
print('predicted max/min', predict_test.max(), predict_test.min())
print('=========================================================================')



###############################################################################################
# plot predictions vs actual on training data and holdout data
##############################################################################################

predict_train = clf.predict(data[features]) 
predict_y = data['time_to_failure']

fig, ax = plt.subplots(figsize=(16, 16))
plt.plot(predict_train, c='r', label='Predicted')
plt.plot(predict_y, c='b', label='Actual', alpha=0.3)
plt.legend()
plt.grid(True)
plt.savefig('predictions.png')
plt.show()



#############################################################################################
# retrain on all training data, including holdout, then run submission data through
# read in test data and run it through clf
# save submission file to upload to kaggle
#############################################################################################
# refit on all training data
x = data[features]
y = data['time_to_failure']
clf.fit(x, y)


predict = clf.predict(x)

fig, ax = plt.subplots(figsize=(16, 16))
plt.plot(predict, c='r', label='Predicted')
plt.plot(y, c='b', label='Actual', alpha=0.3)
plt.legend()
plt.grid(True)
plt.savefig('predictions.png')
plt.show()



submission_data = pd.read_csv('stats_test.csv', index_col=0)
rf_submission = pd.read_csv('rf_submission.csv', index_col=0)
xgb_submission = pd.read_csv('xgboost_submission.csv', index_col=0)

test_data = xgb_submission
test_data.columns = ['xgb']
test_data['rf'] = rf_submission['time_to_failure']


submission_data['time_to_failure'] = clf.predict(test_data[features]) 
submission_data = submission_data[['seg_id','time_to_failure']]
submission_data.set_index('seg_id', inplace=True)

submission_data.to_csv('stacked_submission.csv', header='seg_id,time_to_failure')

print('-----------------  finished --------------------')



