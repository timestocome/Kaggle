

# https://github.com/timestocome
# https://www.kaggle.com/c/LANL-Earthquake-Prediction


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# silence sklearn
import warnings
warnings.filterwarnings('ignore')


# read in train data feature file
train = pd.read_csv('train_stats.csv', index_col=0)
n_samples = len(train)


#print(train.head())
print(list(train.columns.values))

all_features = ['power1', 'power2', 'power3', 
		'median_dev_abs', 'over_1std', 'over_2std', 
		'time_to_failure']


features = ['power1', 
	'power2', 
	'power3', 
	'median_dev_abs',
	'over_1std',
	'over_2std'	
]



train = train[all_features]


#print(train.head())
#print(train.describe())



########################################################################################
# import sklearn libraries for model
#########################################################################################
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score



#########################################################################################
# split into train and holdout data
n_holdout = int(len(train) * .8)
x = train[features][0:n_holdout]
y = train['time_to_failure'][0:n_holdout]

x_holdout = train[features][n_holdout:n_samples]
y_holdout = train['time_to_failure'][n_holdout:n_samples]





print('Splitting data')
# split data up into features, targets, train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



# create model
clf = RandomForestRegressor(n_estimators=512, n_jobs=6, max_depth=8, criterion='mse')



# train model using cross validation 
print('training....')

cv_results = cross_validate(clf, x_train, y_train, cv=5, return_train_score=True)




# re-fit classifier on all training data to check feature values
print('----------------------Feature importance--------------')
clf.fit(x, y)
feature_values = zip(features, clf.feature_importances_)
feature_values = sorted(feature_values, key = lambda z: z[1])

for i, j in feature_values:
    print(' %s     %.9f ' %(i, j))




# check classifier accuracy on holdout data
predict_test = clf.predict(x_holdout) 
y_test = y_holdout.values 

# check training scores
predict = clf.predict(x)

print('=========================================================================')
print('----------------- holdout / test--------------------------')
print('CV Test: mean score %f ' % (np.mean(cv_results['test_score']) * 100.))
mae = (np.abs(predict_test - y_test )).mean()
print('mae error on hold out data', mae)

print('---------------------- train ---------------------------------------')
print('CV Train: mean score %f' % (np.mean(cv_results['train_score']) * 100.))
mae = (np.abs(predict - y.values )).mean()
print('mae error on training data', mae)


print('actual max/min', y_test.max(), y_test.min())
print('predicted max/min', predict_test.max(), predict_test.min())
print('=========================================================================')



###############################################################################################
# plot predictions vs actual on training data and holdout data
##############################################################################################
predict_train = clf.predict(train[features]) 
predict_y = train['time_to_failure']

fig, ax = plt.subplots(figsize=(16, 16))

plt.scatter(np.arange(len(predict_train)), predict_train, c='r', label='Predicted')
plt.plot(predict_y, c='b', label='Actual', alpha=0.3)

plt.legend()
plt.grid(True)
plt.savefig('predictions.png')
plt.show()



#################################################################################################
# refit regressor on all training data
#################################################################################################



# regressor
clf.fit(train[features], train['time_to_failure'])


# sanity check 
predictions = clf.predict(train[features])

mae = (np.abs(predictions - train['time_to_failure'])).mean()
print('mae - all data', mae)

base = list(train.index.values)

fig, ax = plt.subplots(figsize=(16, 16))

plt.scatter(base, predictions, c='r', label='Predicted')
plt.plot(predict_y, c='b', label='Actual', alpha=0.3)

plt.legend()
plt.grid(True)
plt.savefig('predictions.png')
plt.show()



#############################################################################################
# save model
#############################################################################################
import pickle

save_model = pickle.dumps(clf)
# load_model = pickle.loads(file)



#############################################################################################
# read in test data and run it through clf
# save submission file to upload to kaggle
###############################################################################################
submission_data = pd.read_csv('test_stats.csv', index_col=0)


submission_data['time_to_failure'] = clf.predict(submission_data[features]) 
submission_data = submission_data[['seg_id','time_to_failure']] 
submission_data.set_index('seg_id', inplace=True)

submission_data.to_csv('submission.csv', header='seg_id,time_to_failure')

print('-----------------  finished --------------------')



