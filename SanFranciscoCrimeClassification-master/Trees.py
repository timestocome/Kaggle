#/python



import pandas as pd
import numpy as np

from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn import cross_validation


# up to about 31% using voting


# Load Data with pandas, and parse the first column into datetime
#import CategorizeData
#train_set, valid_set, test_set, kaggle_set = CategorizeData.get_data()

import PrepDataForClusters
training_data, kaggle_data = PrepDataForClusters.load_cluster_data()

# split inputs from outputs
features = ['Year', 'Month', 'Day', 'Hour', 'DayNumber', 'DistrictNumber', 'A', 'X', 'Y', 'Intersection']


    
#kaggle_x = kaggle_data[features]
#kaggle_y = kaggle_data['CategoryNumber']
    
training_x = training_data[features]
training_y = training_data['CategoryNumber']    
    
    
  # create a testing and validation set from the training_data
train_x, test_x, train_y, test_y = cross_validation.train_test_split(training_x, training_y, test_size=0.1)
       
estimators = 40       
clf1 = BaggingClassifier(n_estimators=estimators)   #30.5
clf2 = ExtraTreesClassifier(n_estimators=estimators) #30.1
clf3 = RandomForestClassifier(n_estimators=estimators) #31.1


#clf = clf3
clf = VotingClassifier(estimators=[('b', clf1), ('e', clf2), ('r', clf3)], voting='soft' )
clf.fit(train_x, train_y)
predicted = clf.predict(test_x)




np.savetxt("temp_predictions.txt", predicted )

print("Correct", sum(predicted == test_y))
print("Total", len(test_y))
print("Accuracy", sum(predicted == test_y) / len(test_y) * 100.)
print("Uniques", len(np.unique(predicted)))




# sanity check  
import pandas as pd   
     
submission_data = pd.read_csv("temp_predictions.txt", names=['Prediction'])
submission_data['Prediction'] = submission_data['Prediction'].astype(int)
crimesByCategory = submission_data['Prediction'].value_counts()
totalCrimes = len(submission_data)
print("Predicted crimes by categoryNumber")
crimesPercent = crimesByCategory/totalCrimes
print(crimesPercent)


