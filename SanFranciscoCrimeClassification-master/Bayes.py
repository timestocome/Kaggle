#/python


# starting point for this attempt
# http://efavdb.com/predicting-san-francisco-crimes/

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
import numpy as np
 
 
# Load Data with pandas, and parse the first column into datetime



import PrepDataForClusters
train_set, valid_set, test_set, kaggle_set = PrepDataForClusters.load_cluster_data()

#import CategorizeData
#train_set, valid_set, test_set, kaggle_set = CategorizeData.get_data()


train_x, train_y = train_set        
validate_x, validate_y = valid_set
test_x, test_y = test_set
kaggle_x, kaggle_y = kaggle_set



clf = BernoulliNB(fit_prior=True)
clf.fit(train_x, train_y)
predicted = clf.predict(test_x)
np.savetxt("bayes_predictions.txt", predicted )


print("Correct", sum(predicted == test_y))
print("Total", len(test_y))
print("Bernoulli Accuracy", sum(predicted == test_y) / len(test_y) * 100.)



# sanity check  
import pandas as pd   
     
submission_data = pd.read_csv("bayes_predictions.txt", names=['Prediction'])
submission_data['Prediction'] = submission_data['Prediction'].astype(int)
crimesByCategory = submission_data['Prediction'].value_counts()
totalCrimes = len(submission_data)
print("Predicted crimes by categoryNumber")
print(crimesByCategory/totalCrimes)
        
