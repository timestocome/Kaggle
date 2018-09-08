#!/python


import numpy as np
import pandas as pd
import pickle


##############################################################################
# read in csv file
# import it into a pandas dataframe
# store as pickle file
###############################################################################

# read in csv files
def csv_pickle():
    training_data = pd.read_csv("train.csv")
    validation_data = pd.read_csv("test.csv")
        

# save the dataframes to disk as pickle files    
def save_data():
    # save as pickle files
    pickle_out = open('training.pkl', 'wb')
    pickle.dump(training_data, pickle_out)
    pickle_out.close()

    pickle_out = open('validation.pkl', 'wb')
    pickle.dump(validation_data, pickle_out)
    pickle_out.close()


# load df files from pickle file on disk
def load_data():
    # read in pickle files
    file = 'training.pkl'
    with open(file, 'rb') as f:
        training_data = pickle.load(f)
        f.close()

    file = 'validation.pkl'
    with open(file, 'rb') as f:
        validation_data = pickle.load(f)
        f.close()
      
      # convert object to useful types
    training_data['Dates'] = pd.to_datetime(training_data.Dates)
    validation_data['Dates'] = pd.to_datetime(validation_data.Dates)

      # sort by date earliest first  
    training_data = training_data.sort_values(by="Dates")
    validation_data = validation_data.sort_values(by="Dates")
    
      # missing values ?
    #print(training_data.isnull())
    #print(validation_data.isnull())    
        
    return training_data, validation_data
    
    
    

    
