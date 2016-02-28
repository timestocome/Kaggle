#!/python

# Kaggle MNIST submissions
# started with .... 
# http://neuralnetworksanddeeplearning.com/chap1.html source code from "Neural Networks and Deep Learning" Nielsen

##################################################################################
##################################################################################
# to do
# improve network
# maybe better stats and some graphical output




# read in the data files and format as needed
import LoadData
training_data, validation_data, test_data, kaggle_data = LoadData.load_data_wrapper()



########## multi layer network ######################################################
#~ 96.8% accurate first pass
import MultiLayer

# create the network
net = MultiLayer.Network([784, 120, 60, 10])  # layer sizes ( input, hidden, output )

epochs = 30         # number of passes through full data set
batch_size = 5     # size of batches, network updated once per batch
alpha = 1.2         # learning step
lmbda =  0.00005        # regularization 
net.sgd(training_data, epochs, batch_size, alpha, lmbda, test_data=test_data) # train epochs, batch size, alpha


# run validation data through network
validation_results = net.process_data(validation_data)
print("Validation results {0}%".format(validation_results / 10.0))


# run kaggle submission images through net and create csv submission file for kaggle
net.create_submission_data(kaggle_data)

