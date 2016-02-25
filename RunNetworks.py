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
net = MultiLayer.Network([784, 196, 49, 20, 10])  # layer sizes ( input, hidden, output )

epochs = 60         # number of passes through full data set
batch_size = 10     # size of batches, network updated once per batch
alpha = 0.2         # learning step
lmbda = 6.0         # regularization 
net.sgd(training_data, epochs, batch_size, alpha, lmbda, test_data=test_data) # train epochs, batch size, alpha


# run validation data through network
validation_results = net.process_data(validation_data)
print(validation_results)


# run kaggle submission images through net and create csv submission file for kaggle
net.create_submission_data(kaggle_data)

