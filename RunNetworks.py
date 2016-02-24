#!/python

# Kaggle MNIST submissions
# started with .... 
# http://neuralnetworksanddeeplearning.com/chap1.html source code from "Neural Networks and Deep Learning" Nielsen

##################################################################################
##################################################################################
# to do
# save and reload network
# run test, validation data through network after training
# need some better stats - break training into training, testing, validation files
# create submission files for Kaggle



# read in the data files and format as needed
import LoadData
training_data, validation_data, test_data, kaggle_data = LoadData.load_data_wrapper()



########## multi layer network ######################################################
#~ 96.8% accurate first pass
import MultiLayer

# create the network
net = MultiLayer.Network([784, 30, 30, 10])  # layer sizes ( input, hidden, output )

epochs = 40        # number of passes through full data set
batch_size = 10     # size of batches, network updated once per batch
alpha = 0.5         # learning step
lmbda = 3.0         # regularization 
net.sgd(training_data, epochs, batch_size, alpha, lmbda, test_data=test_data) # train epochs, batch size, alpha


# run validation and kaggle data through network
validation_results = net.process_data(validation_data)
print(validation_results)

# create csv submission file for kaggle
net.create_submission_data(kaggle_data)

