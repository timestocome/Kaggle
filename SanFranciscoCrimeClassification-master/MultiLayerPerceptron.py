#!/python


################################################################
# accuracy is only about 20%
###############################################################

# built in
import numpy as np
import pickle, gzip
import timeit
import sys, os


# 3rd party
import theano 
import theano.tensor as T


################################################################################################
# single hidden layer netword - input, hidden, output
# best scores are about 22% altering parameters doesn't seem to help a whole lot

#################################################################################################
# parameters used to tweak learning
##################################################################################################
learning_rate = 0.2    # how fast does net converge - bounce out of local mins
                        # 
L1_reg = 0.00         # lambda - scaling factor for regularizations, slightly better accuracy
L2_reg = 0.0001         #     with  L1 
n_epochs = 5         # max number of times we loop through full training set
batch_size = 50         # number of training examples per batch - smaller is slower but better accuracy( above 20)
n_hidden = 224           # number of nodes in hidden layer increasing or decreasing from 100 slowly drops accuracy
n_outputs = 39

#################################################################################################
# load up data 
# SF Crime data
# labels are single ints 0-39
# training set is locations, times and classifications
# testing and validation peak at about 18-19%
##################################################################################################
        
# inputs = 39                
#import PrepDataforMLP
#train_set, valid_set, test_set, kaggle_set = PrepDataforMLP.prepData()        
 
 
  
# Load Data with pandas, and parse the first column into datetime
#n_inputs = 110
#import CategorizeData
#train_set, valid_set, test_set, kaggle_set = CategorizeData.get_data()




import PrepDataOneHot
n_inputs = 112
train_set, valid_set, test_set, kaggle_set = PrepDataOneHot.get_data()


       
        
# shared variables maintain state, and changes are usually done in place
# borrow=False, makes copy of array and doesn't update it, True allows it to be updated
def shared_dataset(data_xy, borrow=True):
       
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    return shared_x, T.cast(shared_y, 'int32')


# set up data as shared variables
test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)
kaggle_set_x, kaggle_set_y = shared_dataset(kaggle_set)

# set up constants
# double front slash (//) divide and round down to floor
n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
n_kaggle_batches = kaggle_set_x.get_value(borrow=True).shape[0] // batch_size    



             
##########################################################################################
# Hidden Layer 
# uses tanh instead of sigmoid
# activation = tanh(dot(x,W) + b)
##########################################################################################         
class HiddenLayer(object):
    # rng = random state
    # input = output from previous layer
    # n_in = number of inputs, length of previous layer 
    # n_out = number of hidden units
    # activation can be T.tanh or T.nnet.sigmoid
    # W = weights
    # b = bias
    #def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
     def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.nnet.sigmoid):
        
        self.input = input
        
        # initial random weights sqrt( +/-6. / (n_in + n_hidden)), multiply by 4 for sigmoid
        if W is None:
            W_values = np.asarray(rng.uniform(
                                    low = -np.sqrt(1./(n_in + n_out)),
                                    high = np.sqrt(1. /(n_in + n_out)),
                                    size = (n_in, n_out)
                                    ), dtype = theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4.
            W = theano.shared(value = W_values, name = 'W', borrow = True)
            
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True)
        
        self.W = W
        self.b = b
        
        # calculate linear output using dot product + b, else use tanh or sigmoid
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        
        # update weights and bias
        self.params = [self.W, self.b]
        
        
   
     
##########################################################################################
# Logistic Regression Layer
# activation = softmax(dot(x, w) + b)
##########################################################################################
class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
    
        # initialize parameters 
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
   
        # map input to hyperplane to determine output
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
   
        # compute predicted class from softmax layer
        self.y_predict = T.argmax(self.p_y_given_x, axis=1)
   
        self.params = [self.W, self.b]
        self.input = input
   


    # using mean instead of sum allows for different batch sizes with out adjusting learning rate
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    
    
    # number of errors in the mini batch
    def errors(self, y):
        return T.mean(T.neq(self.y_predict, y))
    
    
    
     
     
         
##########################################################################################
# MultiLayer Perceptron Class
# Forward feed, backpropagation network with one or more hidden layers using non-linear activations
# output layer uses softmax
# input layer is the data
##########################################################################################                
class MLP(object):
    
    # random state (initializer)
    # input - input data
    # n_in - number of input units (28*28 for mnist)
    # n_hidden - number of hidden units in hidden layer
    # n_out - number of output labels (10 digits for mnist)
    def __init__(self, rng, input, n_in, n_hidden, n_out):
            
            # hidden layer, takes in weighted inputs and feeds them to the 
            # logistic regression layer
            self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=T.tanh)
            
            # logistic regression layer gets weighted hidden activations as input
            # and outputs the label (class) we think the image belongs to
            self.logisticRegressionLayer = LogisticRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)
                   
            # regularization
            self.L1 = ( abs(self.hiddenLayer.W).sum() + abs(self.logisticRegressionLayer.W).sum() )
            self.L2 = ( (self.hiddenLayer.W ** 2).sum() + (self.logisticRegressionLayer.W ** 2).sum() )
            
            # measure error
            self.negative_log_likelihood = ( self.logisticRegressionLayer.negative_log_likelihood )
            self.errors = self.logisticRegressionLayer.errors
            
            # parameters
            self.params = self.hiddenLayer.params + self.logisticRegressionLayer.params
            
            # data in
            self.input = input
            
            
            # predictions
            self.predicted = self.logisticRegressionLayer.y_predict
            
    # networks predictions for data not used in training
    #def predicted(self, x):
     #   return self.logisticRegressionLayer.p_y_given_x
    
    
###########################################################################################
# run the network on the dataset
def test_mlp():

    ##################################################
    # build the model
    ##################################################
    print("building the model......")
    
    index = T.lscalar()                 # index to minibatch
    x = T.matrix('x')                   # data in
    y = T.ivector('y')                  # output classes/labels
    rng = np.random.RandomState(42)     # seed for random number stream


    # set model up for training  
    classifier = MLP( rng=rng, input=x, n_in=n_inputs, n_hidden=n_hidden, n_out=n_outputs)
    cost = ( classifier.negative_log_likelihood(y) + (L1_reg * classifier.L1) + (L2_reg * classifier.L2) )            
    gradients = [T.grad(cost, param) for param in classifier.params]
    updates = [(param, param - learning_rate * g) for param, g in zip(classifier.params, gradients)]
    
    
    # using givens allows us to push through a new batch each pass without altering existing state of things
    
    # run training data through network and update the weights and biases
    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                givens={
                    x: train_set_x[index * batch_size:(index + 1) * batch_size],
                    y: train_set_y[index * batch_size:(index + 1) * batch_size]
                })
    
    #### no updates on these functions we're running the data through not training the net            
    # run test data through network to track network convergence
    test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]
            })

    # run validation data through network and return errors
    validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            })
                    
    
    predict = theano.function(inputs=[], outputs=classifier.predicted,
                                givens={
                                    x: kaggle_set_x
                                })           
                
                
                
    ##################################################
    # train the model
    ##################################################            
    print("training ....")
    
     # early-stopping parameters --- helps keep network from over training.
    patience = 10000                    # look as this many examples regardless
    patience_increase = 10               # wait this much longer when a new best is found
                                        # increases improve training accuracy but not test or validation
    improvement_threshold = 0.995        # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
    
        epoch = epoch + 1
    
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
            
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = 1.0 - np.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, accuracy %f %%' %
                    ( epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                
                    #improve patience if loss improvement is good enough
                    if ( this_validation_loss < best_validation_loss * improvement_threshold ):
                        patience = max(patience, iter * patience_increase)
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = (1.0 - np.mean(test_losses)) * 100.
                    
                   
                    print(('     epoch %i, minibatch %i/%i, best accuracy %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches, test_score))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score ))
          
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
           
   
    predictions = predict()
    np.savetxt("mlp_predictions.txt", predictions )
    print(predictions)    

    
    # sanity check  
    import pandas as pd        
    submission_data = pd.read_csv("mlp_predictions.txt", names=['Prediction'])
    submission_data['Prediction'] = submission_data['Prediction'].astype(int)
    crimesByCategory = submission_data['Prediction'].value_counts()
    totalCrimes = len(submission_data)
    print("Crimes by categoryNumber")
    print(crimesByCategory/totalCrimes)

##########################################################################################
# run network
test_mlp()