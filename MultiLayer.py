#!/python

import numpy as np
import random


# sizes == number of neurons per layer
# net = Network([2, 3, 1]) two inputs, 3 hidden, one output
# biases not set for input layer
# random weights centered around 0 for initial weights with std of 1



# math helper functions

# sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    
# derivative of sigmoid function            
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
        
       

class Network(object):
        
    def __init__(self, sizes):    
    
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        
        # init with sqrt of random Guassian, mean 0, stdv 1
        # this settles the network down quicker but doesn't change accuracy
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

     
        
   
    # grab total number of correct results
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data] 
        return sum(int(x == y) for (x, y) in test_results)
        
               
    # output = sigmoid (( w * inputs) + b)    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
        
        
    # stochastic gradient descent, calculate and update the network as we go through data
    # eta - is the learning rate
    # epoch - all of the training samples pass through in one epoch 
    # optional test data - lets you see progress but slows things down
    # mini_batch_size is the number of training examples for each network adjustment  
    def sgd ( self, training_data, epochs, mini_batch_size, eta, lmbda, test_data=None ):
            
        # if test data provided print progress, slows things down a lot
        # convert to a list, otherwise python 3 has no way to check length 
        test_data = list(test_data)   
        if test_data: n_test = len(test_data)
        
        # python 3 conversion needed to get length of data set
        training_data = list(training_data)
        n = len(training_data)
            
            
        # loop through full training set    
        for j in range(epochs):
            # shuffled training data works much better
            random.shuffle(training_data)
            
            # break the data into batches
            mini_batches = [training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)]
                
            # update the network once per batch 
            for mini_batch in mini_batches:
                 self.update_mini_batch(mini_batch, eta, lmbda, n)
                    
            # if test data is provided, print the progress        
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
                
     
    
    # used to run validation data through network 
    def process_data(self, data):
        results = self.evaluate(data)
        return results
        
    
    # runs through the Kaggle validation data and writes out a submission file    
    def create_submission_data(self, data):
        test_results = [(np.argmax(self.feedforward(x))) for (x) in data] 
        
        # save results to disk
        np.savetxt('kaggle_MNIST.csv', 
           np.c_[range(1,len(test_results)+1),test_results], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')


     
    # update weights and biases using gradient descent on a single batch
    # eta is the learning rate
    # nabla == gradient for this batch
    def update_mini_batch(self, mini_batch, alpha, lmbda, n):
    
        # misc
        batch_size = len(mini_batch)
        
        # temporary storage 
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]
         
        # calculate the adjustment to the weights and biases 
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)        # adjustments to weights and bias
            new_b = [nb + dnb for nb, dnb in zip(new_b, delta_b)]
            new_w = [nw + dnw for nw, dnw in zip(new_w, delta_w)]
            
        # adjust the weights and biases 
        #self.weights = [w - alpha/batch_size * nw for w, nw in zip(self.weights, new_w) ]  # no regularization
        self.weights = [(1 - lmbda/batch_size) * w - alpha/batch_size * nw for w, nw in zip(self.weights, new_w) ]  # add regularization
        
        # b - (alpha/batch_size * nb)
        self.biases = [b - alpha/batch_size * nb for b, nb in zip(self.biases, new_b)]
        
       
        
        
        
        
        
        
        
    # returns the gradient for the cost function

    def backprop(self, x, y):
        
        # temporary storage
        adjust_b = [np.zeros(b.shape) for b in self.biases]
        adjust_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]   # store layer by layer lists of neurode outputs
        zs = []             # store layer by layer list of weighted inputs
        
        
        # feedforward --- push inputs through network
        for b, w, in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b   # weights * input + b  i.e. 'weighted input'
            zs.append(z)                    # store z
            activation = sigmoid(z)         # run output through sigmoid function    
            activations.append(activation)  # store output for next layer
            
            
        # push errors backwards through the network
        # error between expected and actual
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])  # regular loss function

        
        
        
        
        adjust_b[-1] = delta                                      # the output error
        adjust_w[-1] = np.dot(delta, activations[-2].transpose())  # error * output
        
        
        # for each layer work backwards
        for l in range(2, self.num_layers):     # l = 2 is 2nd to last layer
            z = zs[-l]                          # previous layer output            
            derivative = sigmoid_prime(z)       # derivative of previous layer's output
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * derivative # weights dotted with error * derivative
            adjust_b[-l] = delta                # amount to adjust bias
            adjust_w[-l] = np.dot(delta, activations[-l - 1].transpose()) # amount to adjust weights
            
            
        # send back the adjustments    
        return ( adjust_b, adjust_w )
        