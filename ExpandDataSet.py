#!/python

# https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/expand_mnist.py



from __future__ import print_function

#### Libraries

# Standard library
import pickle
import gzip
import os.path
import random

# Third-party libraries
import numpy as np

print("Expanding the MNIST training set")



f = open("train.pkl", 'rb')
training_data = pickle.load(f)
f.close()
    
expanded_training_pairs = []
j = 0 # counter
    
training_data = list(training_data)
    
    
for x, y in training_data:
        
        expanded_training_pairs.append((x, y))
        
        image = np.reshape(x, (-1, 28))
        j += 1
        if j % 1000 == 0: print("Expanding image number", j)
    
        # iterate over data telling us the details of how to do the displacement
        for d, axis, index_position, index in [
                (1,  0, "first", 0),
                (-1, 0, "first", 27),
                (1,  1, "last",  0),
                (-1, 1, "last",  27)]:
            new_img = np.roll(image, d, axis)
            if index_position == "first": 
                new_img[index, :] = np.zeros(28)
            else: 
                new_img[:, index] = np.zeros(28)
            expanded_training_pairs.append((np.reshape(new_img, 784), y))
    
random.shuffle(expanded_training_pairs)
expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]
    
print("Saving expanded data. This may take a few minutes.")
print("New data count ", len(expanded_training_data[0]))
    
f = gzip.open("mnist_expanded.pkl.gz", "w")
pickle.dump((expanded_training_data), f)
f.close()