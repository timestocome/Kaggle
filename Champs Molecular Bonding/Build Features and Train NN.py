#!/usr/bin/env python
# coding: utf-8

# This kernel is based on
# 
# * https://www.kaggle.com/xwxw2929/keras-neural-net-and-distance-features
# * https://www.kaggle.com/todnewman/keras-neural-net-for-champs
# * https://www.kaggle.com/criskiev/distance-is-all-you-need-lb-1-481
# * https://www.kaggle.com/abazdyrev/nn-w-o-skew
# * https://www.kaggle.com/marcogorelli/criskiev-s-distances-more-estimators-groupkfold
# * https://www.kaggle.com/inversion/atomic-distance-benchmark

# ### https://github.com/timestocome/Kaggle
# ### https://www.kaggle.com/c/champs-scalar-coupling
# 
# ### Goal: improve prediction of scalar coupling constant using only provided dataset
# 
# ####  Improvements
# * added plts of NN convergence
# * added plts of predicted vs actual in validation data
# * added functions to calculate VanDerWaals, Coulomb, Yukawa forces
# * made several changes to database manipulations to make them cleaner, faster
# *     and add in several features not in previous kernels
# * redid the neural network in the previous examples as they only smoothed out when overfitting the data
# 
# 
# * Most of my time in this competition was spent on feature engineering
# * There wasn't enough time at the end to get the ML NN fine tuned
# * Typically the data calculations would be done and stored in a csv so they didn't have to be
# *    recalculated each time the model was run --- but I ran out of time on this contest
# * Also I would've custom build a neural net for each coupling rather than use the same size one for all
# *     the data for each of the 8 coupling types varied from less than half a million rows to 1.5 million rows
# 
# 
# #### Fails
# * I read lots and lots of papers. I tried calculating all the nearest neighbors and angles using
# *     Networkx. Networkx was very slow and the angles turned out to be poor predictors
# 
# * Karplus, Coulomb, Vanderwaals, Yukawa all yielded decent results but are heavily coupled confusing the 
# *     neural network. Given more time I'd've tried all the combinations of each on each coupling type
# 
# * Counting bonds and atoms were decent predictors but were also heavily coupled with distance and bond type
# 
# * I also tried electonegativity, which was an okay predictor. Even in this last version some of the 
# *     coupling types still have large std on test data despite the large size of the dataset
# 
# * The dataset had several feature files for the training set only.  I tried using a two stage neural network
# *      to calculate them, then calculate the target but it wasn't very successful
# 
# 
# 
# 
# #### New skills
# * The Pandas group-stack is very cool, it would've saved me weeks of slow feature calculations
# * Two stage neural networks - pull some outputs out before end layer and train to them as well as final 
# *     target layer
# 
# 
# * Best possible score ~ -20
# * Winning score -3.2
# * My best -1.3

# In[1]:




import pandas as pd
import numpy as np


# In[2]:



import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
warnings.filterwarnings(action="ignore",category=FutureWarning)


# In[3]:



import math
import gc
import copy

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


import tensorflow as tf


from keras.layers import Dense, Input, Activation
from keras.layers import BatchNormalization,Add,Dropout
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras import callbacks
from keras import backend as K


# In[4]:


# Set up GPU preferences, needed for newer GPUs with Keras
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 2} ) 
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config) 

K.set_session(sess)


# ### read in raw datafiles

# In[5]:


def read_train():
    train = pd.read_csv('Data/train.csv', index_col=0)

    # remove batch name from molecule id
    train['molecule_index'] = train.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
    train.drop(['molecule_name'], axis=1, inplace=True)

    coupling_types = sorted(train['type'].unique())

    return train
    
train = read_train()


# In[6]:


def read_test():

    test = pd.read_csv('Data/test.csv', index_col=0)


    # remove batch name from molecule id
    test['molecule_index'] = test.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
    test.drop(['molecule_name'], axis=1, inplace=True)

    return test
    
test = read_test()


# In[7]:


def read_struct():

    struct = pd.read_csv('Data/structures.csv')

    atomic_numbers = { 'H': 1, 'C':6, 'N':7, 'O':8, 'F': 9 }

    # https://www.thoughtco.com/element-charges-chart-603986
    atomic_chg = { 'H': 1, 'C': 4, 'N': -3, 'O': -2, 'F': -1 }

    # https://en.wikipedia.org/wiki/Electronegativity
    atomic_pchgs = { 'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98 }

    # https://en.wikipedia.org/wiki/Atomic_radius
    atomic_size = { 'H': 0.25, 'C': 0.7, 'N': 0.65, 'O': 0.60, 'F': 0.5 }


    # get atom count per molecule and merge back into struct
    u_atoms = struct.groupby(['molecule_name'])['atom'].value_counts()
    u_atoms = u_atoms.unstack()
    u_atoms = u_atoms.fillna(0)


    # get OxBalance
    u_atoms['total atoms'] = u_atoms['O'] + u_atoms['C'] + u_atoms['F'] + u_atoms['N'] + u_atoms['H']
    u_atoms['Ox balance'] = (u_atoms['O'] - 2*u_atoms['C'] - u_atoms['H']) / u_atoms['total atoms']

    struct = pd.merge(struct, u_atoms, how='left', left_on=['molecule_name'], right_on=['molecule_name'])



    # remove batch name from molecule id
    struct['molecule_index'] = struct.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
    struct.drop(['molecule_name'], axis=1, inplace=True)


    # swap out letter id for atoms with atomic_numbers
    #struct['atom'] = struct['atom'].replace(atomic_size)   # -.17
    #struct['atom'] = struct['atom'].replace(atomic_chg)   # -.25
    #struct['atom'] = struct['atom'].replace(atomic_pchgs)   # -.22
    struct['atom'] = struct['atom'].replace(atomic_numbers)   # -.25
    return struct
    
struct = read_struct()


# In[8]:


submission = pd.read_csv('Data/sample_submission.csv', index_col='id')

print(submission.head())


# In[ ]:





# ### prep data for model

# #### split out data types and merge with location db

# In[ ]:



# pull coupling datatype from train/test and structures dfs
def fetch_type_data(df, struct, ctype):
    
    
    
    # pull out all rows with this coupling type from train data and make a copy
    df = df[df['type'] == ctype].drop('type', axis=1).copy()
    
    
    # pull out all rows with this coupling type from structs
    struct = read_struct()
    struct_df = struct[struct['molecule_index'].isin(df['molecule_index'])]
    
    
    return df, struct


# In[ ]:



# merge train/test with structs df
def merge_coordinates(bonds, structures, index):
    
    df = pd.merge(bonds, structures, how='inner',
                  left_on=['molecule_index', f'atom_index_{index}'],
                  right_on=['molecule_index', 'atom_index']).drop(['atom_index'], axis=1)
    
    df = df.rename(columns={
        'atom': f'atom_{index}',
        'x': f'x_{index}',
        'y': f'y_{index}',
        'z': f'z_{index}'
    })
    
    if index == 0:
        df.drop(columns=['C', 'F', 'H', 'N', 'O', 'total atoms', 'Ox balance'], inplace=True)
        
    return df
    


# #### get distances between atoms

# In[ ]:


# calculate distance between each atom 

def add_distance_between(df, suffix1, suffix2):
    
    df[f'd_{suffix1}_{suffix2}'] = ((
        (df[f'x_{suffix1}'] - df[f'x_{suffix2}'])**np.float32(2) +
        (df[f'y_{suffix1}'] - df[f'y_{suffix2}'])**np.float32(2) + 
        (df[f'z_{suffix1}'] - df[f'z_{suffix2}'])**np.float32(2)
    )**np.float32(0.5))




def add_distances(df):
    n_atoms = 1 + max([int(c.split('_')[1]) for c in df.columns if c.startswith('x_')])
    
    for i in range(1, n_atoms):
        for vi in range(min(4, i)):
            add_distance_between(df, i, vi)


# In[ ]:


# calculate coulomb force distance between each atom 
# 1/2 Z^2.4 if i!=j
# Zi*Zj/|Ri - Rj|


# atomic numbers should be used for atom
def add_coulomb_between(df, suffix1, suffix2):
    
    
    Zi = df[f'atom_{suffix1}']
    Zj = df[f'atom_{suffix2}']
    
    #df[f'c_sm_{suffix1}_{suffix2}'] = 0.5 * Zi**2.4
        
    df[f'c_df{suffix1}_{suffix2}'] = (Zi * Zj) / ((
        (df[f'x_{suffix1}'] - df[f'x_{suffix2}'])**np.float32(2) +
        (df[f'y_{suffix1}'] - df[f'y_{suffix2}'])**np.float32(2) + 
        (df[f'z_{suffix1}'] - df[f'z_{suffix2}'])**np.float32(2)
         )**np.float32(0.5))






def add_coulombs(df):
    n_atoms = 1 + max([int(c.split('_')[1]) for c in df.columns if c.startswith('x_')])
    
    for i in range(1, n_atoms):
        for vi in range(min(4, i)):
            add_coulomb_between(df, i, vi)


# In[ ]:


# calculate  yukawa force distance between each atom 

def add_yuka_between(df, suffix1, suffix2):
    
    df[f'yk_{suffix1}_{suffix2}'] =  np.exp(-((
        (df[f'x_{suffix1}'] - df[f'x_{suffix2}'])**np.float32(2) +
        (df[f'y_{suffix1}'] - df[f'y_{suffix2}'])**np.float32(2) + 
        (df[f'z_{suffix1}'] - df[f'z_{suffix2}'])**np.float32(2)
    )**np.float32(0.5)))




def add_yukas(df):
    n_atoms = 1 + max([int(c.split('_')[1]) for c in df.columns if c.startswith('x_')])
    
    for i in range(1, n_atoms):
        for vi in range(min(4, i)):
            add_yuka_between(df, i, vi)


# In[ ]:


# calculate vanderwaal's force distance between each atom 

def add_vander_between(df, suffix1, suffix2):
    
    df[f'v_{suffix1}_{suffix2}'] = 1./ ((
        (df[f'x_{suffix1}'] - df[f'x_{suffix2}'])**np.float32(2) +
        (df[f'y_{suffix1}'] - df[f'y_{suffix2}'])**np.float32(2) + 
        (df[f'z_{suffix1}'] - df[f'z_{suffix2}'])**np.float32(2)
    )**np.float32(0.5))




def add_vanders(df):
    n_atoms = 1 + max([int(c.split('_')[1]) for c in df.columns if c.startswith('x_')])
    
    for i in range(1, n_atoms):
        for vi in range(min(4, i)):
            add_vander_between(df, i, vi)


# In[ ]:





def build_df(df, structures, ctype, n_atoms):
 
    
    # fetch coupling type data from train/test and structs files
    bonds, structs = fetch_type_data(df, struct, ctype)
    
    # merge structs and coupling data for atom col 0, 1
    bonds = merge_coordinates(bonds, structs, 0)
    bonds = merge_coordinates(bonds, structs, 1)
   
   
    
    # make a copy of df just built and drop target from training
    atoms = bonds.copy()
    if 'scalar_coupling_constant' in df:
        atoms.drop(['scalar_coupling_constant'], axis=1, inplace=True)

        
    # add center location between coupled atoms and drop xyz locations of each atom
    atoms['x_c'] = ((atoms['x_1'] + atoms['x_0']) * np.float32(0.5))
    atoms['y_c'] = ((atoms['y_1'] + atoms['y_0']) * np.float32(0.5))
    atoms['z_c'] = ((atoms['z_1'] + atoms['z_0']) * np.float32(0.5))
    
    # add r
    atoms['dx'] = ((atoms['x_1'] + atoms['x_0']) * np.float32(0.5))
    atoms['dy'] = ((atoms['y_1'] + atoms['y_0']) * np.float32(0.5))
    atoms['dz'] = ((atoms['z_1'] + atoms['z_0']) * np.float32(0.5))
    atoms['r'] = np.sqrt(atoms['dx'] + atoms['dy'] + atoms['dz'])
    atoms.drop(['dx', 'dy', 'dz'], axis=1, inplace=True)
    
    # drop location info
    atoms.drop(['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1'], axis=1, inplace=True)
    
    # merge location info back in
    # this creates a row for each atom to all other atoms
    atoms = pd.merge(atoms, structures, how='left',
                  left_on=['molecule_index'],
                  right_on=['molecule_index'], 
                    suffixes=('','_junk'))

    # cleanup
    atoms = atoms.drop([col for col in atoms.columns if '_junk' in col],axis=1)
    atoms = atoms[(atoms.atom_index_0 != atoms.atom_index) & (atoms.atom_index_1 != atoms.atom_index)]
    
    
    
    # add distances for each atom to center of coupled pairs
    atoms['d_c'] = ((
        (atoms['x_c'] - atoms['x'])**np.float32(2) +
        (atoms['y_c'] - atoms['y'])**np.float32(2) + 
        (atoms['z_c'] - atoms['z'])**np.float32(2)
    )**np.float32(0.5))
    
    
    # cleanup
    atoms = atoms.drop(['x_c', 'y_c', 'z_c', 'atom_index'], axis=1)


    # sort by molecule, atom 0, atom 1 and distance to center of coupled pairs
    atoms.sort_values(['molecule_index', 'atom_index_0', 'atom_index_1', 'd_c'], inplace=True)
    
    
    # group and unstack.... https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html
    # group atoms by molecule and atoms
    atom_groups = atoms.groupby(['molecule_index', 'atom_index_0', 'atom_index_1'])
    
    
    
    # count the number of atoms in molecule by distance from coupled pair
    # and remove distance to center --- only needed for sort
    atoms['num'] = atom_groups.cumcount() + 2
    atoms = atoms.drop(['d_c'], axis=1)
    
    # remove rows that are greater than the max of nearby atoms wanted for 
    atoms = atoms[atoms['num'] < n_atoms]
        
   
        
    # convert molecule id, atom 0, atom 1 into index columns
    # this adds cols for each atom up to n_atoms, and x, y, z for each atom that is kept
    # relabel new columns and fix index
    atoms = atoms.set_index(['molecule_index', 'atom_index_0', 'atom_index_1', 'num']).unstack()

    
    atoms.columns = [f'{col[0]}_{col[1]}' for col in atoms.columns]
    atoms = atoms.reset_index()  
    
     # downcast back to int8
    for col in atoms.columns:
        if col.startswith('atom_'):
            atoms[col] = atoms[col].fillna(0).astype('int8')
            
    atoms['molecule_index'] = atoms['molecule_index'].astype('int32')

    
  
  
    # more cleanup
    x_cols = [z for z in atoms.columns if z.startswith('x_')]
    y_cols = [z for z in atoms.columns if z.startswith('y_')]
    z_cols = [z for z in atoms.columns if z.startswith('z_')]
    a_cols = [z for z in atoms.columns if z.startswith('atom_')]
    atoms = atoms.rename( columns={'C_2': 'C', 'F_2': 'F', 'H_2': 'H', 'N_2': 'N', 'O_2': 'O', 
                    'total atoms_2': 'total atoms', 'Ox balance_2': 'Ox balance'})
    
    misc_features = ['molecule_index', #'atom_index_0', 'atom_index_1', 
                     'C', 'F', 'H', 'N', 'O', 'total atoms', 'Ox balance']
    keep_cols = x_cols + y_cols + z_cols + a_cols + misc_features
   
    
    atoms = atoms[keep_cols]
     
    
    
   
   
    # merge it all back together into coupling df
    df = pd.merge(bonds, atoms, how='inner',
                  on=['molecule_index', 'atom_index_0', 'atom_index_1'],
                  suffixes=('','_junk'))
    df = df.drop([col for col in df.columns if '_junk' in col],axis=1)

    
    
    
    # calculate distances from x,y,z of each atom and cleanup
    add_distances(df)
    add_coulombs(df)
    add_yukas(df)
    add_vanders(df)

      
    df = df.fillna(0)
    df.drop(columns=['atom_index_0', 'atom_index_1'], inplace=True)
    df.drop(['atom_0', 'atom_1'], axis=1, inplace=True)
    
    x_col0 = df.columns[df.columns.str.startswith('atom_0')]
    x_col1 = df.columns[df.columns.str.startswith('atom_1')]
    
    df.drop(x_col0, axis=1, inplace=True)
    df.drop(x_col1, axis=1, inplace=True)
    
    
    
    # bond energies
   
    
    df['N/C'] = np.where(df['C'] > 0, df['N'] / df['C'], 0)
    df['O/C'] = np.where(df['C'] > 0, df['O'] / df['C'], 0)
    df['H/C'] = np.where(df['C'] > 0, df['H'] / df['C'], 0)
    df['N-O'] = df['N'] - df['O']
    df['C-N'] = df['C'] - df['N']
    df['C-F'] = df['C'] - df['F']
    df['C-N-H'] = df['C'] - df['N'] - df['H']
    
    df['C-N*O -C*N'] = df['C'] - df['N']*df['O'] - df['C']*df['N']
    
    
   
    return df
    
    
    
    


# #### plots

# In[ ]:


# check predicted against actual - perfect score is line at 45'
# scores alone don't contain enough information, 
#   eyeball predicted validation against actual validation
#   can show trouble spots, outlier problems....

def plot_output(actual, predicted, coupling_type):
  
    print('plotting output', actual.shape, predicted.shape, coupling_type)
  
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.xlim(actual.min(), actual.max())
    plt.ylim(actual.min(), actual.max())

    plt.scatter(actual, predicted)
    plt.title(coupling_type)
    plt.grid(True)
   
    plt.show()


# In[ ]:


# chart NN convergence progress
def plot_history(history, label):
    
    fig, ax = plt.subplots(figsize=(8, 8))
        
    # remove first few points to take initial bounce off plots    
    plt.plot(history.history['loss'][20:])
    plt.plot(history.history['val_loss'][20:])
    
    plt.title('Loss for %s' % label)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    _= plt.legend(['Train','Validation'], loc='upper left')
    
    plt.show()
    
    
   


# In[ ]:



def build_and_split_data(some_csv, coupling_type, n_atoms):
    
    df = build_df(some_csv, struct, coupling_type, n_atoms=n_atoms)
        
    molecule_index = df['molecule_index'].values
    df.drop('molecule_index', axis=1, inplace=True)
    
         
    # features
    #atom_features = [z for z in df.columns if z.startswith('atom_')]
    distance_features = [z for z in df.columns if z.startswith('d_')]
    coulomb_features = [z for z in df.columns if z.startswith('c_')]
    vanderwaal_features = [z for z in df.columns if z.startswith('v_')]
    yuka_features = [z for z in df.columns if z.startswith('yk_')]
    bond_features = ['N/C', 'O/C', 'H/C', 'N-O', 'C-N', 'C-F', 'C-N-H', 'C-N*O -C*N']
    misc_features = ['C', 'F', 'H', 'N', 'O', 'total atoms', 'Ox balance']



    features = coulomb_features + vanderwaal_features + yuka_features + misc_features + bond_features
    #features = atom_features + distance_features
    
        
    df = df.fillna(0)
    
    
    if 'scalar_coupling_constant' in df:
        y_data = df['scalar_coupling_constant'].values.astype('float32')
        X_data = df[features].values.astype('float32')

    else:
        X_data = df[features].values.astype('float32')
        y_data = None
       
    
    X_data = StandardScaler().fit_transform(X_data)
    print(X_data.shape)
    
    return X_data, y_data


# #### model parameters

# In[ ]:


# create NN
def create_nn_model(input_shape):
    
    inp = Input(shape=(input_shape,))
    
    x = Dense(2048, activation="relu")(inp)
    x = BatchNormalization()(x)
    #x = Dropout(0.1)(x)
    
    
    x = Dense(2048, activation="relu")(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.1)(x)
    
    
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    
    
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)

    
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)

    
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)

    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    
    
    out = Dense(1, activation="linear")(x)  
    
    model = Model(inputs=inp, outputs=[out])
    
    return model


# In[ ]:




# set up NN
mol_types = sorted(train["type"].unique())
cv_score = []
cv_score_total = 0
epoch_n = 200
verbose = 1
batch_size = 256


    
# Set to True if we want to train from scratch.  False will reuse saved models as a starting point.
retrain =True

# check time per run
start_time = datetime.now()

# set up submission file
test_prediction = np.zeros(len(test))


# #### build and train model

# In[ ]:





#####################################################################################
# train a different model for each coupling type
# Loop through each molecule type
for mol_type in mol_types:

    # use to save best model for each coupling type
    model_name_wrt = ('molecule_model_%s.hdf5' % mol_type)
    print('Training %s' % mol_type, 'out of', mol_types, '\n')

    #################################################################################
    # data prep
    print(f'*** Training Model for {mol_type} ***')
    
    
    if mol_type == '1JHC': n_atoms = 14
    elif mol_type == '1JHN': n_atoms = 7
    elif mol_type == '2JHC': n_atoms = 14
    elif mol_type == '2JHH': n_atoms = 12
    elif mol_type == '2JHN': n_atoms = 12
    elif mol_type == '3JHC': n_atoms = 14
    elif mol_type == '3JHH': n_atoms = 9
    elif mol_type == '3JHN': n_atoms = 12

    if mol_type == '1JHC': stop_dx = 0.00001
    elif mol_type == '1JHN': stop_dx = 0.0001
    elif mol_type == '2JHC': stop_dx = 0.00001
    elif mol_type == '2JHH': stop_dx = 0.00001
    elif mol_type == '2JHN': stop_dx = 0.00001
    elif mol_type == '3JHC': stop_dx = 0.00001
    elif mol_type == '3JHH': stop_dx = 0.0001
    elif mol_type == '3JHN': stop_dx = 0.001

    # create dataset from input files and split  into train/test 
    
    X_train, y_train = build_and_split_data(train, mol_type, n_atoms)
    test_input, _ = build_and_split_data(test, mol_type, n_atoms)
    
       
    
    # Simple split to provide us a validation set to do our CV checks with
    # set random number for same split, better to shuffle and run different 
    # might show weirdenesses otherwise hidden by unfortunate split
    train_index, cv_index = train_test_split(np.arange(len(X_train)), test_size=0.1)
    
    # Split all our input and targets by train and cv indexes
    train_target = y_train[train_index]
    cv_target = y_train[cv_index]
    
    train_input = X_train[train_index]
    cv_input = X_train[cv_index]
    
    
    
    #test_input = input_data[len(df_train_):,:]
    #y_pred = np.zeros(train_target.shape[0], dtype='float32')

    ################################################################################
    # Build the Neural Net 
    nn_model = create_nn_model(train_input.shape[1])
   
        
        
    # compile model    
    nn_model.compile(loss='mse', optimizer=Adam())   

    
    # Callback for Early Stopping... May want to raise the min_delta for small numbers of epochs
    es = callbacks.EarlyStopping( monitor = 'val_loss', 
                                 min_delta = stop_dx, 
                                 patience = 30, 
                                 verbose = 1, 
                                 mode = 'auto', 
                                 restore_best_weights = True)
    
    # Callback for Reducing the Learning Rate... when the monitor levels out for 'patience' epochs, then the LR is reduced
    rlr = callbacks.ReduceLROnPlateau( monitor = 'val_loss',
                                      factor = 0.5, 
                                      patience = 20, 
                                      min_lr = 1e-6, 
                                      mode = 'auto', 
                                      verbose = 1)
    
    # Save the best value of the model for future use
    sv_mod = callbacks.ModelCheckpoint(model_name_wrt, monitor='val_loss', save_best_only=True, period=1)
    
    ################################################################################
    # train network
    # save convergence history for plots - can use to check overfitting
    history = nn_model.fit(train_input, [train_target], 
                                validation_data = (cv_input, [cv_target]), 
                                callbacks = [rlr, sv_mod], 
                                epochs = epoch_n, 
                                shuffle = True,
                                batch_size = batch_size, 
                                verbose = verbose)
    
    ################################################################################
    # check validation data
    cv_predict = nn_model.predict(cv_input)
    plot_output(cv_predict, cv_target, mol_type)
    
    # plot convergence
    plot_history(history, mol_type)
    accuracy = np.mean(np.abs(cv_target-cv_predict[:,0]))
    print(np.log(accuracy))
    
    # save scores
    cv_score.append(np.log(accuracy))
    cv_score_total += np.log(accuracy)
    
    # Predict on the test data set using our trained model
    test_predict = nn_model.predict(test_input)
    
    ################################################################################
    # for each molecule type grab the predicted values
    test_prediction[test["type"] == mol_type] = test_predict[:,0]
    
    # reset
    K.clear_session()

cv_score_total /= len(mol_types)


# #### prep submission data

# In[ ]:


# check scores

for i in range(len(mol_types)):
    print(mol_types[i], cv_score[i])
    
print('Final ', sum(cv_score) / len(mol_types))


# In[ ]:


# create submission file

def submits(predictions):

    submission['scalar_coupling_constant'] = predictions
    print(submission.head(30))

    submission.to_csv('nn_submission_3.csv')
    
submits(test_prediction) 
    
print('finished')

