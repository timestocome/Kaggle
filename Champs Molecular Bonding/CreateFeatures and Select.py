#!/usr/bin/env python
# coding: utf-8

# This kernel is loosely based on ideas from 
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
# ### This kernel is used to evaluate features - it grossly overfits on submission data
# 
# ####  Improvements
# * added feature importance plots
# * added plts of predicted vs actual in validation data
# * added functions to calculate VanDerWaals, Coulomb, Yukawa forces
# * cleaned and improved db manipulations, more could be done
# 
# * this ML was just used as a feature selector. The data isn't really suitable for a tree algorithm
# 
# 
# * Best possible score ~ -20
# * Winning score -3.2
# * My best -1.3
# 

# In[1]:




import pandas as pd
import numpy as np


# In[2]:



import math
import gc
import copy

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from lightgbm import LGBMRegressor


# In[3]:



import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
warnings.filterwarnings(action="ignore",category=FutureWarning)


# ### read in raw datafiles

# In[4]:


train = pd.read_csv('Data/train.csv', index_col=0)

# remove batch name from molecule id
train['molecule_index'] = train.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
train.drop(['molecule_name'], axis=1, inplace=True)


coupling_types = sorted(train['type'].unique())


print(train.head())


# In[5]:


test = pd.read_csv('Data/test.csv', index_col=0)


# remove batch name from molecule id
test['molecule_index'] = test.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
test.drop(['molecule_name'], axis=1, inplace=True)




print(test.head())


# In[6]:


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

print(u_atoms.head(20))

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





# In[7]:


submission = pd.read_csv('Data/sample_submission.csv', index_col='id')

print(submission.head())


# ### prep data for model

# 

# 
# #### split out data types and merge with location db

# In[8]:



# pull coupling datatype from train/test and structures dfs
def fetch_type_data(df, struct, ctype):
    
    # pull out all rows with this coupling type from train data and make a copy
    df = df[df['type'] == ctype].drop('type', axis=1).copy()
    
    # pull out all rows with this coupling type from structs
    struct_df = struct[struct['molecule_index'].isin(df['molecule_index'])]
    
    
    return df, struct


# In[9]:



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

# In[10]:


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


# In[11]:


# calculate coulomb force distance between each atom 

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


# In[12]:


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


# In[13]:


# calculate vaderwaal's force distance between each atom 
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


# In[14]:




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
                  right_on=['molecule_index'])
    
    atoms.drop(columns=['C_x', 'F_x', 'H_x', 'N_x', 'O_x', 'total atoms_x', 'Ox balance_x'], inplace=True)
    atoms = atoms.rename( columns={'C_y': 'C', 'F_y': 'F', 'H_y': 'H', 'N_y': 'N', 'O_y': 'O', 
                    'total atoms_y': 'total atoms', 'Ox balance_y': 'Ox balance'})

   

    # remove dups ?
    atoms = atoms[(atoms.atom_index_0 != atoms.atom_index) & (atoms.atom_index_1 != atoms.atom_index)]
    
    # add distances for each atom to center of coupled pairs
    atoms['d_c'] = ((
        (atoms['x_c'] - atoms['x'])**np.float32(2) +
        (atoms['y_c'] - atoms['y'])**np.float32(2) + 
        (atoms['z_c'] - atoms['z'])**np.float32(2)
    )**np.float32(0.5))
    
    
    # remove center of coupled pairs
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
                  on=['molecule_index', 'atom_index_0', 'atom_index_1'])
    
    
    df = df.rename( columns={'C_y': 'C', 'F_y': 'F', 'H_y': 'H', 'N_y': 'N', 'O_y': 'O', 
                    'total atoms_y': 'total atoms', 'Ox balance_y': 'Ox balance'})


   
    
    
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
    
   
    df['N/C'] = df['N'] / df['C']
    df['O/C'] = df['O'] / df['C']
    df['H/C'] = df['H'] / df['C']
    df['N-O'] = df['N'] - df['O']
    df['C-N'] = df['C'] - df['N']
    df['C-F'] = df['C'] - df['F']
    df['C-N-H'] = df['C'] - df['N'] - df['H']
    df['C-N*O -C*N'] = df['C'] - df['N']*df['O'] - df['C']*df['N']
    
   
    return df
    
    
    
    


# #### plots

# In[15]:


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


# In[16]:


# https://www.kaggle.com/ashishpatel26/feature-importance-of-lightgbm
def plot_features(feature_imp):
    
    # sorted(zip(clf.feature_importances_, X.columns), reverse=True)
    #feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,X.columns)), columns=['Value','Feature'])

    
    plt.figure(figsize=(10, 30))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features')
    plt.tight_layout()
    
    plt.show()


# In[17]:



def build_and_split_data(some_csv, coupling_type, n_atoms):
    
    df = build_df(some_csv, struct, coupling_type, n_atoms=n_atoms)
        
    molecule_index = df['molecule_index'].values
    df.drop('molecule_index', axis=1, inplace=True)
    
         
    # features
    #atom_features = [z for z in df.columns if z.startswith('atom_')]
    distance_features = [z for z in df.columns if z.startswith('d_')]
    coulomb_features = [z for z in df.columns if z.startswith('c_')]
    vanderwaal_features = [z for z in df.columns if z.startswith('v_')]
    #yuka_features = [z for z in df.columns if z.startswith('yk_')]
    #misc_features = ['C', 'F', 'H', 'N', 'O', 'total atoms', 'Ox balance']
    #bond_features = ['N/C', 'O/C', 'H/C', 'N-O', 'C-N', 'C-F', 'C-N-H', 'C-N*O -C*N']

    misc_features = ['H', 'total atoms', 'Ox balance']
    bond_features = ['N/C', 'O/C', 'H/C', 'N-O', 'C-N-H', 'C-N*O -C*N']
    
    features = coulomb_features + vanderwaal_features  + misc_features + bond_features
    #features = atom_features + distance_features
    
        
    df = df.fillna(0)
    
    if 'scalar_coupling_constant' in df:
        y_data = df['scalar_coupling_constant'].values.astype('float32')
        X_data = df[features].values.astype('float32')

    else:
        X_data = df[features].values.astype('float32')
        y_data = None
       
    
    print(X_data.shape)
    
    return X_data, y_data, molecule_index, features


# #### model parameters

# In[18]:


# configuration params are copied from @artgor kernel:
# https://www.kaggle.com/artgor/brute-force-feature-engineering

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.2,
    'num_leaves': 128,
    'min_child_samples': 128,
    'max_depth': 7,
    'subsample_freq': 1,
    'subsample': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'colsample_bytree': 0.8
}


n_estimators = 2048
early_stopping = 32

model_params = {
    '1JHC': 14,
    '1JHN': 7,
    '2JHC': 14,
    '2JHH': 12,
    '2JHN': 12,
    '3JHC': 14,
    '3JHH': 9,
    '3JHN': 12
}


N_FOLDS = 3


# #### build and train model

# In[19]:


def train_and_predict (coupling_type, submission, n_atoms, n_folds=5, n_splits=5):
    
    print(f'*** Training Model for {coupling_type} ***')
    
    X_data, y_data, groups, features = build_and_split_data(train, coupling_type, n_atoms)
    X_test, _, _, _ = build_and_split_data(test, coupling_type, n_atoms)
    y_pred = np.zeros(X_test.shape[0], dtype='float32')
    

    cv_score = 0
    
    if n_folds > n_splits:
        n_splits = n_folds
    
    kfold = GroupKFold(n_splits=n_splits)

    for fold, (train_index, val_index) in enumerate(kfold.split(X_data, y_data, groups=groups)):
        
        if fold >= n_folds:
            break

        X_train, X_val = X_data[train_index], X_data[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]

        model = LGBMRegressor(**LGB_PARAMS, n_estimators=n_estimators, n_jobs = -1)
        
        model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mse',
            verbose=100, early_stopping_rounds=early_stopping)

        y_val_pred = model.predict(X_val)
        val_score = np.log(mean_absolute_error(y_val, y_val_pred))
        
        print(f'{coupling_type} Fold {fold}, logMAE: {val_score}')
        
        
        plot_output(y_val, y_val_pred, coupling_type)
        
        feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, features)), 
                               columns=['Value','Feature'])
        plot_features(feature_imp)
        
        
        cv_score += val_score
        y_pred += model.predict(X_test)
        
         
        break 
    
    return cv_score


# In[20]:



# loop over coupling types, build df, train model, get predictions

cv_scores = {}
for coupling_type in coupling_types:
    
    cv_score = train_and_predict(
        coupling_type, submission, n_atoms = model_params[coupling_type], n_folds=N_FOLDS)
 
    cv_scores[coupling_type] = cv_score

    
    
    


# #### prep submission data

# In[21]:


# check model accuracy


print( pd.DataFrame({'type': list(cv_scores.keys()), 'cv_score': list(cv_scores.values())}) )
print( np.mean(list(cv_scores.values())) )


