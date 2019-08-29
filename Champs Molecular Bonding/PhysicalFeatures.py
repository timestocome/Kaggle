


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.options.display.width = 0



###############################################################################
# utility functions and constants
###############################################################################

coupling_types = ['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']
epsilon = 1e-9   # not zero



train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
structure = pd.read_csv('data/structures.csv')




###############################################################################
# add a few features to structure before merging
###############################################################################

structure['cx'] = structure.groupby(['molecule_name'])['x'].transform('mean')
structure['cy'] = structure.groupby(['molecule_name'])['y'].transform('mean')
structure['cz'] = structure.groupby(['molecule_name'])['z'].transform('mean')




###############################################################################
# map location info from structure file into train, test data sets
###############################################################################

# https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark
def map_xyz(df, atom_idx):

	df = pd.merge(df, structure, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
	df.drop('atom_index', axis=1, inplace=True)
	df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x{atom_idx}',
                            'y': f'y{atom_idx}',
                            'z': f'z{atom_idx}'})
	return df


###############################################################################
# https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark
def get_distance(df):

	p0 = df[['x0', 'y0', 'z0']].values
	p1 = df[['x1', 'y1', 'z1']].values
	
	df['r'] = np.linalg.norm(p0 - p1, axis=1)

	# drop intermediary calculations
	#df.drop(columns=['x0', 'x1', 'y0', 'y1', 'z0', 'z1'], inplace=True)

	# calculate center of each molecule
	df.drop(columns=['cx_y', 'cy_y', 'cz_y'], inplace=True)
	df = df.rename(columns={'cx_x': 'cx', 
				'cy_x': 'cy',
				'cz_x': 'cz'})

	# get this atom's distance from center ( see Keras NN for Champs kernel )
	c = df[['cx', 'cy', 'cz']].values

	df['dist_center_0'] = np.linalg.norm(c - p0, axis=1)
	df['dist_center_1'] = np.linalg.norm(c - p1, axis=1)


	return df



###############################################################################
# merge in x, y, z 
print('merging x,y,z into main dbs')


# fold x,y,z coordinates into train and test files
train = map_xyz(train, 0)
train = map_xyz(train, 1)

train.drop(['id'], inplace=True, axis=1)
train = get_distance(train)

test = map_xyz(test, 0)
test = map_xyz(test, 1)

test = get_distance(test)







###############################################################################
# get number of bonds and type per atom
###############################################################################
print('getting number of bonds per atom and merging into main dbs')

# get bond count by molecule, atom, bond type for atom_0
bonds = train.groupby(['molecule_name', 'atom_index_0', 'type'], as_index=False).count()
bonds = bonds[['molecule_name', 'atom_index_0', 'type', 'atom_0']]
bonds.columns = ['molecule_name', 'atom_index_0', 'type', 'bonds_0']

train = train.merge(bonds, how='left', left_on=['molecule_name', 'atom_index_0', 'type'],
				right_on=['molecule_name', 'atom_index_0', 'type'])



# get bond count by molecule, atom, bond type for atom_1
bonds = train.groupby(['molecule_name', 'atom_index_1', 'type'], as_index=False).count()
bonds = bonds[['molecule_name', 'atom_index_1', 'type', 'atom_1']]
bonds.columns = ['molecule_name', 'atom_index_1', 'type', 'bonds_1']

train = train.merge(bonds, how='left', left_on=['molecule_name', 'atom_index_1', 'type'],
				right_on=['molecule_name', 'atom_index_1', 'type'])



###############################################################################
# get bond count by molecule, atom, bond type for atom_0
bonds = test.groupby(['molecule_name', 'atom_index_0', 'type'], as_index=False).count()
bonds = bonds[['molecule_name', 'atom_index_0', 'type', 'atom_0']]
bonds.columns = ['molecule_name', 'atom_index_0', 'type', 'bonds_0']

test = test.merge(bonds, how='left', left_on=['molecule_name', 'atom_index_0', 'type'],
				right_on=['molecule_name', 'atom_index_0', 'type'])



# get bond count by molecule, atom, bond type for atom_1
bonds = test.groupby(['molecule_name', 'atom_index_1', 'type'], as_index=False).count()
bonds = bonds[['molecule_name', 'atom_index_1', 'type', 'atom_1']]
bonds.columns = ['molecule_name', 'atom_index_1', 'type', 'bonds_1']

test = test.merge(bonds, how='left', left_on=['molecule_name', 'atom_index_1', 'type'],
				right_on=['molecule_name', 'atom_index_1', 'type'])




###############################################################################
# get bonded atoms count per atom
print('getting bonds by type per atom and merging into main db')

for ct in coupling_types:

	print(ct)
	single_h = train[train['type'] == ct]

	# train atom_idx_0
	column_0 = 'n_' + ct + '_0'
	single_bonds_h = single_h.groupby(['molecule_name', 'atom_index_0', 'type'], as_index=False).count()
	single_bonds_h = single_bonds_h[['molecule_name', 'atom_index_0', 'type', 'atom_0']]
	single_bonds_h.columns = ['molecule_name', 'atom_index_0', 'type', column_0]

	train = train.merge(single_bonds_h, how='left', left_on=['molecule_name', 'atom_index_0', 'type'],
				right_on = ['molecule_name', 'atom_index_0', 'type'])



	# train atom_idx_1
	column_1 = 'n_' + ct + '_1'
	single_bonds_h = single_h.groupby(['molecule_name', 'atom_index_1', 'type'], as_index=False).count()
	single_bonds_h = single_bonds_h[['molecule_name', 'atom_index_1', 'type', 'atom_1']]
	single_bonds_h.columns = ['molecule_name', 'atom_index_1', 'type', column_1]

	train = train.merge(single_bonds_h, how='left', left_on=['molecule_name', 'atom_index_1', 'type'],
				right_on = ['molecule_name', 'atom_index_1', 'type'])




	# test atom_idx_0
	single_h = test[test['type'] == ct]
	single_bonds_h = single_h.groupby(['molecule_name', 'atom_index_0', 'type'], as_index=False).count()
	single_bonds_h = single_bonds_h[['molecule_name', 'atom_index_0', 'type', 'atom_0']]
	single_bonds_h.columns = ['molecule_name', 'atom_index_0', 'type', column_0]

	test = test.merge(single_bonds_h, how='left', left_on=['molecule_name', 'atom_index_0', 'type'],
				right_on = ['molecule_name', 'atom_index_0', 'type'])



	# test atom_idx_1
	single_bonds_h = single_h.groupby(['molecule_name', 'atom_index_1', 'type'], as_index=False).count()
	single_bonds_h = single_bonds_h[['molecule_name', 'atom_index_1', 'type', 'atom_1']]
	single_bonds_h.columns = ['molecule_name', 'atom_index_1', 'type', column_1]

	test = test.merge(single_bonds_h, how='left', left_on=['molecule_name', 'atom_index_1', 'type'],
				right_on = ['molecule_name', 'atom_index_1', 'type'])



	



train.fillna(0, inplace=True)
test.fillna(0, inplace=True)





###############################################################################

def get_features(df):

	# roughly estimate some forces
	# roughly estimate some distance based forces
	def get_square(z): return z **2
	def get_cube(z): return z **3
	def get_exp(z): return np.exp(-z)/z
	def get_sqrt(z): return np.sqrt(z)


	df['r^2'] = df['r'].apply(get_square)
	df['r^3'] = df['r'].apply(get_cube)
	df['r^.5'] = df['r'].apply(get_sqrt)


	df['1/r^2'] = 1. / df['r^2']
	df['1/r^3'] = 1. / df['r^3']
	df['1/r'] = 1. / df['r']
	df['e^-r/r'] = np.exp(-df['r'] )/  df['r']



	df['closest_r_0'] = df.groupby(['molecule_name', 'atom_index_0'])['r'].transform('min')
	df['closest_r_1'] = df.groupby(['molecule_name', 'atom_index_1'])['r'].transform('min')
	
	df['dx'] = df['x0'] - df['x1']
	df['dy'] = df['y0'] - df['y1']
	df['dz'] = df['z0'] - df['z1']

	df['cos 0'] = df['r'] / df['dist_center_0']
	df['cos 1'] = df['r'] / df['dist_center_1']


	# get the number of atoms in each molecule
	atoms = df.groupby('molecule_name')['atom_index_0', 'atom_index_1'].max()
	atoms['n_atoms'] = np.where(atoms['atom_index_0'] > atoms['atom_index_1'], atoms['atom_index_0'], atoms['atom_index_1'])

	df = df.merge(atoms, how='left', left_on=['molecule_name'], right_on=['molecule_name'])
	df.drop(columns=['atom_index_0_y', 'atom_index_1_y'], inplace=True)
	df = df.rename(columns={'atom_index_0_x': 'atom_index_0', 'atom_index_1_x': 'atom_index_1'})




	# location of electons in shells for H, C, N
	def get_1s(atom):
		if atom == 'H': return 1
		else: return 2

	def get_2s(atom):
		if atom == 'H': return 0
		else: return 2

	def get_2p(atom):
		if atom == 'H': return 0
		elif atom == 'C': return 2
		else: return 3

	df['1s'] = df['atom_1'].apply(get_1s)
	df['2s'] = df['atom_1'].apply(get_2s)
	df['2p'] = df['atom_1'].apply(get_2p)
	
	
	# hydrogen 0 neutrons
	# carbon 6
	# Nitrogen 7

	# hydrogen  1.008 - 2
	# carbon 12.01 - 13
	# nitrogen 16.01 - 17
	def get_mass(atom):
		if atom == 'H': return 2
		elif atom == 'C': return 13
		elif atom == 'N': return 17
	df['mass_1'] = df['atom_1'].apply(get_mass) 
	





	# valence electrons
	def get_valence(atom):
		if atom == 'H': return 1
		elif atom == 'C': return 4
		elif atom == 'N': return 5


	df['valence_electrons_1'] = df['atom_1'].apply(get_valence)
	
	def get_protons(atom):
		if atom == 'H': return 1
		elif atom == 'C': return 6
		else: return 7

	def get_neutrons(atom):
		if atom == 'H': return 0
		elif atom == 'C': return 7   # carbon 12, 6:  carbon 13 too? 7 neutrons
		else: return 7


	df['protons'] = df['atom_1'].apply(get_protons)
	df['protons_odd'] = df['protons'] % 2
	
	df['neutrons'] = df['atom_1'].apply(get_neutrons)
	df['neutrons_odd'] = df['neutrons'] % 2

	df['protons_neutrons_odd'] = df['protons_odd'] + df['neutrons_odd']

	df['molecule_s1_electrons'] = df.groupby('molecule_name')['1s'].transform('sum')
	df['molecule_s2_electrons'] = df.groupby('molecule_name')['2s'].transform('sum')
	df['molecule_p2_electrons'] = df.groupby('molecule_name')['2p'].transform('sum')



	# convert atom abreviation to integer
	def convert_atom(a):
		if a == 'H': return 0
		if a == 'C': return 1
		if a == 'N': return 2
		return -1
	

	#df['atom0'] = df['atom_0'].apply(convert_atom)   # they're all H
	df['atom1'] = df['atom_1'].apply(convert_atom)


	# bond strength
	def get_bonds(t):
	
		if t == '1JHC' or t == '1JHN': return 1
		elif t == '2JHC' or t == '2JHH' or t == '2JHN': return 2
		elif t == '3JHC' or t == '3JHH' or t == '3JHN': return 3
		
		return -1

	df['n_bond'] = df['type'].apply(get_bonds)


	# convert type to one hot
	df['1jhc'] = np.where(df['type'] == '1JHC', 1, 0)
	df['1jhn'] = np.where(df['type'] == '1JHN', 1, 0)
	df['2jhc'] = np.where(df['type'] == '2JHC', 1, 0)
	df['2jhh'] = np.where(df['type'] == '2JHH', 1, 0)
	df['2jhn'] = np.where(df['type'] == '2JHN', 1, 0)
	df['3jhc'] = np.where(df['type'] == '3JHC', 1, 0)
	df['3jhh'] = np.where(df['type'] == '3JHH', 1, 0)
	df['3jhn'] = np.where(df['type'] == '3JHN', 1, 0)
	
	# drop index marker
	df.drop(columns=['atom_index_0', 'atom_index_1'], inplace=True)
	
	# drop cartesian in favor of polar
	df.drop(columns=['dx', 'dy', 'dz', 'x0', 'x1', 'y0', 'y1', 'z0', 'z1', 'cx', 'cy', 'cz', 'cos 1'], inplace=True)

	return df


train = get_features(train)
test = get_features(test)



# sanity check
print(train.describe())
print(test.describe())


# see what's useful, drop the rest
print(np.abs(train.corr()).sort_values('scalar_coupling_constant', ascending=False)[['scalar_coupling_constant']])



# save to use as input to StatsFeatures.py
train.to_csv('train_features.csv')
test.to_csv('test_features.csv')



print(train.describe())













































