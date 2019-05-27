import numpy as np

def fill_names(name,init_value,fill_value,M):
	"""
	Fill a string with the char fill_value until reaches the maximum size M. The string init with init_value.
	"""
	N = len(name)
	K = M-N-1
	rest = fill_value*K
	return init_value+name+rest

def create_name_dataset(path, Tx=30):
	"""
	Receive a file formatted as follow:
		Each name is sperated by '\n', so each line contains exactly a name.
	Arguments:
		path: Path to the names files
		Tx: number of maximum characters, which allows to pad the names.
	Returns:
		X: train data (one_hot_encoded)
		Y: train data shifted one time step to the left and terminated with '\n'.
		vocab_size: Number of chars in the vocabulary
		ix_to_char, char_to_ix: Maps from char/int to int/char, where the int represent the one hot encoding index of the char
	"""
	with open(path,'r') as f:
		data = f.read()
	data = data.lower()
	chars = list(set(data))
	data_size, vocab_size = len(data), len(chars)+1
	print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

	char_to_ix = { ch:i for i,ch in enumerate(chars) }
	ix_to_char = { i:ch for i,ch in enumerate(chars) }

	names = [fill_names(x,'','\n',Tx) for x in data.split('\n')]
	M = len(names)

	X = np.zeros([M,Tx,vocab_size])
	Y = np.zeros([M,Tx,vocab_size])

	for i,name in enumerate(names):
		for j,char in enumerate(name):
			idx = char_to_ix[char]
			X[i,j,idx] = 1.
			if  j>0:
				Y[i,j-1,idx] = 1.

	Y = np.swapaxes(Y,0,1)
	Y = np.asarray(Y.tolist())
	return X,Y, vocab_size, ix_to_char, char_to_ix

