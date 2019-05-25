import numpy as np

def one_hot(idx,n):
	res = np.zeros((n,))
	res[idx] = 1.
	return res

def one_hot_to_name(mat, vocab_size, ix_to_char):
	name=""
	N = mat.shape[0]
	for i in range(N):
		print(i)
		print(np.sum(mat[i]))
		ix = np.random.choice([x for x in range(vocab_size)], p = np.ravel(mat[i]))
		name += ix_to_char[ix]
	return name

def fill_names(name,init_value,fill_value,M):
	"""
	Fill a string with a char (fill_value) until reaches the maximum size M
	"""
	N = len(name)
	K = M-N-1
	rest = fill_value*K
	return init_value + name+rest

def create_dataset(path, Tx=30):
	data = open(path,'r').read()
	data = data.lower()
	chars = list(set(data))
	print(chars)
	data_size, vocab_size = len(data), len(chars)+1
	print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))


	# The characters are a-z (26 characters) plus the "\n" (or newline character), which in this assignment plays a role similar to the `<EOS>` (or "End of sentence") token we had discussed in lecture, only here it indicates the end of the dinosaur name rather than the end of a sentence. In the cell below, we create a python dictionary (i.e., a hash table) to map each character to an index from 0-26. We also create a second python dictionary that maps each index back to the corresponding character character. This will help you figure out what index corresponds to what character in the probability distribution output of the softmax layer. Below, `char_to_ix` and `ix_to_char` are the python dictionaries. 

	# In[3]:

	#chars = sorted(chars+['_'])

	char_to_ix = { ch:i for i,ch in enumerate(chars) }
	ix_to_char = { i:ch for i,ch in enumerate(chars) }
	
	names = [fill_names(x,'','\n',Tx) for x in data.split('\n')]
	M = len(names)

	X = np.zeros([M,Tx,vocab_size])
	Y = np.zeros([M,Tx,vocab_size])

	for i,name in enumerate(names):
		#print(str.replace(name,'\n','0'))
		for j,char in enumerate(name):
			idx = char_to_ix[char]
			X[i,j,idx] = 1.
			if  j>0:
				Y[i,j-1,idx] = 1.

	Y = np.swapaxes(Y,0,1)
	Y = np.asarray(Y.tolist())
	return X,Y, vocab_size, ix_to_char, char_to_ix

#X,Y, vocab_size, ix_to_char, char_to_ix = create_dataset('dinos.txt')
# a = one_hot_to_name(X[0],vocab_size,ix_to_char)
# print(a)
# print(len(a))