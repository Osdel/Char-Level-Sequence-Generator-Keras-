from keras.optimizers import Adam
import numpy as np

from utils import create_name_dataset
from train_model import _get_model

import argparse
import sys
import os
import json


def train(X,Y,vocab_size,Tx, n_a,epochs=10000, lr=.01,beta_1=0.9, beta_2=0.999, decay=0.01):
	"""
	Train the model.
	Arguments:
		X: training samples of shape [num_samples,Tx,vocab_size].
		Y: training outputs, the same as X but shifted on time step to the left, and ends with a '\n'.
		vocab_size: Represent the size of the vocabulary.
		epochs: number of traning epochs.
		lr, beta_1, beta_2, decay: Adam optimizer hyperparameters.
	Returns:
		model: The Keras trained Model.
		LSTM_cell: The model LSTM cell
		predictor: The Dense Layer for prediction
	"""
	model,LSTM_cell,predictor = _get_model(n_a,Tx,vocab_size)
	opt = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

	m = X.shape[0]
	a0 = np.zeros((m, n_a))
	c0 = np.zeros((m, n_a))

	model.fit([X, a0, c0], list(Y), epochs=epochs)

	return model,LSTM_cell,predictor

def main(n_a,Tx,epochs,dataset_path='dinos.txt',config_out_path='config.json',weights_out_path='dino_weights.h5'):
	X,Y, vocab_size, ix_to_char, char_to_ix = create_name_dataset(dataset_path,Tx)
	print('Vocab size = '+ str(vocab_size))
	#n_a, Tx = 256,30
	model,_,_ = train(X,Y,vocab_size,Tx,n_a,epochs=epochs)

	x_initializer = np.zeros((1, 1, vocab_size))
	a_initializer = np.zeros((1, n_a))
	c_initializer = np.zeros((1, n_a))

	model.save_weights(weights_out_path)
	conf = {'n_a':n_a, 'vocab_size':vocab_size, 'ix_to_char':ix_to_char}
	with open(config_out_path,'w') as f:
		json.dump(conf,f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train character-level model for names generation')
	parser.add_argument('data', type=str, help='path to the names dataset file')
	parser.add_argument('conf', type=str, help='path to the output pretrained model configuration (json)')
	parser.add_argument('out', type=str, help='path to the output pretrained model weights (h5)')
	parser.add_argument('tx',type=int, help='max lenght of the input sequence')
	parser.add_argument('n_a', type=int, help='number of units of the LSTM cell')
	parser.add_argument('epochs', type=int, help='number of epoch of training')
	args = parser.parse_args(sys.argv[1:])
	main(args.n_a,args.tx,args.epochs,args.data,args.conf,args.out)
