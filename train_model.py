from keras.utils import to_categorical
import keras.backend as K
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.models import load_model, Model

import tensorflow as tf 
import numpy as np

def _get_model(num_units, Tx, vocab_size):
	X = Input(shape=(Tx, vocab_size),name='X')
	a0 = Input(shape=(num_units,),name='a0')
	c0 = Input(shape=(num_units,),name='c0')

	reshapor = Reshape((1,vocab_size))
	LSTM_cell = LSTM(num_units, return_state = True)
	predictor = Dense(vocab_size, activation='softmax')

	a = a0
	c = c0

	outputs = []
	for t in range(Tx):
		x = Lambda(lambda x: X[:,t,:])(X)
		x = reshapor(x)
		a,_,c = LSTM_cell(x, initial_state=[a, c])
		out = predictor(a)
		outputs.append(out)

	model = Model(inputs=[X,a0,c0],outputs=outputs)

	return model,LSTM_cell,predictor