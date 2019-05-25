import keras.backend as K
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.models import load_model, Model
from keras.optimizers import Adam
import tensorflow as tf 
import numpy as np
from utils import create_dataset
from keras.utils import to_categorical
import json

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

def train(X,Y,vocab_size,Tx, n_a,epochs=5000, lr=.01,beta_1=0.9, beta_2=0.999, decay=0.01):
	model,LSTM_cell,predictor = _get_model(n_a,Tx,vocab_size)
	opt = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

	m = X.shape[0]
	a0 = np.zeros((m, n_a))
	c0 = np.zeros((m, n_a))

	model.fit([X, a0, c0], list(Y), epochs=epochs)

	return model,LSTM_cell,predictor

X,Y, vocab_size, ix_to_char, char_to_ix = create_dataset('dinos.txt')
print(vocab_size)
n_a, Tx = 256,30
model,_,_ = train(X,Y,vocab_size,Tx,n_a,epochs=10000)

x_initializer = np.zeros((1, 1, vocab_size))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

model.save_weights('inference_model_weights.h5')
conf = {'n_a':n_a, 'vocab_size':vocab_size, 'ix_to_char':ix_to_char}
with open('config.json','w') as f:
	json.dump(conf,f)
# results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
# print(results)
# print("np.argmax(results[12]) =", np.argmax(results[12]))
# print("np.argmax(results[17]) =", np.argmax(results[17]))
# print("list(indices[12:18]) =", list(indices[12:18]))

# name=""
# for i in range(results.shape[0]):
# 	pred = np.argmax(results[i])
# 	char = char_to_ix[pred]
# 	name += char
# 	if char == '\n':
# 		break

# print(name)