import tensorflow as tf 
from keras.utils import to_categorical
import numpy as np
import keras.backend as K
from utils import create_dataset
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.models import load_model, Model

#with open('config.json','r') as f:
#  config = json.load(f)
#  vocab_size = config['vocab_size']
#  n_a = config['n_a']
#  ix_to_char = config['ix_to_char']
n_a = 256
X,Y, vocab_size, ix_to_char, char_to_ix = create_dataset('dinos.txt')

def one_hot(x,vocab_size=28):
    #x = np.random.choice(a = vocab_size, p = x.ravel())
    x = tf.multinomial(tf.log(x), 1)
    x = tf.cast(x[0][0],tf.int32)
    x = tf.one_hot(x, vocab_size)
    x = tf.reshape(x,[1,1,-1])
    #x = tf.expand_dims(x,0)#RepeatVector(1)(x)
    return x

def inference_model(n_a,vocab_size,char_to_ix):
  x0 = Input(shape=(1, vocab_size))
  a0 = Input(shape=(n_a,), name='a0')
  c0 = Input(shape=(n_a,), name='c0')

  #reshapor = Reshape((1,vocab_size))
  LSTM_cell = LSTM(n_a, return_state = True)
  predictor = Dense(vocab_size, activation='softmax')

  a = a0
  c = c0
  x = x0

  outputs = []
    
  it = 0
  # Step 2: Loop over Ty and generate a value at every time step
  while it <= 25 and (it==0 or not K.argmax(out) == char_to_ix['\n']):
    #print(it)
    a, _, c =  LSTM_cell(x, initial_state=[a, c])
    out = predictor(a)
    outputs.append(out)
    x = Lambda(one_hot)(out)
    it += 1
        
  inference_model = Model(inputs=[x0,a0,c0],outputs=outputs)
    
  return inference_model


def predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer, ix_to_char ,vocab_size=29):
  pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
  indices = np.expand_dims(np.argmax(pred, axis = -1),0)
  indices = indices.swapaxes(1,2)
  name = ''.join([ix_to_char[i] for i in indices[0][0]])
  print(name)
  return indices

inference_model = inference_model(n_a,vocab_size,char_to_ix)
inference_model.summary()
inference_model.load_weights('inference_model_weights.h5')

x_initializer = np.zeros((1, 1, vocab_size))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

for i in range(1):
	name = predict_and_sample(inference_model,x_initializer,a_initializer,c_initializer,ix_to_char,vocab_size)
	print(name)
