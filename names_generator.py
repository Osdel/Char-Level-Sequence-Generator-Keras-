import tensorflow as tf 
from keras.utils import to_categorical
import numpy as np
import keras.backend as K
from utils import create_dataset
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.models import load_model, Model
import json

import argparse
import sys
import os

def one_hot(vocab_size=28):
  """Sample the next character from distribution of x, this is like np.random.choice(np.arange(vocab_size),x.ravel())
      but using tensors. Then the choosen character is converted into one_hot encoding and reshaped into the model's input shape.
      Note that the function was splitted into 2 functions, cause the wrapper function is what the Keras Lambda layers, 
      is waiting. So to use the extra_parameter vocab_size the outer function 
      Arguments:
        x:Tensor of shape (1,vocab_size) where each index i represent the probability of the character i-th\
          to be the next in the current sequence
        vocab_size:Number of characters in the sequence
      """
  def wrapper(x):
      x = tf.random.categorical(tf.log(x),1)
      x = tf.cast(x[0][0],tf.int32)
      x = tf.one_hot(x, vocab_size)
      #Reshape to the input's shape (1,1,vocab_size)
      x = tf.reshape(x,[1,1,-1])
      return x
  return wrapper

def inference_model(Ty, n_a,vocab_size):
  """
  Build the inference model.
  Inputs:
    Ty: Number of chars to generate (lenght of the sequence)
    n_a: Number of neurons of the LSTM cell
    vocab_size: Size of the vocabulary
  """
  x0 = Input(shape=(1, vocab_size))
  a0 = Input(shape=(n_a,), name='a0')
  c0 = Input(shape=(n_a,), name='c0')

  LSTM_cell = LSTM(n_a, return_state = True)
  predictor = Dense(vocab_size, activation='softmax')

  a = a0
  c = c0
  x = x0

  outputs = []
    
  #Loop over Ty and generate a value at every time step
  for it in range(Ty):
    a, _, c =  LSTM_cell(x, initial_state=[a, c])
    out = predictor(a)
    outputs.append(out)
    x = Lambda(one_hot(vocab_size))(out)
  
  inference_model = Model(inputs=[x0,a0,c0],outputs=outputs)
    
  return inference_model


def predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer ,ix_to_char,vocab_size=29):
  """
    Predicts the next value of values using the inference model.
    
    Arguments:
      inference_model -- Keras model instance for inference time
      x_initializer -- numpy array of shape (1, 1, vocab_size), one-hot vector initializing the values generation
      a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
      c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    
    Returns:
      indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
      generated_text -- string representation of the values generated
    """
  pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
  indices = np.expand_dims(np.argmax(pred, axis = -1),0)
  indices = indices.swapaxes(1,2)
  generated_text = ''.join([ix_to_char[i] for i in indices[0][0]])
  
  return indices, generated_text

def generate(Ty,samples_to_generate,n_a,vocab_size,ix_to_char, post_processing_fn=None):
  """
  Main API function. Generate n samples using the inference model.
  Arguments:
    Ty: Lenght of the sequence to generate.
    samples_to_generate: Number of predictions or samples to generate
    n_a: Number of units of the LSTM cell
    ix_to_char: dictionary which maps from int to char, where the int represent the one_hot encoding index 
  """
  model = inference_model(Ty,n_a,vocab_size)
  model.load_weights('inference_model_weights.h5')

  a_initializer = np.zeros((1, n_a))
  c_initializer = np.zeros((1, n_a))

  outputs = []
  for i in range(samples_to_generate):
    #randomizing the first character
    x_initializer = np.zeros((1, 1, vocab_size))
    x_initializer[0,0,np.random.choice(np.arange(vocab_size))]=1.

    _,generated_text = predict_and_sample(model,x_initializer,a_initializer,c_initializer,ix_to_char,vocab_size)

    #Using the post_processing function over the generated sequence
    if post_processing_fn is not None:
      generated_text = post_processing_fn(generated_text)

    outputs.append(generated_text)

  return outputs


def main(Ty,samples_to_generate,config_path='config.json'):
  with open(config_path,'r') as f:
    config = json.load(f)
    vocab_size = config['vocab_size']
    n_a = config['n_a']
    ix_to_char = {int(x):config['ix_to_char'][x] for x in config['ix_to_char']}

  outputs = generate(Ty,samples_to_generate,n_a,vocab_size,ix_to_char,\
    post_processing_fn=lambda x:x.split('\n')[0])
  for g in outputs:
    print(g)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate character-level sequences using a pretrained keras model')
  parser.add_argument('conf', type=str, help='path to the pretrained model configuration (json)')
  parser.add_argument('ty',type=int, help='lenght of the generated sequence')
  parser.add_argument('n', type=int, help='number of sequences to generate')
  args = parser.parse_args(sys.argv[1:])
  main(args.ty,args.n,args.conf)
