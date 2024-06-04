# Machine Translation Project
## Artificial Intelligence Nanodegree
### Overview
This project involves building a deep neural network as part of an end-to-end machine translation pipeline. The pipeline will accept English text as input and return the corresponding French translation.

#### Table of Contents
Introduction

Setup

Dataset

Preprocessing

Models

Prediction

Submission

Optional Enhancements

#### Introduction
In this notebook, sections ending with '(IMPLEMENTATION)' indicate that the following blocks of code will require additional functionality which you must provide. 
You will:

Preprocess text by converting it to sequences of integers.
Create models to accept sequences of integers as input and return a probability distribution over possible translations.
Design and experiment with different neural network architectures.
Run the model on English text to predict French translations.
#### Setup
Ensure you have the necessary libraries installed, including keras, tensorflow, numpy, and other dependencies. Load the helper functions and tests.

```python
%load_ext autoreload
%aimport helper, tests
%autoreload 1

import collections
import helper
import numpy as np
import project_tests as tests
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Embedding, LSTM, concatenate
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


```

Verify access to the GPU, which is useful for training the neural network faster.

#### Dataset
The dataset used for this project is a smaller vocabulary set compared to the WMT datasets, allowing for reasonable training time.

Load Data
Load the English and French data from data/small_vocab_en and data/small_vocab_fr.

```python
english_sentences = helper.load_data('data/small_vocab_en')
french_sentences = helper.load_data('data/small_vocab_fr')
print('Dataset Loaded')
```

Files
Each line in small_vocab_en contains an English sentence, and each line in small_vocab_fr contains the respective French translation.

Vocabulary
Inspect the complexity of the dataset by analyzing the number of unique words in both English and French sentences.

python
Copy code
english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
print('{} unique French words.'.format(len(french_words_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')
Preprocessing
Convert text into sequences of integers and add padding to make all sequences the same length.

Tokenize (IMPLEMENTATION)
Convert sentences into sequences of word ids using Keras's Tokenizer function.

```python
def tokenize(x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    sequences = tokenizer.texts_to_sequences(x)
    return sequences, tokenizer

tests.test_tokenize(tokenize)
Padding (IMPLEMENTATION)
Pad sequences to ensure they are of equal length.
```
```python

def pad(x, length=None):
    if length is None:
        length = max(len(seq) for seq in x)
    padded_array = np.zeros((len(x), length), dtype=np.int32)
    for i, seq in enumerate(x):
        seq_length = len(seq)
        padded_array[i, :seq_length] = seq[:length]
    return padded_array

tests.test_pad(pad)
```
Preprocess Pipeline
Provided function to preprocess both English and French sentences.

python
Copy code
def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = preprocess(english_sentences, french_sentences)
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)
Models
Experiment with various neural network architectures, including RNN, embedding RNN, bidirectional RNN, encoder-decoder, and a custom model combining these features.

Model 1: Simple RNN (IMPLEMENTATION)
Build and train a basic RNN model.

python
Copy code
def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 1e-3
    model = Sequential()
    model.add(SimpleRNN(english_vocab_size, input_shape=input_shape[1:], return_sequences=True))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    return model

tests.test_simple_model(simple_model)
Model 2: Embedding RNN (IMPLEMENTATION)
Use word embeddings in a RNN model.

python
Copy code
def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 1e-3
    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size, output_dim=256, input_length=input_shape[1]))
    model.add(SimpleRNN(400, return_sequences=True))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(learning_rate), metrics=['accuracy'])
    return model

tests.test_embed_model(embed_model)
Model 3: Bidirectional RNN (IMPLEMENTATION)
Build a bidirectional RNN model.

python
Copy code
def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 0.001
    model = Sequential([
        Bidirectional(GRU(units=english_vocab_size, return_sequences=True), input_shape=input_shape[1:]),
        TimeDistributed(Dense(french_vocab_size)),
        Activation('softmax')
    ])
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(learning_rate), metrics=['accuracy'])
    return model

tests.test_bd_model(bd_model)
Model 4: Encoder-Decoder (IMPLEMENTATION)
Build an encoder-decoder model.

python
Copy code
def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 0.001
    encoder_input_seq = Input(shape=input_shape[1:])
    encoder_output, state_h, state_c = LSTM(units=english_vocab_size, return_sequences=False, return_state=True)(encoder_input_seq)
    decoder_input_seq = RepeatVector(output_sequence_length)(encoder_output)
    decoder_out = LSTM(units=english_vocab_size, return_sequences=True, return_state=False)(decoder_input_seq, initial_state=[state_h, state_c])
    logits = TimeDistributed(Dense(units=french_vocab_size))(decoder_out)
    model = Model(encoder_input_seq, Activation('softmax')(logits))
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    return model

tests.test_encdec_model(encdec_model)
Model 5: Custom Model (IMPLEMENTATION)
Create a model that incorporates embedding, bidirectional RNN, and encoder-decoder architecture.

python
Copy code
def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 1e-3
    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size, output_dim=400, input_length=input_shape[1:][0]))
    model.add(Bidirectional(GRU(units=400, return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    return model

tests.test_model_final(model_final)
Prediction
Train the final model and get predictions.

