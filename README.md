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
Setup
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

