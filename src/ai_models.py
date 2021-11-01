from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Lambda
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate

def simple_CNN(n_timesteps, n_features):
    ''' A simple CNN model, should probably add some dropout layers. '''
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(1))
    return model, 1

def complex_CNN(n_timesteps, n_features):
    ''' A three-way CNN with different kernel sizes for different levels of generalisation. '''
    # head 1
    inputs1 = Input(shape=(n_timesteps,n_features))
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
    conv12 = Conv1D(filters=64, kernel_size=3, activation='relu')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv12)
    conv13 = Conv1D(filters=64, kernel_size=3, activation='relu')(pool1)
    conv14 = Conv1D(filters=64, kernel_size=3, activation='relu')(conv13)
    pool1 = MaxPooling1D(pool_size=2)(conv14)
    flat1 = Flatten()(pool1)
    
    # head 2
    inputs2 = Input(shape=(n_timesteps,n_features))
    conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
    conv22 = Conv1D(filters=64, kernel_size=5, activation='relu')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv22)
    conv23 = Conv1D(filters=64, kernel_size=5, activation='relu')(pool2)
    conv24 = Conv1D(filters=64, kernel_size=5, activation='relu')(conv23)
    pool2 = MaxPooling1D(pool_size=2)(conv24)
    flat2 = Flatten()(pool2)
    
    # head 3
    inputs3 = Input(shape=(n_timesteps,n_features))
    conv3 = Conv1D(filters=64, kernel_size=11, activation='relu')(inputs3)
    conv32 = Conv1D(filters=64, kernel_size=11, activation='relu')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv32)
    conv33 = Conv1D(filters=64, kernel_size=11, activation='relu')(pool3)
    conv34 = Conv1D(filters=64, kernel_size=11, activation='relu')(conv33)
    pool3 = MaxPooling1D(pool_size=2)(conv34)
    flat3 = Flatten()(pool3)
    
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = Dense(100)(merged)
    dense2 = Dense(1)(dense1)
    outputs = LeakyReLU(alpha=0.25)(dense2)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    return model, 3

def simple_CNN_double_output(n_timesteps, n_features):
    ''' Last layer has two nodes, for when both DYS and CA have to be predicted. '''
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(2))    
    return model, 1

def simple_LSTM(n_timesteps, n_features):
    ''' A simple LSTM model, no good results yet. '''
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    return model, 1

def complex_LSTM(n_timesteps, n_features):
    ''' Multiple LSTM layers, better results. '''
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features), return_sequences=True))
    model.add(LSTM(100, return_sequences=True))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100, return_sequences=True))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(1))
    return model, 1