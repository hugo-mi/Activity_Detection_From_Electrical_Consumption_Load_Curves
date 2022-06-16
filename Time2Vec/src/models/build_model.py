from contextlib import redirect_stdout
import os
from pathlib import Path


## Build a model
"""
We will build a convolutional reconstruction autoencoder model. 
The model will take input of shape (``batch_size``, ``sequence_length``, ``num_features``) 
and return output of the same shape. In this case, ``sequence_length`` is 10 and ``num_features`` is 1.
"""

# ---- Deep Learning libs ----
from keras.layers import Dropout, Layer, LSTM, Dense, Input
from keras.models import Model
from keras import backend as K

### DEFINE T2V LAYER ###

class T2V(Layer):
    
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)
        
    def build(self, input_shape):

        self.W = self.add_weight(name='W',
                                shape=(input_shape[-1], self.output_dim),
                                initializer='uniform',
                                trainable=True)

        self.P = self.add_weight(name='P',
                                shape=(input_shape[1], self.output_dim),
                                initializer='uniform',
                                trainable=True)

        self.w = self.add_weight(name='w',
                                shape=(input_shape[1], 1),
                                initializer='uniform',
                                trainable=True)

        self.p = self.add_weight(name='p',
                                shape=(input_shape[1], 1),
                                initializer='uniform',
                                trainable=True)

        super(T2V, self).build(input_shape)
        
    def call(self, x):
        
        original = self.w * x + self.p
        sin_trans = K.sin(K.dot(x, self.W) + self.P)
        
        return K.concatenate([sin_trans, original], -1)

def T2V_NN(param, dim):
    
    inp = Input(shape=(dim,1))
    x = T2V(param['t2v_dim'])(inp)
    x = Dropout(0.2)(x)
    # return_sequences=False : ne retourne que le dernier état
    x = LSTM(param['unit'], activation=param['act'], return_sequences=True)(x)
    x = Dense(1)(x)
    
    m = Model(inp, x)
    m.compile(loss='mse', optimizer='adam')
    
    return m

### DEFINE CLASSIFICATION MODEL STRUCTURES ###
def T2V_NN_C(param, dim):
    
    # set_seed(33)
    
    inp = Input(shape=(dim,1))
    x = T2V(param['t2v_dim'])(inp)
    x = Dropout(0.2)(x)
    x = LSTM(param['unit'], activation=param['act'], return_sequences=True)(x)
    # we want to make a classification with 2 classes
    x = Dense(1, activation='sigmoid')(x)
    
    m = Model(inp, x)
    m.compile(loss='bce', optimizer='adam')
    
    return m


def model_embeddings(X_train, params=None):
    if params is None:
        params = {'unit': 32, 't2v_dim': 128, 'act': 'relu'}

    model = T2V_NN(param=params, dim=X_train.shape[1])
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'model_emb_summary.txt'
    
    with open(path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    print(model.summary())
    
    return model

def model_classifier(X_train, params=None):
    if params is None:
        params = {'unit': 32, 't2v_dim': 128, 'act': 'relu'}

    model = T2V_NN_C(param=params, dim=X_train.shape[1])
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'model_c_summary.txt'
    
    with open(path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    print(model.summary())
    
    return model