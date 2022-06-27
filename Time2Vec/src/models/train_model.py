import os
from pathlib import Path
import numpy as np
import pandas as pd

### Train the model
"""
Please note that we are using ``X_train`` as both the input 
and the target since this is a reconstruction model.
"""

# ---- Deep Learning libs ----
from tensorflow import keras
from keras.callbacks import EarlyStopping

def train_embeddings(model, X_train, params=None):
    if params is None:
        params = {'lr': 0.001, 'epochs': 20, 'batch_size': 128}

    early_stop = EarlyStopping(patience=3, verbose=0, min_delta=0.0001, monitor='val_loss', mode='auto', restore_best_weights=True)

    history =  model.fit(X_train,
                         X_train,
                         epochs=params['epochs'],
                         batch_size=params['batch_size'],
                         validation_split=0.1,
                         callbacks=[early_stop])
    
    embeddings = model.layers[1].get_weights()
    
    history_df = pd.DataFrame(history.history)
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'training_emb_history.txt'

    with open(path, mode='w') as f:
        history_df.to_csv(f)
        
    return history, embeddings

def train_classifier(model, embeddings, X_train, y_train, params=None):
    if params is None:
        params = {'lr': 0.001, 'epochs': 15, 'batch_size': 128}
    
    early_stop = EarlyStopping(patience=3, verbose=0, min_delta=0.0001, monitor='val_loss', mode='auto', restore_best_weights=True)
    
    model.layers[1].set_weights(embeddings)
    model.layers[1].trainable = False

    history =  model.fit(X_train,
                         y_train,
                         epochs=params['epochs'],
                         batch_size=params['batch_size'],
                         validation_split=0.1,
                         callbacks=[early_stop])
    
    history_df = pd.DataFrame(history.history)
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'training_c_history.txt'

    with open(path, mode='w') as f:
        history_df.to_csv(f)
        
    return history