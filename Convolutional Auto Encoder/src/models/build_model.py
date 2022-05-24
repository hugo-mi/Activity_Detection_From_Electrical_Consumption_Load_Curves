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
from tensorflow import keras
from tensorflow.keras import layers

def model(X_train):

    model = keras.Sequential(
        [
            layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=4, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'model_summary.txt'
    
    with open(path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    print(model.summary())
    
    return model