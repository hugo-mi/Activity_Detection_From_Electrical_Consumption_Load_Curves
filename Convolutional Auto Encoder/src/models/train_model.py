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


def train(model, X_train):

    history = model.fit(
        X_train,
        X_train,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )
    
    history_df = pd.DataFrame(history.history)
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'training_history.txt'

    with open(path, mode='w') as f:
        history_df.to_csv(f)
        
    return history
    
    