### Saving / Loading of the model

"""
Allows us to save / load our models in pickle format
"""
# ---- utils libs ----
import os
from pathlib import Path
import pickle

def save_model(model, file_name='model.pkl'):
        
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'src' / 'data' / file_name

    pickle.dump(model, open(path, 'wb'))
    print(f"Model saved in {path}")

def load_model(file_name='model.pkl'):
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'src' / 'data' / file_name

    model = pickle.load(open(path, 'rb'))

    return model


