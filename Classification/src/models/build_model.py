from contextlib import redirect_stdout
import os
from pathlib import Path


## Build a model
"""
Build the model with the arguments passed. If no arguments are passed the default, best parameters will be used
"""

# ---- Deep Learning libs ----
from lightgbm import LGBMClassifier


def model_classifier(params=None):
    if params is None:
        params = {'reg_alpha': 5.7697156354232726e-08, 'reg_lambda': 0.09393757061413283, 'num_leaves': 21, 'colsample_bytree': 0.8562051606254208, 'subsample': 0.6693493351224262, 'subsample_freq': 7, 'min_child_samples': 33}

    model = LGBMClassifier(**params)
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'model_summary.txt'
    
    with open(path, 'w') as f:
        with redirect_stdout(f):
            model.get_params()
    
    print(model.get_params())
    
    return model