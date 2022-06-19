import os
from pathlib import Path
import numpy as np
import pandas as pd

### Train the model
"""
Fit our model
"""

def train_classifier(model, X_train, y_train):

    return  model.fit(X_train, y_train)