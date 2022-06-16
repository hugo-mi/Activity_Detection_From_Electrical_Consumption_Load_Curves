## Detecting anomalies

"""
We will detect anomalies by determining how well our model can reconstruct the input data.

**1/** Find ``MAE`` loss on training samples.

**2/** Find max MAE loss value. This is the worst our model has performed trying to reconstruct a sample. We will make this the ``threshold`` for anomaly detection.

**3/** If the reconstruction loss for a sample is greater than this ``threshold`` value then we can infer that the model is seeing a pattern that it isn't familiar with. We will label this sample as an **anomaly.**
"""

# ---- utils libs ----
import numpy as np
import pandas as pd
import os
from pathlib import Path


# ---- Data Viz libs ---- 
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

def X_train_predict(model, X_train):
    X_train_pred = model.predict(X_train)
    print("\n---- X_train_pred sequence shape ----")
    print(X_train_pred.shape)
    
    return X_train_pred

def X_test_predict(model, X_test):
    X_test_pred = model.predict(X_test)
    print("\n---- X_test_pred sequence shape ----")
    print(X_test_pred.shape)
    
    return X_test_pred

def y_test_predict(model, X_test):
    y_test_pred = model.predict(X_test)
    print("\n---- y_test_pred sequence shape ----")
    print(y_test_pred.shape)
    
    return y_test_pred

def plot_train_mse_loss(X_train_pred, X_train):
    
    train_mse_loss = np.mean(np.sqrt((X_train_pred - X_train)**2), axis=1)
    
    plt.figure(figsize = (10, 7))
    sns.distplot(train_mse_loss, bins=50, kde = True)
    plt.xlabel("Train MSE loss")
    plt.ylabel("No of samples")
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'figures' / 'train_MSE_loss.png'

    plt.savefig(path) 
    
def plot_test_mse_loss(X_test_pred, X_test):
    
    test_mse_loss = np.mean(np.sqrt((X_test_pred - X_test)**2), axis=1)
    test_mse_loss = test_mse_loss.reshape((-1))
    
    plt.figure(figsize = (10, 7))
    sns.distplot(test_mse_loss, bins=50, kde = True)
    plt.xlabel("Test MSE loss")
    plt.ylabel("No of samples")
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'figures' / 'test_MSE_loss.png'

    plt.savefig(path) 

def detect_activity_sequence(y_test, seq_length, overlap_period):
    
    # Make our predictions
    step = seq_length - overlap_period
    y_unfolded = np.zeros((y_test.shape[0]*step, ))
    y_weight = np.zeros((y_test.shape[0]*step, ))

    for i in range(seq_length):
        y_unfolded[i::step] += y_test[:-int(i/step) if int(i/step) > 0 else None,i,0]
        y_weight[i::step] += 1

    return y_unfolded/y_weight
    
def get_df_predict(sequences_activity, test_df):
    df_predict = test_df.copy()

    df_predict.iloc[:len(sequences_activity),:].loc[:, 'activity'] = (sequences_activity > 0.5).astype(int)
    df_predict.iloc[len(sequences_activity):, :].loc[:, 'activity'] = np.nan
    df_predict = df_predict.dropna(axis=0)

    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'prediction_dataframe.txt'

    with open(path, mode='w') as f:
        df_predict.to_csv(f)

    return df_predict