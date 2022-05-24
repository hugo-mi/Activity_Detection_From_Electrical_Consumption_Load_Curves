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
    
    

def compute_threshold(X_train_pred, X_train):
    
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
    # Get reconstruction loss threshold.
    threshold = np.max(train_mae_loss)
    print("Reconstruction error treshold: ", threshold)
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'threshold.txt'
    
    with open(path, 'w') as f:
        f.write('###################################')
        f.write('\n')
        f.write("THRESHOLD")
        f.write('\n')
        f.write("###################################")
        f.write('\n\n\n\n')
        f.write("Reconstruction error treshold: " + str(threshold))

    return threshold



def compute_train_mae_loss(X_train_pred, X_train):
    
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
    return train_mae_loss

def compute_test_mae_loss(X_test_pred, X_test):
    
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
    return test_mae_loss



def plot_train_mae_loss(X_train_pred, X_train):
    
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
    
    plt.figure(figsize = (10, 7))
    sns.distplot(train_mae_loss, bins=50, kde = True)
    plt.xlabel("Train MAE loss")
    plt.ylabel("No of samples")
    
    # Get reconstruction loss threshold.
    threshold = np.max(train_mae_loss)
    print("Reconstruction error threshold: ", threshold)
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'figures' / 'train_MAE_loss.png'

    plt.savefig(path) 
    
def plot_test_mae_loss(X_test_pred, X_test):
    
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
    test_mae_loss = test_mae_loss.reshape((-1))
    
    plt.figure(figsize = (10, 7))
    sns.distplot(test_mae_loss, bins=50, kde = True)
    plt.xlabel("Test MAE loss")
    plt.ylabel("No of samples")
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'figures' / 'test_MAE_loss.png'

    plt.savefig(path) 
    
def detect_anomaly_sequence(test_mae_loss, threshold, SEQUENCE_LENGTH, y_test):
    
    # Detect all the samples which are anomalies.
    anomalies = test_mae_loss > threshold
    
    anomalies_counter = np.sum(anomalies)
    anomalies_idx = np.where(anomalies)
    
    print("Number of anomaly samples: ")
    print(anomalies_counter)
    
    print("\n\nIndices of anomaly samples: ")
    print(anomalies_idx)
    
    # get index of each sequence considered as an anomaly
    sequences_anomalies_idx = list()
    for i in range(len(anomalies)):
        if anomalies[i] == True:
            sequences_anomalies_idx.append(i)
            
    # get index of each data point from X_test considered as an anomaly 
    data_anomalies_idx = list()
    for elm in sequences_anomalies_idx:
        for i in range(SEQUENCE_LENGTH):
            data_idx = y_test[elm][i][2] 
            data_anomalies_idx.append(data_idx)
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'anomaly_sequences.txt'
    
    with open(path, 'w') as f:
        f.write('###################################')
        f.write('\n')
        f.write("ANOMALY SEQUENCES")
        f.write('\n')
        f.write("###################################")
        f.write('\n\n\n\n')
        
        f.write("Number of anomaly samples: " + str(anomalies_counter))
        
        f.write('\n\n\n')
        f.write("Indices of anomaly samples:\n\n")
        f.write(str(sequences_anomalies_idx))
        
        f.write('\n\n\n\n')

        f.write('###################################')
        f.write('\n')
        f.write("ANOMALY DATA POINTS")
        f.write('\n')
        f.write("###################################")
        f.write('\n\n\n\n')
        
        f.write("Number of anomaly samples: " + str(len(data_anomalies_idx)))
        
        f.write('\n\n\n')
        f.write("Indices of anomaly samples:\n\n")
        f.write(str(data_anomalies_idx))
    
    return sequences_anomalies_idx, data_anomalies_idx

def get_anomaly_df(data_anomalies_idx, test_df):
    df_test_mains = pd.DataFrame(test_df["mains"])
    df_anomalies = df_test_mains.iloc[data_anomalies_idx]
    print(df_anomalies)
    df_anomalies["activity_pred"] = 1
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'anomalies_dataframe.txt'

    with open(path, mode='w') as f:
        df_anomalies.to_csv(f)
    
    return df_anomalies

def get_df_predict(test_df, df_anomalies):
    df_predict = test_df.copy()
    df_predict["activity"] = 0
    idx_anom = df_anomalies.index
    df_predict.loc[idx_anom, "activity"] = 1
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'prediction_dataframe.txt'

    with open(path, mode='w') as f:
        df_predict.to_csv(f)

    return df_predict


def get_specific_anomaly_sequence(sequence_num, sequences_anomalies_idx, y_test):
    
    #os.getcwd()
    #path = Path(os.getcwd())
    #path = path.parent.absolute() / 'reports' / 'figures' / 'anomaly_sequences.txt'

    #y_test_anomaly_seq = y_test[sequences_anomalies_idx[0][sequence_num]]
    #np.savetxt(path, y_test_anomaly_seq, delimiter=',')
    return

def plot_anomaly_sequences(test_df, data_anomalies_idx):
    df_test_mains = test_df["mains"]
    df_anomalies = df_test_mains.iloc[data_anomalies_idx]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_test_mains.index, y=df_test_mains.values, name='Normal'))
    fig.add_trace(go.Scatter(x=df_anomalies.index, y=df_anomalies.values, mode='markers', name='Anomaly = Activity (Predicted)'))
    fig.update_layout(showlegend=True, title='Detected anomalies')
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'figures' / 'anomaly_sequences.html'
    fig.write_html(path)

def plot_anomaly_data_points(test_df, data_anomalies_idx, threshold):
    test_df_value = test_df["mains"]
    df_subset_anomalies = test_df_value.iloc[data_anomalies_idx]
    df_subset_anomalies = df_subset_anomalies[df_subset_anomalies.values > threshold]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df_value.index, y=test_df_value.values, name='Normal'))
    fig.add_trace(go.Scatter(x=df_subset_anomalies.index, y=df_subset_anomalies.values, mode='markers', name='Anomaly = Activity (Predicted)'))
    fig.update_layout(showlegend=True, title='Detected anomaly data points > threshold')
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'figures' / 'anomaly_data_points.html'
    fig.write_html(path)
