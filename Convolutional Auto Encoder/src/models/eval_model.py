### Evaluation of the model

"""
Let's plot training and validation loss to see how the training went.
"""
# ---- utils libs ----
import os
from pathlib import Path
from typing import Optional

# ---- Data Viz libs ---- 
from matplotlib import pyplot as plt
import seaborn as sns

# --- Import functions from utils.py ---
import sys
sys.path.insert(0,'..')
from utils import plot_confusion_matrix, plot_activity_hist


def plot_train_val_loss(history):

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Training & Validation Loss Evolution\n")
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'figures' / 'train_validation_loss.png'

    plt.savefig(path)
    
def plot_reconstructed_base_load_curve(X_train, X_train_pred, sequence_num: Optional[int]= 1):
    
    # Checking how the first sequence is learnt
    plt.figure(figsize = (10, 5))
    plt.plot(X_train[sequence_num], label="real load curve")
    plt.plot(X_train_pred[sequence_num], label="reconstructed load curve")
    plt.title("Reconstruction load curve comparison(" + str(sequence_num) + ")\n", fontsize=15)
    plt.legend()
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'figures' / 'reconstructed_base_load_curve.png'

    plt.savefig(path)
    
def plot_activity_histogram(df_anomalies, test_df):
    
    fig, ax = plt.subplots()
    plot_activity_hist(df_anomalies['activity_pred'], figsize=(12, 6), alpha=0.5, label='predictions', ax=ax)
    plot_activity_hist(test_df["activity"], figsize=(12, 6), alpha=0.5, label='truth', color='tab:orange', ax=ax)
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'figures' / 'activity_predicted_histogram.png'

    plt.savefig(path) 

    
def plot_activity_distibrution(df_predict):
    
    sns.histplot(data=df_predict, x="activity").set(title='Activity prediction distribution (Activity VS Non Activity)')
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'figures' / 'activity_predicted_distribution.png'

    plt.savefig(path) 
    
def confusion_matrix(test_df, df_predict):
    
    plot_confusion_matrix(test_df["activity"], df_predict['activity'])
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'figures' / 'confusion_matrix.png'

    plt.savefig(path) 
