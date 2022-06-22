### Evaluation of the model

"""
Let's plot training and validation loss to see how the training went.
"""
# ---- utils libs ----
import numpy as np
import os
from pathlib import Path
from typing import Optional

# ---- Data Viz libs ---- 
from matplotlib import pyplot as plt
import seaborn as sns

# --- Import functions from utils.py ---
import sys
sys.path.insert(0,'..')
from utils import plot_confusion_matrix, plot_activity_hist, detect_stages, get_TPTNFPFN, get_IoU, get_activity_stages, broken_barh_x


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
    
def evaluate(pred, df_gt, display_plots=True):
    """
    Evaluer les prédictions en terme de mAP (mean avergae precision) et mAR (mean average recall)
    Args :
        - pred : dataframe de 2 colonnes : (timestamp, activity_prediction)
        - gt : dataframe de 2 colonnes : (timestamp, true_activity)
        - plot_recap : wether or not to display the summary of predicted and true activity
        - plot_metrics : wether or not to display the metrics plots
    Returns :
        - list of (IoU threshold, mAP, mAR)
        (- side effect : plots)
    """

    # resample_period = pred.iloc[1,0] - pred.iloc[0,0] # non utilisé
    
    colActivity_df_gt = df_gt.columns[1]
    colActivity_pred = pred.columns[1]

    df_gt_period = detect_stages(df_gt, colActivity_df_gt, df_gt.columns[0])
    pred_period = detect_stages(pred, colActivity_pred, pred.columns[0])
    
    df_merged = pred.set_index(pred.columns[0]).join(df_gt.set_index(df_gt.columns[0]), how='outer').fillna(method="ffill")
    df_merged = get_TPTNFPFN(df_merged, col_pred=colActivity_pred, col_gt=colActivity_df_gt)#.loc[:, ["TP"	,"TN",	"FP",	"FN"]]
    df_merged = df_merged.reset_index().rename(columns={"index":'datetime'})

    # restriction aux périodes ground_truth d'activité
    df_gt_period_activity = get_activity_stages(df_gt_period, colActivity_df_gt)
    # restriction aux périodes prédites d'activité
    pred_period_activity = get_activity_stages(pred_period, colActivity_pred)
    
    # # ajout de la colonne de metrique IoU à la dataframe des predictions
    l = []
    for ts_min, ts_max in zip(pred_period_activity.iloc[:,1], pred_period_activity.iloc[:,2]): # col 1 for timestamp min, col 2 for timestamp max
        l.append((get_IoU(df_gt_period, df_gt_period.columns[1], df_gt_period.columns[2]
                            , ts_min, ts_max, colActivity_df_gt,  activity=1))[3])
    pred_period_activity["IoU"] = np.array(l)

     # ajout de la colonne de metrique IoU à la dataframe ground_truth
    l = []
    for ts_min, ts_max in zip(df_gt_period_activity.iloc[:,1], df_gt_period_activity.iloc[:,2]):
        l.append((get_IoU(pred_period,pred_period.columns[1], pred_period.columns[2]
                        , ts_min, ts_max, colActivity_pred, 1)[3]))
    df_gt_period_activity["IoU"] = np.array(l)
    df_gt_period_activity.head()

    
    # === Calcul des métriques ===

    tau_range = np.linspace(0,1,101)
    
    # calcul du mAP
    N = len(pred_period_activity)
    map = []
    for tau in tau_range:
        map.append(len(pred_period_activity[pred_period_activity["IoU"]>tau])/N)

    # calcul du mAR
    N = len(df_gt_period_activity)
    mar = []
    for tau in tau_range:
        mar.append(len(df_gt_period_activity[df_gt_period_activity["IoU"]>tau])/N)


     #=================== = Plots = ====================

    if display_plots:
        fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [4, 1]})
        fig.set_size_inches(20, 6)
    
        # Plot summary 0
        col_timestamp_min_pred_period = pred_period.columns[1]
        col_timestamp_max_pred_period = pred_period.columns[2]
        times1_pred, times0_pred = broken_barh_x(pred_period, colActivity_pred, col_timestamp_min_pred_period, col_timestamp_max_pred_period)
        ax[0,0].broken_barh(times1_pred, (0,1), label = "Activity")
        ax[0,0].broken_barh(times0_pred, (0,1), facecolors='lightgray')

        col_timestamp_min_dfgt_period = df_gt_period.columns[1]
        col_timestamp_max_dfgt_period = df_gt_period.columns[2]
        times1_gt, times0_gt = broken_barh_x(df_gt_period, colActivity_df_gt, col_timestamp_min_dfgt_period, col_timestamp_max_dfgt_period)
        ax[0,0].broken_barh(times1_gt, (1.05,1))
        ax[0,0].broken_barh(times0_gt, (1.05,1), facecolors='lightgray')
        
        try:
            ax[0,0].set_yticks([0.5, 1.5], labels=['pred', 'gt'])
        except(TypeError):
            ax[0,0].set_yticks([0,5, 1,5])
            ax[0,0].set_yticklabels(['pred', 'gt'])
            ax[0,0].set_ylim(bottom = 0, top = 2)
            
        #ax[0,0].set_yticks([0.5, 1.5], labels=['pred', 'gt'])
        ax[0,0].legend()
        ax[0,0].set_title("Ground Truth(top) and pred(bottom)")

        # Plot summary 1
        for case, color in zip(["TP", "TN", "FN","FP"], ["green", "lightgreen", "#F9F691","red"]):
            df_tp = detect_stages(df_merged, case, df_merged.columns[0])
            times, _ = broken_barh_x(df_tp, case, df_tp.columns[1], df_tp.columns[2])
            ax[1,0].broken_barh(times, (0,1), label = case, facecolors=color)

        ax[1,0].legend()
        ax[1,0].set_title("Pred vs ground_truth - brut instantané")
        plt.tight_layout()

        # Plot curves
        ax[0,1].plot(map, label = "mAP : rate of correct activity period")
        ax[0,1].plot(mar, c='orange', label="mAR : rate of detected activity period")
        ax[0,1].set_title("mAP and mAR curves for IoU metric")
        ax[0,1].set_ylabel("Rate")
        ax[0,1].set_xlabel("IoU threshold tau (%)")
        ax[0,1].legend()
        
        os.getcwd()
        path = Path(os.getcwd())
        path = path.parent.absolute() / 'reports' / 'figures' / 'evaluation_direct_and_IoU.png'
        plt.savefig(path) 

        
    return (tau_range, map, mar)