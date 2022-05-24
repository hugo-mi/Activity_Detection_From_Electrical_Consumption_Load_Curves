# ---- utils libs ----
import pandas as pd
from typing import Optional
import os
from pathlib import Path

# ---- Data Viz libs ---- 
import plotly.graph_objects as go

# --- Import functions from utils.py ---
import sys
sys.path.insert(0,'..')
from utils import load_dataset


## Visualize Load Curve Dataset

def visualize_load_curve_dataset(filename: str, resample_period :Optional[str]="60min"):
    dataset_resampled = load_dataset(filename, resample_period)
    dataset_resampled = dataset_resampled["mains"]
    df_mains_resampled = pd.DataFrame(dataset_resampled)
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'load_curve_dataset.txt'

    
    df_mains_resampled.to_csv(path, sep=' ')
    
    print("Load Curve Dataset Resampled each", resample_period)
    print(df_mains_resampled.head(30))
 

## Visualize Load Curve Resampled

def visualize_load_curve_resampled(filename: str, resample_period :Optional[str]="60min"):

    dataset_resampled = load_dataset(filename, resample_period)
    dataset_resampled = dataset_resampled["mains"]
    df_mains_resampled = pd.DataFrame(dataset_resampled)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_mains_resampled.index, y=df_mains_resampled['mains'], name='Load Curve'))
    title = filename.split(".")[0]
    fig.update_layout(showlegend=True, title=title)
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'figures' / 'load_curve.html'
    fig.write_html(path)

## Visualize Train Load Curve 

def visualize_train_load_curve(resample_train_df, strategy :Optional[str]="off_peak_time"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resample_train_df.index, y=resample_train_df['mains'], name='Load Curve'))
    fig.update_layout(showlegend=True, title='resampled'+ strategy +'train load curve')
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'figures' / 'train_load_curve.html'
    fig.write_html(path)
    
## Visualize Test Load Curve 

def visualize_test_load_curve(resample_test_df, strategy :Optional[str]="off_peak_time"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resample_test_df.index, y=resample_test_df['mains'], name='Load Curve'))
    fig.update_layout(showlegend=True, title='resampled'+ strategy +'test load curve')
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'figures' / 'test_load_curve.html'
    fig.write_html(path)
    
## Visualize Train Test Split Load Curve 
    
def visualize_test_train_load_curve(resample_train_df, resample_test_df, strategy :Optional[str]="off_peak_time"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resample_train_df.index, y=resample_train_df['mains'], name='Train'))
    fig.add_trace(go.Scatter(x=resample_test_df.index, y=resample_test_df['mains'], name='Test'))
    fig.update_layout(showlegend=True, title='train test split load curve',xaxis_title="Date",
        yaxis_title="mains (power, normalized)",
        legend_title="Dataframe")
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'figures' / 'train_test_split_load_curve.html'
    fig.write_html(path)

    
    print("\n power mean train load curve", resample_train_df["mains"].mean())
    print("\n power mean test load curve", resample_test_df["mains"].mean())    
    


def visualize_report_preprocessing(X_train, y_train, X_test, y_test
                                   ,TIMEFRAMES
                                   ,SEQUENCE_LENGTH
                                   ,OVERLAP_PERIOD
                                   ,TIME_STEP
                                   ,STRATEGY):
    
    print("###################################")
    print("PREPROCESSING REPORT")
    print("###################################")
    
    print("---- Post Processing Parameters ----")
    print("TIMEFRAMES = ", TIMEFRAMES)
    print("SEQUENCE_LENGTH = ", SEQUENCE_LENGTH)
    print("OVERLAP_PERIOD = ", OVERLAP_PERIOD)
    print("OVERLAP_PERIOD = ", TIME_STEP)
    print("STRATEGY = ", STRATEGY)
    
    print("\n---- X_train sequence shape ----")
    print(X_train.shape)
    
    print("\n---- y_train sequence shape ----")
    print(y_train.shape)
    
    print("\n\n---- X_test sequence shape ----")
    print(X_test.shape)
    
    print("\n---- y_test sequence shape ----")
    print(y_test.shape)
    
    print("\n\n---- X_train sequence ----")
    print(X_train)
    
    print("\n---- y_train sequence ----")
    print(y_train)
    
    print("\n\n---- X_test sequence ----")
    print(X_test)
    
    print("\n---- y_test sequence ----")
    print(y_test)
    
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'preprocessing_report.txt'
    
    with open(path, 'w') as f:
        f.write('###################################')
        f.write('\n')
        f.write("PREPROCESSING REPORT")
        f.write('\n')
        f.write("###################################")
        f.write('\n\n\n\n')
        
        f.write("---- Post Processing Parameters ----")
        f.write('\n\n')
        f.write("TIMEFRAMES = " + str(TIMEFRAMES))
        f.write('\n')
        f.write("SEQUENCE_LENGTH = " + str(SEQUENCE_LENGTH))
        f.write('\n')
        f.write("OVERLAP_PERIOD = " + str(OVERLAP_PERIOD))
        f.write('\n')
        f.write("OVERLAP_PERIOD = " + str(TIME_STEP))
        f.write('\n')
        f.write("STRATEGY = " + str(STRATEGY))
        f.write('\n\n\n')
        
        f.write("\n---- X_train sequence shape [samples, sequence_length, features] ----\n")
        f.write(str(X_train.shape))
        f.write('\n')
        
        f.write("\n---- y_train sequence shape [samples, sequence_length, [Timestamp, activity_label, index]]----\n")
        f.write(str(y_train.shape))
        f.write('\n')
        
        f.write("\n\n---- X_test sequence shape [samples, sequence_length, features] ----\n")
        f.write(str(X_test.shape))
        f.write('\n')
        
        f.write("\n---- y_test sequence shape [samples, sequence_length, [Timestamp, activity_label, index]] ----\n")
        f.write(str(y_test.shape))
        f.write('\n\n')
        
        f.write("\n\n---- X_train sequence [samples, sequence_length, features] ----\n")
        f.write(str(X_train))
        f.write('\n\n')
        
        f.write("\n---- y_train sequence [samples, sequence_length, [Timestamp, activity_label, index]] ----\n")
        f.write(str(y_train))
        f.write('\n\n')
        
        f.write("\n\n---- X_test sequence [samples, sequence_length, features] ----\n")
        f.write(str(X_test))
        f.write('\n\n')
        
        f.write("\n---- y_test sequence [samples, sequence_length, [Timestamp, activity_label, index]] ----\n")
        f.write(str(y_test))