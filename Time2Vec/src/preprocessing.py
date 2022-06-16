# ---- utils libs ----
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ---- ML libs ----
from sklearn.preprocessing import StandardScaler

# --- Import functions from utils.py ---
import sys
sys.path.insert(0,'..')
from utils import load_dataset, load_aggregate_dataset, segmentDf, create_sequence, train_test_split_dataset

def data_preprocessing(timeframes: list
                  ,sequence_length: int
                  , overlap_period: int
                  ,resample_period :Optional[str]=None
                  ,use_labels :Optional[bool]=False
                  ,split_rate :Optional[float]=0.2) -> np.array:
    """
    1/ Loads the dataset and resample timeseries
    2/ Split a dataframe into train set and test set according to the split rate
    3/ Standardize Data
    4/ Construction of the dataset according to peak and off-peak hours 
    or according to activity labels
    5/ Creation of sequences of length T and according to the overlapping period
    
    Args:
        - resample_period: (optional) the reasmple period, if None the default period of 1 second will be used
        - timeframes: list of tuples indicating the periods of the day ex: timeframes = [(datetime.time(10,0,0), datetime.time(6,0,0)), (datetime.time(12,0,0), datetime.time(13,0,0))
        - use_labels: (False by default) use the activities labels
        - sequence_length: length of the sequence
        - overlap_period: overlap the sequences of timeseries
        - device_approach: the aggregated load curve of the devices which, when in operation, do not allow us to predict an activity 
        - split_rate: Rate of the test set size
        - device_strategy: use inactive devices base load curve
    Returns: 
        - list of prepocessed 3D-array [samples, sequence_length, features] (i.e sequences from the timeseries) 
    """
    
    # Diplay preprocessing parameters
    print("\n---- Post Processing Parameters ----")
    print("TIMEFRAMES = ", timeframes)
    print("SEQUENCE_LENGTH = ", sequence_length)
    print("RESAMPLE_PERIOD = ", resample_period)
    print("OVERLAP_PERIOD = ", overlap_period)
        
    # load dataset with labels and resampled timeseries
    print("")
    print("")
    print("#### Loading and Resampling Data... ####")
    print("")
    df_resampled = load_dataset("house1_power_blk2_labels.zip", resample_period)
    
    print("#### Creating Train and Test set... ####")
    print("")
    print("")
    # split dataframe into train set and test set
    train_df, test_df = train_test_split_dataset(df_resampled)
    
    # Standardize Data
    print("#### Rescaling Data... ####")
    print("")
    print("")
    scaler = StandardScaler()
    scaler_train = scaler.fit(train_df.loc[:, ['mains']])
    
    train_df.loc[:, 'mains'] = scaler_train.transform(train_df.loc[:, ['mains']])
    test_df.loc[:, 'mains'] = scaler_train.transform(test_df.loc[:, ['mains']])
        
    # ---- TEST SEQUENCES ----
    print("#### Creating Test Sequence... ####")
    print("")
    print("")

    X_sequences_test, y_sequences_test = [], []
    for sequence in create_sequence(test_df, sequence_length, overlap_period, ['mains']):
        X_sequences_test.append(sequence)
        
        # gen_labels
    for sequence in create_sequence(test_df, sequence_length, overlap_period, ['activity']):
        y_sequences_test.append(sequence)
        
    X_sequences_test = np.asarray(X_sequences_test)
    y_sequences_test = np.asarray(y_sequences_test)

    ### TRAIN TEST SPLIT HOUSE 1 ###
    print("#### Creating Train Sequence... ####")
    print("")
    print("")
    # --- TRAIN SEQUENCES ----
    X_sequences_train, y_sequences_train = [], []
    for sequence in create_sequence(train_df, sequence_length, overlap_period, ['mains']):
        X_sequences_train.append(sequence)
        
        # gen_labels
    for sequence in create_sequence(train_df, sequence_length, overlap_period, ['activity']):
        y_sequences_train.append(sequence)
        
    X_sequences_train = np.asarray(X_sequences_train)
    y_sequences_train = np.asarray(y_sequences_train)
    
    return train_df, test_df, X_sequences_train, y_sequences_train, X_sequences_test, y_sequences_test