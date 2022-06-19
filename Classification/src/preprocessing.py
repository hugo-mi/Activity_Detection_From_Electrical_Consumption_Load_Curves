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
from utils import load_dataset, generate_features, train_test_split_dataset

def data_preprocessing(resample_period :Optional[str]=None
                      ,use_labels :Optional[bool]=False
                      ,split_rate :Optional[float]=0.2) -> np.array:
    """
    1/ Loads the dataset and resample timeseries
    2/ Split a dataframe into train set and test set according to the split rate
    3/ Generate extra features
    3/ Standardize Data
    5/ Creation of the train/test sequences
    
    Args:
        - resample_period: (optional) the reasmple period, if None the default period of 1 second will be used
        - use_labels: (False by default) use the activities labels
        - split_rate: Rate of the test set size
    Returns: 
        - list of prepocessed arrays [samples, features] 
    """
    
    # Diplay preprocessing parameters
    print("\n---- Post Processing Parameters ----")
    print("RESAMPLE_PERIOD = ", resample_period)
        
    # load dataset with labels and resampled timeseries
    print("")
    print("")
    print("#### Loading and Resampling Data... ####")
    print("")
    df_resampled = load_dataset("house1_power_blk2_labels.zip", resample_period)

    # generate our features
    print("#### Generating extra features... ####")
    print("")
    print("")
    windows = ['1h', '10min']
    features_col = ['mains', 'hour']

    df_resampled, cols = generate_features(df_resampled, window=windows)
    features_col += cols
    
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
    X_test, y_test = test_df[features_col].values, test_df["activity"].values

    # --- TRAIN SEQUENCES ----
    print("#### Creating Train Sequence... ####")
    print("")
    print("")
    X_train, y_train = train_df[features_col].values, train_df["activity"].values
    
    return train_df, test_df, X_train, y_train, X_test, y_test