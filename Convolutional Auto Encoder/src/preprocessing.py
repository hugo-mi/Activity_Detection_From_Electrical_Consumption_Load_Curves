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
                  ,strategy :Optional[str] = "off_peak_time" 
                  ,split_rate :Optional[float]=0.2
                  , split_method=None) -> np.array:
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
    print("\n---- Pre Processing Parameters ----")
    print("TIMEFRAMES = ", timeframes)
    print("SEQUENCE_LENGTH = ", sequence_length)
    print("RESAMPLE_PERIOD = ", resample_period)
    print("OVERLAP_PERIOD = ", overlap_period)
    print("STRATEGY = ", strategy)
        
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
    train_df, test_df, mask_test = train_test_split_dataset(df_resampled, method=split_method, split_rate=split_rate)
    
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
    if split_method=="random_days":
        list_df_test = []
        mask = ((mask_test) != (np.roll(mask_test, 1)))[mask_test]
        a = np.where(mask)[0]
        if 0 in a and len(a>1):
            a = a[1:]
        for df in np.split(test_df, a):
            list_df_test.append(df)
        # init 3D-array [samples, sequence_length, features]
        first_df_test = list_df_test[0]
        X_sequences_test, y_sequences_test = create_sequence(first_df_test, sequence_length, overlap_period)
        list_df_test.pop(0) # delete the first element of the list of train dataframes

        # Creation of sequences of length T and according to the overlapping period
        for df_test_ in list_df_test:
            next_X_sequences_test, next_y_sequences_test = create_sequence(df_test_, sequence_length, overlap_period)
            X_sequences_test = np.append(X_sequences_test, next_X_sequences_test, axis = 0)
            y_sequences_test = np.append(y_sequences_test, next_y_sequences_test, axis = 0)
    else:
        X_sequences_test, y_sequences_test = create_sequence(test_df, sequence_length, overlap_period)
    
    if strategy == "device":
        print("Strategy chosen : ", strategy)
        print("")
        print("#### Creating Train Sequence... ####")
        print("")
        print("")

        # load dataset with labels and resampled timeseries
        df_resampled_with_labels = load_dataset("house1_power_blk2_labels.zip", resample_period)
        # load dataset with inactive devices
        df_resampled_devices_inactive = load_aggregate_dataset("house1_power_blk2.zip", "inactive_house2", resample_period)
        activity = df_resampled_with_labels["activity"]
        df_resampled_device = df_resampled_devices_inactive.join(activity)
        df_resampled_device['mains'] = scaler_train.transform(df_resampled_device[['mains']])
        
        # --- TRAIN SEQUENCES ----
        X_sequence_train_device, y_sequence_train_device = create_sequence(df_resampled_device, sequence_length, overlap_period)
        
        return df_resampled_device, test_df, X_sequence_train_device, y_sequence_train_device, X_sequences_test, y_sequences_test
    
    
    if strategy == "label":
        print("Strategy chosen : ", strategy)
        print("")
        print("#### Creating Train Sequence... ####")  
        print("")
        print("")
        # load dataset with labels and resampled timeseries
        df_resampled_with_labels = load_dataset("house1_power_blk2_labels.zip", resample_period)
        df_resampled_with_labels = df_resampled_with_labels[df_resampled_with_labels.activity == 0]
        df_resampled_with_labels['mains'] = scaler_train.transform(df_resampled_with_labels[['mains']])
        
        # --- TRAIN SEQUENCES ----
        X_sequence_train_label, y_sequence_train_label = create_sequence(df_resampled_with_labels, sequence_length, overlap_period)
        
        return df_resampled_with_labels, test_df, X_sequence_train_label, y_sequence_train_label, X_sequences_test, y_sequences_test
    
    
    if strategy == "off_peak_time":
        print("Strategy chosen : ", strategy)
        print("")
        print("#### Creating Train Sequence... ####")
        print("")
        print("")
        # --- TRAIN SEQUENCES ----
        # Construction of the dataset according to peak and off-peak hours 
        if split_method=="random_days":
            list_df_train = []
            mask = ((~mask_test) != (np.roll(~mask_test, 1)))[~mask_test]
            for df in np.split(train_df, np.where(mask)[0]):
                list_df_train.extend(segmentDf(df, timeframes = timeframes))
        else:
            list_df_train = segmentDf(train_df, timeframes = timeframes)

        # init 3D-array [samples, sequence_length, features]
        first_df_train = list_df_train[0]
        list_X_sequence_train, list_y_sequence_train = create_sequence(first_df_train, sequence_length, overlap_period)
        list_df_train.pop(0) # delete the first element of the list of train dataframes

        # Creation of sequences of length T and according to the overlapping period
        for df_train_ in list_df_train:
            X_sequences_train, y_sequences_train = create_sequence(df_train_, sequence_length, overlap_period)
            list_X_sequence_train = np.append(list_X_sequence_train, X_sequences_train, axis = 0)
            list_y_sequence_train = np.append(list_y_sequence_train, y_sequences_train, axis = 0)
        
        return train_df, test_df, list_X_sequence_train, list_y_sequence_train, X_sequences_test, y_sequences_test
      