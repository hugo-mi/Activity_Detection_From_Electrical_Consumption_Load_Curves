from typing import List, Tuple, Optional, Any
import os
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.metrics import fbeta_score, accuracy_score
import datetime
import math

def convertToSequenceParameters(timestep=datetime.timedelta(minutes=1), duration_time=datetime.timedelta(minutes=60), overlap_perc=0.5):
    """
    Args :
        - timestep : datetime.delta objects, la période de resample du dataset
        - duration_time : datetime.delta objects, la durée d'observation d'une séquence
        - overlap_perc : float in [0, 1]
    Returns :
        - SEQUENCE_LENGTH : int longueur de séquence correspondante à la duration_time selon le timestep
        - overlap_period : int
    """
    SEQUENCE_LENGTH = math.ceil(duration_time/timestep)
    overlap_period = int(SEQUENCE_LENGTH*overlap_perc)
    return SEQUENCE_LENGTH, overlap_period

def load_dataset(filename: str, resample_period :Optional[str]=None) -> pd.DataFrame:
    """
    Loads the dataset
    Args:
        filename: the path to the file to load
        resample_period: (optional) the reasmple period, if None the default period of 1 second will be used
    returns: 
        a DataFrame containing the dataset
    """
    os.getcwd()
    path = Path(os.getcwd())
    # Path('data/raw/house1_power_blk2.zip')
    path = path.parent.absolute() / 'data' / 'raw' / filename

    
    dataset = pd.read_csv(path, index_col='datetime').interpolate('linear')
    dataset.index = pd.to_datetime(dataset.index)
    dataset = dataset.asfreq('s')

    if resample_period:
        dataset = dataset.resample(resample_period).nearest()
    
    dataset['hour'] = dataset.index.hour + dataset.index.minute / 60 #+ dataset.index.seconde / 3600

    return dataset

def load_aggregate_dataset(filename: str, sub_panels:Optional[str or List[str]]='all', resample_period:Optional[str]=None) -> pd.DataFrame:
    """
    Loads the disaggregated dataset, aggregates the targetted sub-panels and removes the other columns
    filename: the path to the file to load
    sub_panels: (optional) the sub-panels to aggregate can be a list of strings containing the names of the sub-panels, or 'all' to select all panels or 'active_house1', 'active_house2', 'inactive_house1' or 'inactive_house2'
    resample_period: (optional) the reasmple period, if None the default period of 1 second will be used
    returns: a DataFrame containing the dataset
    """

    os.getcwd()
    path = Path(os.getcwd())
    # Path('data/raw/house1_power_blk2.zip')
    path = path.parent.absolute() / 'data' / 'raw' / filename

    dataset = pd.read_csv(path)
    dataset['datetime'] = pd.to_datetime(dataset['unix_ts'], unit='s')
    dataset['datetime'] = dataset['datetime'] - pd.Timedelta("8 hours")

    dataset = dataset.set_index(dataset['datetime'])
    # we drop unnecessary columns
    dataset = dataset.drop(columns=['unix_ts', 'datetime', 'ihd', 'mains'])
    dataset = dataset.asfreq('s').interpolate('linear')

    if isinstance(sub_panels, str):
        if sub_panels == 'active_house1':# [ 1, 2, 3, 4, 5, 6, 9, 10, 17, 18, 19]
            sub_panels = ['sub1', 'sub2', 'sub3', 'sub4', 'sub5', 'sub6', 'sub9', 'sub10', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19', 'sub21', 'sub22', 'sub24'] # 'sub12',  'sub21', 'sub22', 'sub24'
        elif sub_panels == 'inactive_house1':
            sub_panels = ['sub7', 'sub8', 'sub11', 'sub12', 'sub13', 'sub14', 'sub20', 'sub23']
        elif sub_panels == 'active_house2':
            sub_panels = ['sub1', 'sub2', 'sub3', 'sub4', 'sub5', 'sub6', 'sub7', 'sub9', 'sub10', 'sub12', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19', 'sub20', 'sub21']
        elif sub_panels == 'inactive_house2':
            sub_panels = ['sub8', 'sub11', 'sub13']
        elif sub_panels == 'all':
            sub_panels = dataset.columns
        else:
            raise Exception(f"Wrong value for argument sub_panels. Expected 'all', 'active_house1/2', 'inactive_house1/2' or list, got {sub_panels}")
    
    dataset['mains'] = dataset[sub_panels].sum(axis=1)
    
    dataset = dataset.drop(columns=dataset.columns[:-1])

    if resample_period:
        dataset = dataset.resample(resample_period).nearest()
    

    return dataset


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Calculates and plots the confusion matrix and prints the f_beta and accuracy scores
    y_true: the true values
    y_pred: the predictions
    returns: the f_beta and accuracy scores
    """
    f_beta = fbeta_score(y_true, y_pred, average="macro", beta=0.5)
    acc = accuracy_score(y_true, y_pred)
    print(f'Score f_beta : {f_beta:.3%}')
    print(f'Score accuracy : {acc:.3%}')
    ax = sns.heatmap(pd.crosstab(y_true, y_pred, normalize=True), annot=True, fmt='.2%', vmin=0, vmax=1, square=True, cmap=sns.cm.rocket_r);
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('vérité');
    ax.set_ylabel('predictions');

    return f_beta, acc


def plot_activity_hist(data: pd.Series, density: Optional[bool]=True, **kwargs: Any):
    """
    Plot the histogram of activity per hour of the day
    Args:
        - data: Pandas time Series containing the activity (data['activity'])
        - density (optional): whether to normalize the histogram or not (default = True_)
        - kwargs (optional): arguments passed to pyplot 
    Return:
        - None
    """
    if density:
        norm=data.sum()
    else:
        norm=1

    pd.DataFrame(data[data > 0].index.hour.value_counts().reindex(range(24)).fillna(0) / norm).reset_index(drop=False).sort_values(by='index').plot.bar(x='index', y='datetime', **kwargs)
    

def train_test_split_dataset(dataframe: pd.DataFrame, split_rate :Optional[float]=0.2)-> pd.DataFrame:
    """
    Split a dataframe into train set and test set according to the split rate
    Args:
        - Dataframe to split into train set and test set
        - split_rate: the rate of the test set size
    Return:
        - Two dataframe (Train dataframe and test dataframe)
    """
    size = dataframe.shape[0]
    test_idx = size * split_rate 
    split_idx = size - test_idx
    split_date = dataframe.iloc[int(split_idx) - 1].name
    train_df, test_df = dataframe.loc[dataframe.index <= split_date], dataframe.loc[dataframe.index > split_date]
    
    return train_df, test_df

def time_in_range(start, end, x):
        """
        Return true if x is in the range [start, end]
        """
        if start <= end:
            return start <= x <= end
        else:
            return start <= x or x <= end
            
def segmentDf(df, timeframes, use_labels=False):
    """
    Transforms df_house into a list of datframe filtered according to the periods defined in timeframes
    Args:
        - df_house: dataframe with a hands column indicating the instantaneous power and the datetime as index
        - timeframes: list of tuples indicating the periods of the day ex: timeframes = [(datetime.time(10,0,0), datetime.time(6,0,0)), (datetime.time(12,0,0), datetime.time(13,0,0))
    Returns:
        - DataFrame:
            - index: datetime of the time series
            - mains: power
            - activity: 1/0 according to the labelling done beforehand
            - hour: time of day
            - timeframe_of_interest: True/False if the period is to be used according to timeframe
            - next_toi: same column as timeframe_of_interest shifted by 1 (for range construction)
            - beginning_of_toi: beginning of a period
            - timeframe_id: period number
    """
    l_res = []
    df_house = df.copy()
    for i in range(len(timeframes)):
        df_house.loc[:, "timeframe_of_interest"] = df_house.index.map(lambda x : time_in_range(*timeframes[i], x.time())) # modifier timeframes pour itérer
        
        df_house.loc[:, "next_toi"] = df_house.loc[:,"timeframe_of_interest"].shift(1)
        df_house.loc[:,"next_toi"]= df_house.loc[:,"next_toi"].fillna(method='bfill')
        df_house.loc[:,"beginning_of_toi"] = np.where(df_house.loc[:,"timeframe_of_interest"]==df_house.loc[:,"next_toi"], 0, 1)

        df_house.loc[:,"timeframe_id"] = df_house.loc[:,'beginning_of_toi'].cumsum()

        groupby_object = df_house[df_house["timeframe_of_interest"]].groupby(["timeframe_id"])
        
        
        for key in list(groupby_object.groups.keys()):
            #print(key)
            l_res.append(df_house.loc[df_house[df_house["timeframe_of_interest"]].groupby(["timeframe_id"]).groups[key]])
        
    return l_res

def create_sequence(dataframe: pd.DataFrame, sequence_length: int, overlap_period: int) -> np.array:
    """
    dataframe: resampled dataframe
    sequence_length: length of the sequence
    overlap_period: Overlap the sequences of timeseries
    returns: Two 3D-array [samples, sequence_length, features] for the train set and the test set
    """
    dataframe = dataframe.reset_index()
    X_dataframe = dataframe[["mains"]]
    X_sequence_list = list()
    idx = 0
    length_df = X_dataframe.shape[0]
    while idx + sequence_length <= length_df - 1: 
        current_sequence =  X_dataframe.iloc[idx: idx + sequence_length].values
        X_sequence_list.append(current_sequence)
        idx = idx - overlap_period + sequence_length
    
    # generate list of index
    idx_list = list(np.arange(0, len(dataframe)))
    
    y_dataframe = dataframe[["datetime", "activity"]]
    y_dataframe["index"] = idx_list
    y_sequence_list = list()
    idx = 0
    length_df = y_dataframe.shape[0]
    while idx + sequence_length <= length_df - 1: 
        current_sequence =  y_dataframe.iloc[idx: idx + sequence_length].values
        y_sequence_list.append(current_sequence)
        idx = idx - overlap_period + sequence_length
    
    return np.array(X_sequence_list), np.array(y_sequence_list)

def read_pickle_dataset(pickle_filename):
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'src' / 'data' / pickle_filename
    df_picke = pd.read_pickle(path)
    return df_picke