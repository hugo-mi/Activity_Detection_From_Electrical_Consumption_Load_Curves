from typing import List, Set, Dict, Tuple, Optional, Any
import os
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
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
    # Path('Data/house1_power_blk2.zip')
    path = path.parent.absolute() / 'Data' / filename

    
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
    # Path('Data/house1_power_blk2.zip')
    path = path.parent.absolute() / 'Data' / filename

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

def pick_random_indexes(data: pd.DataFrame, percentage: Optional[float]=0.3) -> pd.DatetimeIndex:
    """
    Returns random indexes from the index of the DataFrame passed in parameter
    data: the DataFrame to use
    percentage: the percentage of indexes to randomly pick
    returns: a DatetimeIndex with random dates
    """
    # tirage de jours aléatoires
    delta_time = data.index[-2] - data.index[0]
    nb_days = int(delta_time.days * percentage)
    random_dates = data.index[0] + pd.to_timedelta(np.random.choice(delta_time.days, nb_days, replace=False), unit='day')

    data_freq = pd.Timedelta(data.index.freq).seconds if data.index.freq else 1
    # définition des indexes test, train
    rand_indexes = pd.DatetimeIndex(np.array([pd.date_range(d, periods=24*60*60/data_freq, freq=data.index.freq) for d in random_dates]).ravel())

    return rand_indexes

def split_train_test_indexes(data: pd.DataFrame, percentage: Optional[float]=0.3) -> Tuple[pd.DatetimeIndex]:
    """
    Generate train and test indexes on a time series by picking random full days
    data: the DataFrame to use
    percentage: the percentage of indexes to randomly pick for the test
    returns: a tuple of DatetimeIndex with random days for the train indexes and test indexes
    """
    # tirage de jours aléatoires
    delta_time = data.index[-2] - data.index[0]
    nb_days = int(delta_time.days * percentage)
    random_dates = data.index[0] + pd.to_timedelta(np.random.choice(delta_time.days, nb_days, replace=False), unit='day')

    data_freq = pd.Timedelta(data.index.freq).seconds if data.index.freq else 1
    # définition des indexes test, train
    test_indexes = pd.DatetimeIndex(np.array([pd.date_range(d, periods=24*60*60/data_freq, freq=data.index.freq) for d in random_dates]).ravel())
    train_indexes = data.index[~np.isin(data.index, test_indexes)]

    return train_indexes, test_indexes

def split_train_test_scale_df(data: pd.DataFrame, features_col:List[str], label_col: Optional[List[str]]=['activity'], percentage: Optional[float]=0.3, scaler: Optional[Any]=StandardScaler()) -> Tuple[np.array]:
    """
    Performs a split train test on a time series by picking random full days and then scales then feature columns
    data: the DataFrame to use
    features_col: a list containing the names of the feature columns
    features_col: a list containing the name of the label column
    percentage: the percentage of indexes to randomly pick for the test
    scale: (optional) the scaler to use
    returns: a tuple consisting of X_train, X_test, y_train, y_test
    """
    # tirage de jours aléatoires
    train_indexes, test_indexes = split_train_test_indexes(data, percentage)

    # on crée un DF normalisé
    data_norm = data[features_col + label_col].copy()

    # on fit le scaler et normalise le jeu de train
    data_norm.loc[train_indexes, features_col] = scaler.fit_transform(data_norm.loc[train_indexes, features_col].values)
    # on normalise le jeu de test
    data_norm.loc[test_indexes, features_col] = scaler.transform(data_norm.loc[test_indexes, features_col].values)

    # on génère les X/y train/test
    X_train, X_test = data_norm.loc[train_indexes, features_col].values, data_norm.loc[test_indexes, features_col].values
    y_train, y_test = data_norm.loc[train_indexes, 'activity'].values, data_norm.loc[test_indexes, 'activity'].values

    return X_train, X_test, y_train, y_test

def generate_scaled_features(data: pd.DataFrame, column_name: Optional[str]='mains', window: Optional[str]='1h', scaler: Optional[Any]=StandardScaler(), fillna_method :Optional[str]='bfill') -> Tuple[pd.DataFrame, List[str], StandardScaler]:
    """
    Generates scaled features for classifications
    data: the DataFrame to use
    column_name: the name of the column with the power data
    window: (optional) the window of time for the rolling transformations
    scaler: (optional) the scaler to use
    fillna_method: (optional) the method to use the fill na values that will occure due to the rolling transformations
    returns: a DataFrame with the old and new features, a list containing the names of the new features columns, and the fitter scaler
    """
    warnings.warn("This function is depreciated, use generate_features and manually scale the features instead", DeprecationWarning)

    # we prepare our features
    data['mains_scaled'] = data[column_name].values.reshape(-1,1)
    data['mean_'+window+'_scaled'] = data[column_name].rolling(window).mean().values.reshape(-1,1)
    data['std_'+window+'_scaled'] = data[column_name].rolling(window).std().values.reshape(-1,1)
    data['maxmin_'+window+'_scaled'] = data[column_name].rolling(window).max().values.reshape(-1,1) - data[column_name].rolling(window).min().values.reshape(-1,1)
    data['peaks_'+window+'_scaled'] = ((data[column_name] - data['mean_'+window]) < 1e-3).astype(int).rolling(window, center=True).sum().values.reshape(-1,1)
    data['hour_scaled'] = data['hour'].values.reshape(-1,1)
    data['weekend'] = data.index.day_of_week.isin([5, 6]).astype(int)

    # we fill the na values with the chosen method
    data = data.fillna(method=fillna_method)

    # we generate a list of the column names generated
    features_col = ['mains_scaled', 'hour_scaled', 'std_'+window+'_scaled', 'mean_'+window+'_scaled', 'maxmin_'+window+'_scaled']

    # we fit the data
    data[features_col] = scaler.fit_transform(data[features_col].values)

    return data, features_col+['weekend'], scaler

def generate_features(data: pd.DataFrame, column_name: Optional[str]='mains', window: Optional[str or List[str]]='1h', fillna_method :Optional[str]='bfill') -> Tuple[pd.DataFrame, List[str]]:
    """
    Generates features for classifications
    data: the DataFrame to use
    column_name: the name of the column with the power data
    windows: (optional) the window(s) of time for the rolling transformations
    fillna_method: (optional) the method to use the fill na values that will occure due to the rolling transformations
    returns: a DataFrame with the old and new features, a list containing the names of the new features columns, and the fitter scaler
    """
    # we prepare our features
    if not isinstance(window, list):
        window = [window]

    features_col = []
    for w in window:
        data['mean_'+w] = data[column_name].rolling(w, center=True).mean().values.reshape(-1,1)
        data['std_'+w] = data[column_name].rolling(w, center=True).std().values.reshape(-1,1)
        data['maxmin_'+w] = data[column_name].rolling(w, center=True).max().values.reshape(-1,1) - data[column_name].rolling(w, center=True).min().values.reshape(-1,1)
        data['peaks_'+w] = ((data[column_name] - data['mean_'+w]) < 1e-3).astype(int).rolling(w, center=True).sum().values.reshape(-1,1)

        # we generate a list of the column names generated
        features_col += ['std_'+w, 'mean_'+w, 'maxmin_'+w, 'peaks_'+w]

    data['weekend'] = data.index.day_of_week.isin([5, 6]).astype(int)
    features_col += ['weekend']
    
    # we remove the NA values
    data = data.fillna(method=fillna_method)


    return data, features_col

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


def plot_scores_param(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                      estimator: Any, param_name: str, param_range: List[float], other_params: Optional[Dict[str, Any]]={}, recalculate_scores: Optional[bool]=True) -> Tuple[str, float, float]:
    """
    Performs a grid search on a model on a single parameter and displays the scores vs parameter
    X_train: the features to train the model on
    y_train: the labels to train the model
    X_test: the features to evaluate the model on
    y_test: the labels to evaluate the model
    estimator: the estimator to fit
    param_name: the name of the parameter on which we perform the grid search
    param_range: the range of the parameter to perform the grid search
    other_params: (optional) additional parameters to pass to the classifier
    recalculate_scores: (optional) whether or not we recalculate the score with the best parameters to plot a confusion matrix
    returns: the parameter name, its the best value, the accuracy and the fbeta score associated with the best value
    """
    f2_score = []
    score = []
    for p in param_range:
        classifier = estimator(**{param_name:p}, **other_params)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        f2_score.append((fbeta_score(y_test, y_pred, average='macro', beta=0.5)))
        score.append(accuracy_score(y_test, y_pred))

        print(f"tested {param_name}={p} ...")

    best_param = np.argmax(f2_score)

    plt.figure(figsize=(10, 6));
    plt.plot(param_range, score, label='score', color='grey', linestyle='dashed');
    plt.plot(param_range, f2_score, label='fb score');
    plt.scatter(param_range[best_param], f2_score[best_param], label='fb max', marker='x', s=100, color='red')
    plt.legend();
    plt.title('fb and score = f({})'.format(param_name));
    plt.show();

    print('Meilleur fb score={:.2f} obtenu pour {}={:.2f}'.format(f2_score[best_param], param_name, param_range[best_param]))

    if recalculate_scores:
        classifier = estimator(**{param_name:param_range[best_param]})
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        plot_confusion_matrix(y_test, y_pred)

    return param_name, param_range[best_param], score[best_param], f2_score[best_param]

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


def activity_periods(data_activity: pd.Series)->pd.Series:
    """
    Extracts the activity time periods
    Args:
       data_activity: the time series from which we want to extract the activity time periods
    returns: a pandas series with a line for each activity time period as a pandas Intervalle
    """
    data_activity.iloc[0] = 0
    data_activity.iloc[-1] = 0
    
           # on définit un nouveau dataframe avec diff = 1 si début d'activité, -1 si fin
    return pd.DataFrame(data_activity.diff(1)[data_activity.diff(1)!=0].dropna().replace(-1, 'end').replace(1, 'begin')) \
           .reset_index().set_axis(['datetime', 'activity'], axis=1, inplace=False)\
           .pivot(values='datetime', columns='activity').fillna(method='ffill').iloc[1::2, :].reset_index(drop=True).apply(lambda x: pd.Interval(x['begin'], x['end'], closed='neither'), axis=1) # on fait un pivot, ffil et on supprime une ligne sur deux

def detect_overlaps(data_act_true: pd.Series, data_act_pred: pd.Series)-> pd.Series:
    """
    Defines for each line of data_act_true how many lines of data_act_pred it overlaps
    Args:
        data_act_true: true activity periods
        data_act_pred: predicted activity periods
    return: A pandas series with the for each data_act_true period how many pperiod of data_act_pred they overlap
    """
    return data_act_true.apply(lambda x: np.array([x.overlaps(data_act_pred.iloc[i]) for i in range(data_act_pred.shape[0])]).sum())

def score_overlap(activity_true, activity_pred, resample_period='30min'):
    """
    Determines the number of activity periods that are detected (TP) and the number of activity periods detected when there was no true activity (FP)
    Args:
        activity_true: time series that contains the true activity labels
        activity_pred: time series that contains the predicted activity labels
        resample_period (optional): the time period to use to resample both time series
    returns: the percentage of true activity periods that are detected (TP) and the percentage of activity periods predicted when there was no true activity (FP)
    """

    # we start by smoothing the data
    activity_true = (activity_true.rolling(resample_period).mean()>0).astype(int).copy()
    activity_pred = (activity_pred.rolling(resample_period).mean()>0).astype(int).copy()

    # we get the activity perdiods for each activity series
    activity_per_true = activity_periods(activity_true)
    activity_per_pred = activity_periods(activity_pred)

    # we get the count of true activity periods
    T = activity_per_true.shape[0]
    F = activity_per_pred.shape[0]

    # we estimate the true positives and false positives
    TP = (detect_overlaps(activity_per_true, activity_per_pred) > 0).sum()
    FP = (detect_overlaps(activity_per_pred, activity_per_true) == 0).sum()

    return TP/T, FP/F

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


def detect_stages(dataframe, col, col_datetime):
    """
    Transforme une dataframe de time series en une dataframe des périodes délimitées par la colonne binaire col
    Args :
        -df_house : dataframe avec index datetime et une colonne col binaire
        -col : colonne binaire utilisée pour séparer les périodes
        -col_datetime : colonne contenant les timestamps
    Return :
        - df_stages : dataframe avec pour chaque ligne une période : (col, début, fin, durée en minutes, durée en secondes)
    """
    df_house=dataframe.copy()
    df_house["next"] = df_house[col].shift(1)
    df_house["next"]=df_house["next"].fillna(method="bfill").astype(int)
    df_house["switch"] = np.where(df_house["next"]==df_house[col], 0, 1)
    df_house["stage"]=df_house["switch"].cumsum()
    df_house=df_house.reset_index().groupby(by='stage').agg({col : ['mean'], col_datetime: ['min', 'max']})
    df_house.columns = ['_'.join(col) for col in df_house.columns.values]
    df_house=df_house.rename(columns={col+"_mean": col})
    df_house[col]=df_house[col].astype(int)
    df_house["duration_min"]=(df_house[col_datetime+"_max"]-df_house[col_datetime+"_min"]).astype("timedelta64[m]")
    df_house["duration_sec"]=(df_house[col_datetime+"_max"]-df_house[col_datetime+"_min"]).astype("timedelta64[s]")
    return df_house





### ============ EVALUATION ============ 


def get_TPTNFPFN(df_merged, col_pred, col_gt="activity"):
    """
    Computes TP, TN FP FN for an input dataframe with prediction and ground_truth columns 
    Args :
        -
    Return :
        - 
    """
    df = df_merged.copy()
    df["TP"] = np.where((df[col_pred]==1)&(df[col_gt]==1), 1, 0)
    df["TN"] = np.where((df[col_pred]==0)&(df[col_gt]==0), 1, 0)
    df["FP"] = np.where((df[col_pred]==1)&(df[col_gt]==0), 1, 0)
    df["FN"] = np.where((df[col_pred]==0)&(df[col_gt]==1), 1, 0)

    return df


def get_IoU(df_period, col_period_min, col_period_max, ts_min, ts_max, col_bin,  activity):
    """
    Computes IoU between the period [ts_min, ts_max] and the corresponding periods of df_gt for activity/inactivity specified by activity 
    Args :
        -
    Return :
        - 
    """
    df = df_period[(df_period[col_period_max]>=ts_min)&(df_period[col_period_min]<=ts_max)].copy()
    df=df.loc[df[col_bin]==activity, :]
    
    # Compute Intersection I
    df["datetime_min_cut"] = ts_min
    df["datetime_min_cut"] = np.maximum(df[col_period_min], df["datetime_min_cut"])
    df["datetime_max_cut"] = ts_max
    df["datetime_max_cut"] = np.minimum(df[col_period_max], df["datetime_max_cut"])
    I = np.sum(df["datetime_max_cut"] - df["datetime_min_cut"]).seconds
    
    
    # Compute Union U
    U = np.sum(df["duration_sec"]) + (ts_max-ts_min).seconds - I # nécessité d'avoir une colonne "duration_sec" dans les dataframes utilisées

    IoU = I/U

    return df, U, I, IoU

def get_activity_stages(pred_period, col_method):
    return pred_period[pred_period[col_method]==1].copy()

def broken_barh_x(df, col_bin, col_ts_min, col_ts_max):
    s1 = df.loc[df[col_bin]==1, col_ts_min]
    s0 = df.loc[df[col_bin]==0, col_ts_min]

    s1_length = df.loc[df[col_bin]==1, col_ts_max] - df.loc[df[col_bin]==1, col_ts_min]
    s0_length = df.loc[df[col_bin]==0, col_ts_max] - df.loc[df[col_bin]==0, col_ts_min]

    times1 = list(zip(s1,s1_length))
    times0 = list(zip(s0,s0_length))
    
    return times1, times0

def eval(pred, df_gt, display_plots=True):
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

        ax[0,0].set_yticks([0.5, 1.5], labels=['pred', 'gt'])
        ax[0,0].legend()
        ax[0,0].set_title("Pred and Ground Truth")

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
        
    return (tau_range, map, mar)