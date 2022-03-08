from typing import List, Set, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score, accuracy_score

def load_dataset(filename: str, resample_period :Optional[str]=None) -> pd.DataFrame:
    """
    Loads the dataset
    filename: the path to the file to load
    resample_period: (optional) the reasmple period, if None the default period of 1 second will be used
    returns: a DataFrame containing the dataset
    """
    dataset = pd.read_csv(filename, index_col='datetime').interpolate('linear')
    dataset.index = pd.to_datetime(dataset.index)
    dataset = dataset.asfreq('s')

    if resample_period:
        dataset = dataset.resample(resample_period).nearest()
    
    dataset['hour'] = dataset.index.hour

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
    Performs a split train test on a time series by picking random full days
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

def generate_scaled_features(data: pd.DataFrame, column_name: Optional[str]='mains', window: Optional[str]='1h', scaler: Optional[Any]=StandardScaler(), fillna_method :Optional[str]='bfill') -> Tuple[pd.DataFrame, List[str]]:
    """
    Generates features for classifications
    data: the DataFrame to use
    column_name: the name of the column with the power data
    window: (optional) the window of time for the rolling transformations
    scale: (optional) the scaler to use
    fillna_method: (optional) the method to use the fill na values that will occure due to the rolling transformations
    returns: a DataFrame with the old and new features and the list containing the names of the new features columns
    """
    # we prepare our features
    data['mains_scaled'] = scaler.fit_transform(data[column_name].values.reshape(-1,1))
    data['mean_'+window+'_scaled'] = scaler.fit_transform(data[column_name].rolling(window).mean().values.reshape(-1,1))
    data['std_'+window+'_scaled'] = scaler.fit_transform(data[column_name].rolling(window).std().values.reshape(-1,1))
    data['maxmin_'+window+'_scaled'] = scaler.transform(data[column_name].rolling(window).max().values.reshape(-1,1) - data[column_name].rolling(window).min().values.reshape(-1,1))
    data['hour_scaled'] = scaler.fit_transform(data['hour'].values.reshape(-1,1))
    data = data.fillna(method=fillna_method)

    features_col = ['mains_scaled', 'hour_scaled', 'std_'+window+'_scaled', 'mean_'+window+'_scaled', 'maxmin_'+window+'_scaled']

    return data, features_col

def plot_scores_param(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                      estimator: Any, param_name: str, param_range: List[float], other_params: Optional[Dict[str, Any]]={}) -> Tuple[str, float, float]:
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
    returns: the parameter name, its the best value, the accuracy and the f2 score associated with the best value
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

    classifier = estimator(**{param_name:param_range[best_param]})
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    sns.heatmap(pd.crosstab(y_test, y_pred, normalize=True), annot=True, fmt='.1%', vmin=0, vmax=1, cmap=sns.cm.rocket_r);
    plt.xlabel('vérité');
    plt.ylabel('prédiction');
    plt.title('Resultats avec {}={:.2f}'.format(param_name, param_range[best_param]));
    plt.show();

    return param_name, param_range[best_param], score[best_param], f2_score[best_param]