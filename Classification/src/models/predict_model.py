## Detecting anomalies

"""
Make predictions based on pur model and generate a timeseries with predictions
"""

# ---- utils libs ----
import numpy as np
import pandas as pd
import os
from pathlib import Path


def y_test_predict(model, X_test):
    y_test_pred = model.predict(X_test)
    print("\n---- y_test_pred sequence shape ----")
    print(y_test_pred.shape)
    
    return y_test_pred
    
def get_df_predict(y_test_pred, test_df):
    df_predict = test_df.copy()

    df_predict.iloc[:len(y_test_pred),:].loc[:, 'activity'] = y_test_pred
    df_predict.iloc[len(y_test_pred):, :].loc[:, 'activity'] = np.nan
    df_predict = df_predict.dropna(axis=0)

    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'reports' / 'prediction_dataframe.txt'

    with open(path, mode='w') as f:
        df_predict.to_csv(f)

    return df_predict