# ---- utils libs ----
import os
from pathlib import Path
import pandas as pd
import numpy as np

# ---- Data Viz libs ---- 
import plotly.graph_objects as go

def data_postprocessing(y_test, SEQUENCE_LENGTH, sequences_anomalies_idx):
    """
    Post process the model prediction with different strategies
    Args:
        - y_test: 3D-array that contain sequence of timestamp, activity labels and index (Ex: [Timestamp('2016-04-25 08:48:00'), 0, 0])
        - sequence_length : The length of the y_test sequence
        - data_anomalies_idx : The list of index of each sequence predict as an anomaly by the model
    Returns:
        - DataFrame:
            - Timestamp: datetime of the time series 
            - list_idx_sequence_no_activity : index of the sequence for which no activity is predicted
            - list_idx_sequence_activity : index of the sequence for which activity is predicted
            - nb_no_activity : number of times the timestamp is in a sequence for which no activity has been predicted by the model 
            - nb_activity : number of times the timestamp is in a sequence for which activity has been predicted by the model
            - total : nb_no_activity + nb_activity
            - method_prediction_1 : Process a Majority Vote between no_activity_rate attribute and activity_rate attribute
            
        - Export DataFrame to pickle format
    """
    
    # Init dataframe with prediction
    data_prediction = pd.DataFrame(columns=['Timestamp', 
                              'list_idx_sequence_no_activity',
                              'list_idx_sequence_activity', 
                              'nb_no_activity',
                              'nb_activity', 
                              'total'])

    timestamp_list = list()
    for i in range(y_test.shape[0]):
        for k in range(SEQUENCE_LENGTH):
            timestamp_list.append(y_test[i][k][0])

    # drop duplicate
    timestamp_list = list(dict.fromkeys(timestamp_list))
    print(len(timestamp_list))

    counter = 0
    for timestamp in timestamp_list:
        counter = counter + 1
        print(str(timestamp) + ", " + str(counter))
        list_idx_sequence_no_activity = list()
        list_idx_sequence_activity = list()
        for i in range(y_test.shape[0]):
            for k in range(SEQUENCE_LENGTH):
                if timestamp == y_test[i][k][0]:
                    if i in sequences_anomalies_idx:
                        list_idx_sequence_activity.append(i)
                    else:
                        list_idx_sequence_no_activity.append(i)

        data_prediction = data_prediction.append({'Timestamp': timestamp, 
                          'list_idx_sequence_no_activity': list_idx_sequence_no_activity,
                          'list_idx_sequence_activity': list_idx_sequence_activity, 
                          'nb_no_activity': len(list_idx_sequence_no_activity),
                          'nb_activity': len(list_idx_sequence_activity), 
                          'total': len(list_idx_sequence_no_activity) + len(list_idx_sequence_activity)}
                          ,ignore_index=True)
        
    ### Majortity vote post process strategy ###
    # Process a **Majority Vote** between no_activity_rate attribute and activity_rate attribute
    data_prediction["method_prediction_1"] = np.where(data_prediction["nb_activity"] > data_prediction["nb_no_activity"], 1, 0)

        
    # Export prediction to .pickle format
    os.getcwd()
    path = Path(os.getcwd())
    path = path.parent.absolute() / 'src' / 'data' / 'data_prediction.pkl'

    data_prediction.to_pickle(path)
        
    return data_prediction

def plot_postprocessing_anomalies(data_prediction_post_process, test_df):
    
    # Detect all the samples which are anomalies.
    anomalies_method_1 = data_prediction_post_process["method_prediction_1"] == 1
    
    idx_anomalies = np.where(anomalies_method_1)
    
    df_anomalies = data_prediction_post_process.iloc[idx_anomalies]
    test_df_value = test_df["mains"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df_value.index, y=test_df_value.values, name='Normal'))
    fig.add_trace(go.Scatter(x=df_anomalies["Timestamp"], y=df_anomalies["method_prediction_1"], mode='markers', name='Anomaly = Activity (Predicted)'))
    fig.update_layout(showlegend=True, title='Detected anomalies with method prediction 1')
    
    os.getcwd()
    path_bis = Path(os.getcwd())
    path_bis = path_bis.parent.absolute() / 'reports' / 'figures' / 'detected_anomalies_method_1.html'
    fig.write_html(path_bis)