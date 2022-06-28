# ---- utils libs ----
import datetime
import sys
import os
from pathlib import Path


# --- Import functions from utils.py ---
sys.path.insert(0,'..')
from utils import convertToSequenceParameters, read_pickle_dataset

# --- Import functions from preprocessing.py ---
sys.path.insert(0,'..')
from preprocessing import data_preprocessing

# --- Import functions from visualize.py ---
sys.path.insert(0,'../src/visualization')
from visualize import visualize_load_curve_dataset, visualize_load_curve_resampled, visualize_train_load_curve, visualize_test_load_curve, visualize_test_train_load_curve, visualize_report_preprocessing

# --- Import functions from build_model.py ---
sys.path.insert(0,'../src/models/')
from build_model import model

# --- Import functions from train_model.py ---
sys.path.insert(0,'../src/models/')
from train_model import train

# --- Import functions from eval_model.py ---
sys.path.insert(0,'../src/models/')
from eval_model import plot_train_val_loss, plot_reconstructed_base_load_curve, plot_activity_histogram, plot_activity_distibrution, confusion_matrix, evaluate

# --- Import functions from predict_model.py ---
sys.path.insert(0,'../src/models/')
from predict_model import X_train_predict, X_test_predict, compute_threshold, compute_train_mae_loss, compute_test_mae_loss, plot_train_mae_loss, plot_test_mae_loss, detect_anomaly_sequence, get_anomaly_df, get_df_predict, get_specific_anomaly_sequence, plot_anomaly_sequences, plot_anomaly_data_points

# --- Import functions from postprocessing.py ---
sys.path.insert(0,'..')
from postprocessing import data_postprocessing, plot_postprocessing_anomalies


# --- Define global variable ---
DATASET = "house1_power_blk2_labels.zip"
TIME_STEP = datetime.timedelta(minutes=1, seconds=30) # duration of a step in the resample dataset, originally 1 second
DURATION_TIME = datetime.timedelta(minutes=60) # duration of a sequence
OVERLAP_PERIOD_PERCENT = 0.8 # 0.5 <=> 50% overlapping
TIMEFRAMES = [(datetime.time(0,0,0), datetime.time(3,0,0))] # timeframes we consider as unactive
STRATEGY = "off_peak_time" # device, off_peak_time, label 
METHOD = "method_prediction_1" # method to choose for aggregating sequences
SPLIT_METHOD = "random_days" # method for train test split, None or "random_days"

print("======== INPUT HYPERPARAMETERS SUMMARY ========")
print("DATASET : ", DATASET)
print("TIME_STEP : ", TIME_STEP)
print("DURATION_TIME : ", DURATION_TIME)
print("OVERLAP_PERIOD_PERCENT : ", OVERLAP_PERIOD_PERCENT)
print("TIMEFRAMES : ", TIMEFRAMES)
print("STRATEGY : ", STRATEGY)
print("METHOD : ", METHOD)
print("SPLIT_METHOD : ", SPLIT_METHOD)

# --- Save Hyperparameters in a text file ---
os.getcwd()
path = Path(os.getcwd())
path = path.parent.absolute() / 'reports' / 'input_hyperparameters_summary.txt'
    
with open(path, 'w') as f:
    f.write("======== INPUT HYPERPARAMETERS SUMMARY ========")
    f.write('\n\n')
    f.write("DATASET : " + str(DATASET))
    f.write("TIME_STEP : " + str(TIME_STEP))
    f.write('\n')
    f.write("DURATION_TIME : " + str(DURATION_TIME))
    f.write('\n')
    f.write("OVERLAP_PERIOD_PERCENT : " + str(OVERLAP_PERIOD_PERCENT))
    f.write('\n')
    f.write("TIMEFRAMES : " + str(TIMEFRAMES))
    f.write('\n')
    f.write("STRATEGY : " + str(STRATEGY))
    f.write('\n')
    f.write("METHOD : " + str(METHOD))
    f.write('\n')
    f.write("SPLIT_METHOD : " + str(SPLIT_METHOD))

# --- Converting global variable for the model ---
print("\n\nCONVERTING GLOBAL USER PARAMETERS...")
SEQUENCE_LENGTH, OVERLAP_PERIOD = convertToSequenceParameters(TIME_STEP, DURATION_TIME, OVERLAP_PERIOD_PERCENT)
print("\t\tChoosen value \t Equivalent sequence\nTimestep : \t {}\nDuration :\t {} \t -->  {} \nOverlap :\t {} \t\t -->  {}".format(TIME_STEP, DURATION_TIME, SEQUENCE_LENGTH, OVERLAP_PERIOD_PERCENT, OVERLAP_PERIOD))


# --- Load Curve Dataset ---
print("\n\nLOAD DATASET...\n\n")
df_load_curve = visualize_load_curve_dataset(DATASET)


# --- Visualize Load Curve ---
print("\n\nPLOTING LOAD CURVE RESAMPLED...\n\n")
load_curve_resampled = visualize_load_curve_resampled(DATASET,"60min")

# --- Pre Processing ---
print("STARTING PREPROCESSING...\n")
train_df, test_df, X_train, y_train, X_test, y_test = data_preprocessing(timeframes = TIMEFRAMES
                                                                         ,sequence_length = SEQUENCE_LENGTH
                                                                         ,overlap_period = OVERLAP_PERIOD
                                                                         ,resample_period = TIME_STEP
                                                                         ,strategy = STRATEGY
                                                                         ,split_method=SPLIT_METHOD)

print("PRINTING PREPROCESSING REPORT...\n")
report_classification = visualize_report_preprocessing(X_train, y_train, X_test, y_test
                                                       ,TIMEFRAMES
                                                       ,SEQUENCE_LENGTH
                                                       ,OVERLAP_PERIOD
                                                       ,TIME_STEP
                                                       ,STRATEGY)
                                                                                                                         

print("\nPLOTING TRAIN LOAD CURVE (base load curve)(" + STRATEGY + ")...")
visualize_train_load_curve(train_df, STRATEGY)

print("\nPLOTING TEST LOAD CURVE (" + STRATEGY + ")...")
visualize_test_load_curve(test_df, STRATEGY)

print("\nPLOTING TRAIN TEST LOAD CURVE (" + STRATEGY + ")...")
visualize_test_train_load_curve(train_df, test_df, STRATEGY)

# --- Build Model ---
print("\n\nBUILDING MODEL...")
model = model(X_train)

# --- Train Model ---
print("\n\nTRAININ MODEL...")
history = train(model, X_train)

# --- Evaluation Model ---
print("\n\nPLOTING TRAIN AND VALIDATION LOSS...")
plot_train_val_loss(history)


###### X_train prediction ######
# --- Prediction Model on X_train ---
print("\n\nMAKING PREDICTION FOR X_train...")
X_train_pred = X_train_predict(model, X_train)

# --- Compute train mae loss
print("\n\nCOMPUTING TRAIN MAE LOSS...")
train_mae_loss = compute_train_mae_loss(X_train_pred, X_train)

# --- Compute Threshold ---
print("\n\nCOMPUTING THRESHOLD...")
threshold = compute_threshold(X_train_pred, X_train)

# --- Plot train mae loss ---
print("\n\nPLOTING TRAIN MAE LOSS...")
plot_train_mae_loss(X_train_pred, X_train)



# --- Plot reconstructed base load curve
print("\n\nPLOTING RECONSTRUCTED BASE LOAD CURVE...")
plot_reconstructed_base_load_curve(X_train, X_train_pred)


###### X_test prediction ######
# --- Prediction Model on X_test ---
print("\n\nMAKING PREDICTION FOR X_test...")
X_test_pred = X_test_predict(model, X_test)

# --- Compute test mae loss
print("\n\nCOMPUTING TEST MAE LOSS...")
test_mae_loss = compute_test_mae_loss(X_test_pred, X_test)

# --- Plot test mae loss ---
print("\n\nPLOTING TEST MAE LOSS...")
plot_test_mae_loss(X_test_pred, X_test)


# --- Detect anomalies ---
print("\n\nDETECTING ANOMALIES...")
sequences_anomalies_idx, data_anomalies_idx = detect_anomaly_sequence(test_mae_loss, threshold, SEQUENCE_LENGTH, y_test)

# --- Build anomaly dataframe ---
print("\n\nBUILDING ANOMALY DATAFRAME...")
df_anomalies = get_anomaly_df(data_anomalies_idx, test_df)
print(df_anomalies)

# --- Build prediction dataframe ---
print("\n\nBUILDING PREDICTION DATAFRAME...")
df_predict = get_df_predict(test_df, df_anomalies)

# --- Visualize Specific sequence anomalies ---
print("\n\nEXPORTING SPECIFIC SEQUENCE ANOMALIES...")
get_specific_anomaly_sequence(0, sequences_anomalies_idx, y_test)

# --- Plot anomaly sequences ---
print("\n\nPLOTING ANOMALY SEQUENCE...")
plot_anomaly_sequences(test_df, data_anomalies_idx)

# --- Plot anomaly data points ---
print("\n\nPLOTING ANOMALY DATA POINTS...")
plot_anomaly_data_points(test_df, data_anomalies_idx, threshold)

# --- Plot activity histogram ---
print("\n\nPLOTING ACTIVITY HISTOGRAM...")
plot_activity_histogram(df_anomalies, test_df)

# --- Plot activity distribution ---
print("\n\nPLOTING ACTIVITY DISTRIBUTION...")
plot_activity_distibrution(df_predict)

# --- Plot confusion matrix ---
print("\n\nPLOTING CONFUSION MATRIX...")
confusion_matrix(test_df, df_predict)

# --- Post Processing --- #
print("\n\nPOSTPROCESSING...")
data_postprocessing(y_test, SEQUENCE_LENGTH, sequences_anomalies_idx, True)

# --- Load data prediction post processing --- #
print("\n\nLOADING DATA PREDICTION POSTPROCESSING...")
data_prediction_post_process = read_pickle_dataset("data_prediction.pkl")

# --- Plot detected anoamlies after post processing --- #
print("\n\nPLOTING DETECTED ANOMALIES (AFTER POST PROCESSING)...")
plot_postprocessing_anomalies(data_prediction_post_process, test_df)

# --- PLot direct and IoU threshold
print("\n\nPLOTING EVALUATION PLOT (DIRECT AND IoU THRESHOLD)...")


y_pred = data_prediction_post_process[["Timestamp", METHOD]]
y_true = df_load_curve[(df_load_curve.index>=y_pred["Timestamp"].min())&(df_load_curve.index<=y_pred["Timestamp"].max())].reset_index()[["datetime", "activity"]]
y_true = y_true[y_true["datetime"].isin(y_pred["Timestamp"])] #restriction de y_true aux timestamps contenus dans y_pred

IoU_thresholds, MAP, MAR = evaluate(y_pred, y_true, display_plots=True)