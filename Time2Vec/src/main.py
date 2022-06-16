# ---- utils libs ----
import datetime
import sys


# --- Import functions from utils.py ---
sys.path.insert(0,'..')
from utils import convertToSequenceParameters, read_pickle_dataset

# --- Import functions from preprocessing.py ---
sys.path.insert(0,'..')
from preprocessing import data_preprocessing

# --- Import functions from visualize.py ---
sys.path.insert(0,'../src/visualization')
from visualize import visualize_load_curve_dataset

# --- Import functions from build_model.py ---
sys.path.insert(0,'../src/models/')
from build_model import model_embeddings, model_classifier

# --- Import functions from train_model.py ---
sys.path.insert(0,'../src/models/')
from train_model import train_embeddings, train_classifier

# --- Import functions from eval_model.py ---
sys.path.insert(0,'../src/models/')
from eval_model import plot_train_val_loss, plot_activity_histogram, plot_activity_distibrution, confusion_matrix

# --- Import functions from predict_model.py ---
sys.path.insert(0,'../src/models/')
from predict_model import X_train_predict, X_test_predict, plot_train_mse_loss, plot_test_mse_loss, get_df_predict, detect_activity_sequence, y_test_predict


# --- Import functions from postprocessing.py ---
sys.path.insert(0,'..')
from postprocessing import plot_postprocessing_anomalies


# --- Visualize Load Curve Dataset ---
print("LOADING DATASET...\n\n")
df_load_curve = visualize_load_curve_dataset("house1_power_blk2_labels.zip","60min")

# --- Define global variable ---
TIME_STEP = datetime.timedelta(minutes=1, seconds=30) # duration of a step in the resample dataset, originally 1 second
DURATION_TIME = datetime.timedelta(minutes=60) # duration of a sequence
OVERLAP_PERIOD_PERCENT = 0.8 # 0.5 <=> 50% overlapping
TIMEFRAMES = [(datetime.time(0,0,0), datetime.time(3,0,0))] # timeframes we consider as unactive

print("CONVERTING GLOBAL USER PARAMETERS...\n")
SEQUENCE_LENGTH, OVERLAP_PERIOD = convertToSequenceParameters(TIME_STEP, DURATION_TIME, OVERLAP_PERIOD_PERCENT)
print("\t\tValeur choisie \t Equivalent sequence\nTimestep : \t {}\nDuration :\t {} \t -->  {} \nOverlap :\t {} \t\t -->  {}".format(TIME_STEP, DURATION_TIME, SEQUENCE_LENGTH, OVERLAP_PERIOD_PERCENT, OVERLAP_PERIOD))


# --- Pre Processing ---
print("STARTING PREPROCESSING...\n")
train_df, test_df, X_train, y_train, X_test, y_test = data_preprocessing(timeframes = TIMEFRAMES
                                                                         ,sequence_length = SEQUENCE_LENGTH
                                                                         ,overlap_period = OVERLAP_PERIOD
                                                                         ,resample_period = TIME_STEP)
                                                                                                                         

# --- Build Model ---
print("\n\nBUILDING MODELS...")
model_emb = model_embeddings(X_train)
model_c = model_classifier(X_train)

# --- Train Model ---
print("\n\nTRAINING MODEL FOR EMBEDDINGS...")
history_emb, embeddings = train_embeddings(model_emb, X_train)

print("\n\nTRAINING MODEL FOR CLASSIFICATION...")
history_c = train_classifier(model_c, embeddings, X_train, y_train)

# --- Evaluation Model ---
print("\n\nPLOTING TRAIN AND VALIDATION LOSS...")
plot_train_val_loss(history_c)


###### X_train prediction ######
# --- Prediction Model on X_train ---
print("\n\nMAKING PREDICTION FOR X_train...")
X_train_pred = X_train_predict(model_emb, X_train)

# --- Plot train mse loss ---
print("\n\nPLOTING TRAIN MSE LOSS...")
plot_train_mse_loss(X_train_pred, X_train)


###### X_test prediction ######
# --- Prediction Model on X_test ---
print("\n\nMAKING PREDICTION FOR X_test...")
X_test_pred = X_test_predict(model_emb, X_test)

# --- Plot test mse loss ---
print("\n\nPLOTING TEST MSE LOSS...")
plot_test_mse_loss(X_test_pred, X_test)

# --- Make predictions ---
print("\n\nMAKING PREDICTION FOR y_test...")
y_test_pred = y_test_predict(model_c, X_test)
print("\n\nDETECTING ACTIVITY...")
sequences_activity = detect_activity_sequence(y_test_pred, SEQUENCE_LENGTH, OVERLAP_PERIOD)

# --- Build prediction dataframe ---
print("\n\nBUILDING PREDICTION DATAFRAME...")
df_predict = get_df_predict(sequences_activity, test_df)

# --- Plot activity histogram ---
print("\n\nPLOTING ACTIVITY HISTOGRAM...")
plot_activity_histogram(df_predict, test_df)

# --- Plot activity distribution ---
print("\n\nPLOTING ACTIVITY DISTRIBUTION...")
plot_activity_distibrution(df_predict)

# --- Plot confusion matrix ---
print("\n\nPLOTING CONFUSION MATRIX...")
confusion_matrix(test_df, df_predict)

# --- Load data prediction post processing --- #
print("\n\nLOADING DATA PREDICTION POSTPROCESSING...")
data_prediction_post_process = read_pickle_dataset("data_prediction.pkl")

# --- Plot detected anoamlies after post processing --- #
print("\n\nPLOTING DETECTED ANOMALIES (AFTER POST PROCESSING)...")
plot_postprocessing_anomalies(data_prediction_post_process, test_df)
