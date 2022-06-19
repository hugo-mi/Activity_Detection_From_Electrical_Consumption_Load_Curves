# ---- utils libs ----
import datetime
import sys


# --- Import functions from utils.py ---
sys.path.insert(0,'..')
from utils import read_pickle_dataset

# --- Import functions from preprocessing.py ---
sys.path.insert(0,'..')
from preprocessing import data_preprocessing

# --- Import functions from visualize.py ---
sys.path.insert(0,'../src/visualization')
from visualize import visualize_load_curve_dataset

# --- Import functions from build_model.py ---
sys.path.insert(0,'../src/models/')
from build_model import model_classifier

# --- Import functions from train_model.py ---
sys.path.insert(0,'../src/models/')
from train_model import train_classifier

# --- Import functions from eval_model.py ---
sys.path.insert(0,'../src/models/')
from eval_model import plot_activity_histogram, plot_activity_distibrution, confusion_matrix

# --- Import functions from predict_model.py ---
sys.path.insert(0,'../src/models/')
from predict_model import get_df_predict, y_test_predict

# --- Import functions from eval_model.py ---
sys.path.insert(0,'../src/models/')
from save_model import save_model, load_model


# --- Import functions from postprocessing.py ---
sys.path.insert(0,'..')
from postprocessing import plot_postprocessing_anomalies


# --- Visualize Load Curve Dataset ---
print("LOADING DATASET...\n\n")
df_load_curve = visualize_load_curve_dataset("house1_power_blk2_labels.zip","60min")

# --- Define global variable ---
TIME_STEP = datetime.timedelta(minutes=1, seconds=30) # duration of a step in the resample dataset, originally 1 second

# --- Pre Processing ---
print("STARTING PREPROCESSING...\n")
train_df, test_df, X_train, y_train, X_test, y_test = data_preprocessing(resample_period = TIME_STEP)

### Below lines are commented since we will load an existing model
## Uncomment them to build a new model
# --- Build Model ---
print("\n\nBUILDING MODEL...")
model = model_classifier()

print("\n\nTRAINING MODEL FOR CLASSIFICATION...")
train_classifier(model, X_train, y_train)

# --- Save Model ---
print("\n\nSAVING MODEL...")
save_model(model)

# --- Load Model ---
print("\n\nLOADING MODEL...")
model = load_model()

# --- Make predictions ---
print("\n\nMAKING PREDICTION FOR y_test...")
y_test_pred = y_test_predict(model, X_test)

# --- Build prediction dataframe ---
print("\n\nBUILDING PREDICTION DATAFRAME...")
df_predict = get_df_predict(y_test_pred, test_df)

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
