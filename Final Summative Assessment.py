import os
from pathlib import Path

# Reduce TensorFlow terminal messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.get_logger().setLevel("ERROR")

# ============================================================
# mODEL SETTINGS
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

TICKER = "GOOGL"
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

SEQUENCE_LENGTH = 60

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

EPOCHS = 150
BATCH_SIZE = 32

# Output Folder Setup
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

RUN_NAME = f"bs{BATCH_SIZE}_adam_default_lr"

BASE_OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR = BASE_OUTPUT_DIR / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nOutput folder will be saved at:")
print(OUTPUT_DIR.resolve())


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_ratio_as_percent(value):
    percent = value * 100

    if float(percent).is_integer():
        return f"{int(percent)}%"
    else:
        return f"{percent:.1f}%"

def format_date_range(date_array):
    start_date = pd.to_datetime(date_array[0]).strftime("%d %B %Y")
    end_date = pd.to_datetime(date_array[-1]).strftime("%d %B %Y")
    return f"{start_date} to {end_date}"

# ============================================================
# QUESTION 1: DATA COLLECTION AND PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("QUESTION 1: DATA COLLECTION AND PREPROCESSING")
print("=" * 70)


# Download Google stock price data
raw_download_df = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False
)

if raw_download_df.empty:
    raise ValueError("No stock data was downloaded. Please check the ticker or internet connection.")

print("\nRaw Data Head:")
print(raw_download_df.head().to_string())

# Data Cleaning
raw_df = raw_download_df.copy()

if isinstance(raw_df.columns, pd.MultiIndex):
    raw_df.columns = raw_df.columns.get_level_values(0)

raw_df = raw_df[["Close", "High", "Low", "Open", "Volume"]].copy()
raw_df = raw_df.dropna()

print("\nDataset shape:", raw_df.shape)

print("\nMissing values:")
print(raw_df.isnull().sum())

# Select Close price for model input
df = raw_df[["Close"]].copy()

# Original closing price plot
plt.figure(figsize=(14, 5))
plt.plot(df.index, df["Close"], label="GOOGL Close Price")
plt.title("GOOGL Historical Closing Price")
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.grid(True)
plt.tight_layout()

historical_price_path = OUTPUT_DIR / "figure_1_1_historical_closing_price.png"
plt.savefig(historical_price_path, dpi=300, bbox_inches="tight")
plt.show()


# Data spilting to prevent data leakage
close_values = df[["Close"]].values
n_total = len(close_values)

if TRAIN_RATIO + VAL_RATIO >= 1.0:
    raise ValueError("TRAIN_RATIO + VAL_RATIO must be less than 1.0 so that test data exists.")

if n_total <= SEQUENCE_LENGTH + 10:
    raise ValueError(
        "Dataset is too small for the selected SEQUENCE_LENGTH. "
        "Use a longer date range or reduce SEQUENCE_LENGTH."
    )

train_raw_end = int(n_total * TRAIN_RATIO)
val_raw_end = int(n_total * (TRAIN_RATIO + VAL_RATIO))

train_raw_end = min(train_raw_end, n_total - 2)
val_raw_end = min(val_raw_end, n_total - 1)

train_raw_end = max(train_raw_end, SEQUENCE_LENGTH + 1)
val_raw_end = max(val_raw_end, train_raw_end + 1)

# Data Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(close_values[:train_raw_end])

scaled_close = scaler.transform(close_values)

# Sequence Preparation
def create_sequences(data, dates, sequence_length):
    X = []
    y = []
    sequence_dates = []

    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i, 0])
        y.append(data[i, 0])
        sequence_dates.append(dates[i])

    X = np.array(X)
    y = np.array(y)
    sequence_dates = np.array(sequence_dates)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y, sequence_dates


X, y, sequence_dates = create_sequences(
    scaled_close,
    df.index,
    SEQUENCE_LENGTH
)

print("\nSequence Dataset Format:")
print(f"X shape: {X.shape}  -> (samples, time steps, features)")
print(f"y shape: {y.shape}  -> target values")
print(
    f"Each input sample uses {SEQUENCE_LENGTH} previous trading days "
    f"and {X.shape[2]} feature: normalized Close price."
)

# Chronological train-validation-test split
train_end = train_raw_end - SEQUENCE_LENGTH
val_end = val_raw_end - SEQUENCE_LENGTH

train_end = max(train_end, 1)
val_end = max(val_end, train_end + 1)
val_end = min(val_end, len(X) - 1)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

dates_train = sequence_dates[:train_end]
dates_val = sequence_dates[train_end:val_end]
dates_test = sequence_dates[val_end:]

print("\nFinal RNN Input Format After Chronological Split:")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape:   {X_val.shape}")
print(f"X_test shape:  {X_test.shape}")
print("Format explanation: (number of samples, 60 time steps, 1 feature)")

# chronological split summary table
split_summary_table = pd.DataFrame({
    "Subset": ["Training", "Validation", "Test"],
    "Ratio": [
        format_ratio_as_percent(TRAIN_RATIO),
        format_ratio_as_percent(VAL_RATIO),
        format_ratio_as_percent(TEST_RATIO)
    ],
    "Sequences": [
        f"{len(X_train):,}",
        f"{len(X_val):,}",
        f"{len(X_test):,}"
    ],
    "Target Date Range": [
        format_date_range(dates_train),
        format_date_range(dates_val),
        format_date_range(dates_test)
    ]
})

print("\nSummary of Chronological Train, Validation and Test Split:")
print(split_summary_table.to_string(index=False))


# ============================================================
# QUESTION 3: MODEL DEVELOPMENT, TRAINING, AND VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("QUESTION 3: MODEL DEVELOPMENT, TRAINING, AND VISUALIZATION")
print("=" * 70)

INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])

# ML models
def build_simple_rnn(input_shape):
    model = Sequential(name="Simple_RNN_Model")

    model.add(Input(shape=input_shape))
    model.add(SimpleRNN(64))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.Huber(),
        metrics=["mse"]
    )

    return model

def build_gru(input_shape):
    model = Sequential(name="GRU_Model")

    model.add(Input(shape=input_shape))
    model.add(GRU(64))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.Huber(),
        metrics=["mse"]
    )

    return model

def build_lstm(input_shape):
    model = Sequential(name="LSTM_Model")

    model.add(Input(shape=input_shape))
    model.add(LSTM(64))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.Huber(),
        metrics=["mse"]
    )

    return model

# Training function with EarlyStopping and ReduceLROnPlateau callbacks
def train_model(model, model_name):
    print("\n" + "=" * 70)
    print(f"Training {model_name}")
    print("=" * 70)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=False,
        verbose=1
    )

    return model, history


# Model Creation
models = {
    "Simple RNN": build_simple_rnn(INPUT_SHAPE),
    "GRU": build_gru(INPUT_SHAPE),
    "LSTM": build_lstm(INPUT_SHAPE)
}


# Model Architecture Summary
print("\n" + "=" * 70)
print("MODEL ARCHITECTURE SUMMARY")
print("=" * 70)

for model_name, model in models.items():
    print("\n" + "-" * 70)
    print(f"Model: {model_name}")
    print("-" * 70)
    model.summary()

# Train models
trained_models = {}
histories = {}

for model_name, model in models.items():
    trained_model, history = train_model(model, model_name)

    trained_models[model_name] = trained_model
    histories[model_name] = history

# Training and validation loss plot
for model_name, history in histories.items():
    plt.figure(figsize=(9, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name}: Training Loss vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    safe_model_name = model_name.lower().replace(" ", "_")
    loss_path = OUTPUT_DIR / f"figure_3_training_validation_loss_{safe_model_name}.png"

    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# QUESTION 4: MODEL EVALUATION AND CRITICAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("QUESTION 4: MODEL EVALUATION AND CRITICAL ANALYSIS")
print("=" * 70)


# Evaluation function
def evaluate_model(y_true_scaled, y_pred_scaled):
    y_true_actual = scaler.inverse_transform(
        y_true_scaled.reshape(-1, 1)
    ).ravel()

    y_pred_actual = scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()

    mse = mean_squared_error(y_true_actual, y_pred_actual)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true_actual, y_pred_actual)
    mape = mean_absolute_percentage_error(y_true_actual, y_pred_actual) * 100
    r2 = r2_score(y_true_actual, y_pred_actual)

    return y_true_actual, y_pred_actual, mse, rmse, mae, mape, r2

# Model prediction and evaluation
results = []
predictions = {}

for model_name, model in trained_models.items():
    y_pred_scaled = model.predict(X_test, verbose=0).ravel()

    y_true_actual, y_pred_actual, mse, rmse, mae, mape, r2 = evaluate_model(
        y_test,
        y_pred_scaled
    )

    predictions[model_name] = y_pred_actual

    results.append({
        "Model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "R2 Score": r2,
        "Parameters": model.count_params()
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("RMSE").reset_index(drop=True)

actual_test_price = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).ravel()

# Final metric comparison table
comparison_table = results_df[
    ["Model", "MSE", "RMSE", "MAE", "MAPE (%)", "R2 Score", "Parameters"]
].copy()

comparison_table = comparison_table.round({
    "MSE": 4,
    "RMSE": 4,
    "MAE": 4,
    "MAPE (%)": 4,
    "R2 Score": 4
})

print("\nTable 4.1: Final Metric Comparison between Simple RNN, GRU, and LSTM")
print(comparison_table.to_string(index=False))


# Prediction comparison graph for all models
plt.figure(figsize=(14, 6))

plt.plot(
    dates_test,
    actual_test_price,
    label="Actual Test Price",
    linewidth=2,
    color="black"
)

for model_name, pred_values in predictions.items():
    plt.plot(
        dates_test,
        pred_values,
        label=f"{model_name} Predicted Price",
        linestyle="--"
    )

plt.title("Actual Price vs ML Model Predictions")
plt.xlabel("Date")
plt.ylabel("GOOGL Stock Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()

all_model_prediction_path = OUTPUT_DIR / "figure_4_1_actual_vs_all_model_predictions.png"
plt.savefig(all_model_prediction_path, dpi=300, bbox_inches="tight")
plt.show()

# Full Time Series Forecasting Graph
def plot_forecast_result(model_name, prediction_values):
    plt.figure(figsize=(14, 6))

    training_data = df["Close"].iloc[:train_raw_end]
    validation_data = df["Close"].iloc[train_raw_end:val_raw_end]
    actual_test_data = df["Close"].iloc[val_raw_end:]

    plt.plot(
        training_data.index,
        training_data.values,
        label="Training Data",
        color="blue"
    )

    plt.plot(
        validation_data.index,
        validation_data.values,
        label="Validation Data",
        color="orange"
    )

    plt.plot(
        actual_test_data.index,
        actual_test_data.values,
        label="Actual Test Price",
        color="green",
    )

    plt.plot(
        dates_test,
        prediction_values,
        label=f"{model_name} Predicted Test Price",
        linewidth=1,
        color="red",
        linestyle="--",
        marker="x",
        markersize=2
    )

    plt.title(f"{TICKER} Stock Price Forecasting using {model_name}")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    safe_model_name = model_name.lower().replace(" ", "_")
    forecast_path = OUTPUT_DIR / f"figure_4_2_forecasting_graph_{safe_model_name}.png"

    plt.savefig(forecast_path, dpi=300, bbox_inches="tight")
    plt.show()

for model_name, pred_values in predictions.items():
    plot_forecast_result(model_name, pred_values)

# Actual vs Predicted Graph for each model (Zoomed-in)
def plot_zoomed_actual_vs_predicted(model_name, prediction_values):
    plt.figure(figsize=(14, 6))

    plt.plot(
        dates_test,
        actual_test_price,
        label="Actual Test Price",
        linewidth=1,
        color="green",
        marker="o",
        markersize=2
    )

    plt.plot(
        dates_test,
        prediction_values,
        label=f"{model_name} Predicted Price",
        linewidth=1,
        color="red",
        linestyle="--",
        marker="x",
        markersize=2
    )

    plt.title(f"Test Set Actual vs Predicted Stock Prices for {model_name} Model")
    plt.xlabel("Date")
    plt.ylabel("GOOGL Stock Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    safe_model_name = model_name.lower().replace(" ", "_")
    zoomed_path = OUTPUT_DIR / f"figure_4_3_zoomed_actual_vs_predicted_{safe_model_name}.png"

    plt.savefig(zoomed_path, dpi=300, bbox_inches="tight")
    plt.show()


for model_name, pred_values in predictions.items():
    plot_zoomed_actual_vs_predicted(model_name, pred_values)


# output CSV file for final metric results
metrics_csv_path = OUTPUT_DIR / "google_stock_prediction_metric_results_original_scale.csv"
results_df.to_csv(metrics_csv_path, index=False)

# output CSV file for model settings summary
settings_summary = {
    "Ticker": TICKER,
    "Start Date": START_DATE,
    "End Date": END_DATE,
    "Sequence Length": SEQUENCE_LENGTH,
    "Train Ratio": TRAIN_RATIO,
    "Validation Ratio": VAL_RATIO,
    "Test Ratio": TEST_RATIO,
    "Epochs": EPOCHS,
    "Batch Size": BATCH_SIZE,
    "Loss Function": "Huber",
    "Model Compile Metric": "MSE",
    "Learning Rate": "Adam default",
    "Dropout": 0.1,
    "EarlyStopping Patience": 20,
    "ReduceLROnPlateau Patience": 5,
    "Input Feature": "Close price only",
    "Main Proposed Model": "GRU",
    "Comparison Models": "Simple RNN and LSTM"
}

settings_df = pd.DataFrame(
    list(settings_summary.items()),
    columns=["Setting", "Value"]
)

settings_csv_path = OUTPUT_DIR / "model_settings_summary.csv"
settings_df.to_csv(settings_csv_path, index=False)

# End of program summary
best_model = results_df.iloc[0]

print("\nSaved files location:")
print(OUTPUT_DIR.resolve())

print("\nSaved files:")
print("1. figure_1_1_historical_closing_price.png")
print("2. figure_3_training_validation_loss_simple_rnn.png")
print("3. figure_3_training_validation_loss_gru.png")
print("4. figure_3_training_validation_loss_lstm.png")
print("5. figure_4_1_actual_vs_all_model_predictions.png")
print("6. figure_4_2_forecasting_graph_simple_rnn.png")
print("7. figure_4_2_forecasting_graph_gru.png")
print("8. figure_4_2_forecasting_graph_lstm.png")
print("9. figure_4_3_zoomed_actual_vs_gru_predicted_price.png")
print("10. google_stock_prediction_metric_results_original_scale.csv")
print("11. model_settings_summary.csv")

print("\n" + "=" * 70)
print("PROGRAM COMPLETED")
print("=" * 70)

print(f"\nMain proposed model: GRU")
print(f"Best model based on RMSE: {best_model['Model']}")
print(f"Lowest RMSE: {best_model['RMSE']:.4f}")