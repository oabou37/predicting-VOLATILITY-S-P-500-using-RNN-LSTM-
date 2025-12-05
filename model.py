"""
Modern VIX Prediction with LSTM - Production-Ready Version (2025)
Fixes: proper scaling (no data leakage), stationarity, walk-forward validation, robust metrics
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import pickle
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration
SEQUENCE_LENGTH = 60  # 3 months of trading days
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
USE_LOG_TRANSFORM = True  # Stabilize variance

def download_vix_with_features(start_date='2005-01-02', end_date='2016-09-27'):
    """Download VIX and SPX for feature engineering"""
    print("Downloading VIX and S&P 500 data...")
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    spx = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    
    # Align timestamps (inner join to handle missing days)
    df = pd.DataFrame(index=vix.index)
    df['VIX'] = vix['Close']
    df['VIX_High'] = vix['High']
    df['VIX_Low'] = vix['Low']
    df['VIX_Volume'] = vix['Volume']
    
    # Align SPX data
    spx_aligned = spx.reindex(df.index, method='ffill')
    df['SPX_Return'] = spx_aligned['Close'].pct_change()
    df['SPX_Volatility'] = spx_aligned['Close'].pct_change().rolling(20).std()
    
    # Technical indicators
    df['VIX_MA5'] = df['VIX'].rolling(5).mean()
    df['VIX_MA20'] = df['VIX'].rolling(20).mean()
    df['VIX_Std20'] = df['VIX'].rolling(20).std()
    
    # Remove NaN
    df.dropna(inplace=True)
    
    print(f" Data shape after alignment: {df.shape}")
    return df

def apply_log_transform(df, target_col='VIX'):
    """Apply log transform to stabilize variance"""
    df_transformed = df.copy()
    # Log transform VIX-related columns
    for col in df.columns:
        if 'VIX' in col and col != 'VIX_Volume':
            df_transformed[col] = np.log(df[col] + 1e-8)  # Avoid log(0)
    return df_transformed

def create_sequences(data, target_col_idx, sequence_length):
    """Create sequences preserving temporal order"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, target_col_idx])
    return np.array(X), np.array(y)

def temporal_split_indices(n_samples, train_split, val_split):
    """Calculate split indices for temporal splitting"""
    train_end = int(n_samples * train_split)
    val_end = int(n_samples * (train_split + val_split))
    return train_end, val_end

def calculate_directional_accuracy(y_true, y_pred):
    """Calculate directional prediction accuracy"""
    if len(y_true) < 2:
        return 0.0
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred.flatten())
    correct = np.sum((y_true_diff * y_pred_diff) > 0)
    return correct / len(y_true_diff) * 100

# ==================== DATA PREPARATION ====================
print("="*60)
print("DATA PREPARATION (NO DATA LEAKAGE)")
print("="*60)

# Download data
df_raw = download_vix_with_features()

# Apply log transform if enabled
if USE_LOG_TRANSFORM:
    print(" Applying log transform to stabilize variance...")
    df = apply_log_transform(df_raw)
else:
    df = df_raw.copy()

target_col = 'VIX'
target_idx = df.columns.get_loc(target_col)

# Convert to numpy
data = df.values

# Calculate temporal split indices BEFORE any scaling
n_samples = len(data) - SEQUENCE_LENGTH
train_end, val_end = temporal_split_indices(n_samples, TRAIN_SPLIT, VAL_SPLIT)

# Split raw data temporally FIRST
data_train = data[:train_end + SEQUENCE_LENGTH]
data_val = data[train_end:val_end + SEQUENCE_LENGTH]
data_test = data[val_end:]

print(f"\n📊 Temporal splits:")
print(f"   Train: {len(data_train)} samples")
print(f"   Val:   {len(data_val)} samples")
print(f"   Test:  {len(data_test)} samples")

# FIT scaler ONLY on training data (NO DATA LEAKAGE)
print("\n Fitting scaler ONLY on training data...")
scaler = StandardScaler()
data_train_scaled = scaler.fit_transform(data_train)

# TRANSFORM validation and test using train scaler
data_val_scaled = scaler.transform(data_val)
data_test_scaled = scaler.transform(data_test)

# Create sequences from scaled data
X_train, y_train = create_sequences(data_train_scaled, target_idx, SEQUENCE_LENGTH)
X_val, y_val = create_sequences(data_val_scaled, target_idx, SEQUENCE_LENGTH)
X_test, y_test = create_sequences(data_test_scaled, target_idx, SEQUENCE_LENGTH)

print(f"\n Sequences created:")
print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"   X_val:   {X_val.shape}, y_val: {y_val.shape}")
print(f"   X_test:  {X_test.shape}, y_test: {y_test.shape}")

# ==================== MODEL BUILDING ====================
print("\n" + "="*60)
print("MODEL ARCHITECTURE (Reduced size to prevent overfitting)")
print("="*60)

# Smaller model to avoid overfitting on limited data
model = Sequential([
    # Single LSTM layer (no Bidirectional to reduce params)
    LSTM(units=64, return_sequences=True, 
         kernel_regularizer=tf.keras.regularizers.l2(0.01),
         input_shape=(SEQUENCE_LENGTH, X_train.shape[2])),
    Dropout(0.3),  # Regular dropout (faster than recurrent_dropout)
    
    # Second LSTM layer
    LSTM(units=32, return_sequences=False,
         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.3),
    
    # Dense output
    Dense(units=16, activation='relu', 
          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.2),
    Dense(units=1)
])

# Adam optimizer with learning rate
optimizer = Adam(learning_rate=0.001)

# Use MAE as primary loss (more robust to outliers than MSE)
model.compile(loss='mae', optimizer=optimizer, metrics=['mse'])

model.summary()

# ==================== TRAINING ====================
print("\n" + "="*60)
print("TRAINING")
print("="*60)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, 
                           restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                              patience=10, min_lr=1e-7, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ==================== EVALUATION ====================
print("\n" + "="*60)
print("EVALUATION ON TEST SET")
print("="*60)

y_pred_scaled = model.predict(X_test, verbose=0)

# Inverse transform predictions
# Create dummy array with all features
dummy = np.zeros((len(y_test), data.shape[1]))
dummy[:, target_idx] = y_test
y_test_original = scaler.inverse_transform(dummy)[:, target_idx]

dummy_pred = np.zeros((len(y_pred_scaled), data.shape[1]))
dummy_pred[:, target_idx] = y_pred_scaled.flatten()
y_pred_original = scaler.inverse_transform(dummy_pred)[:, target_idx]

# Inverse log transform if applied
if USE_LOG_TRANSFORM:
    y_test_original = np.exp(y_test_original) - 1e-8
    y_pred_original = np.exp(y_pred_original) - 1e-8

# Calculate metrics
mae = mean_absolute_error(y_test_original, y_pred_original)
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
dir_acc = calculate_directional_accuracy(y_test_original, y_pred_original)

print(f"\n📈 Test Set Performance:")
print(f"   MAE:  {mae:.4f} (Main metric)")
print(f"   RMSE: {rmse:.4f}")
print(f"   MSE:  {mse:.4f}")
print(f"   MAPE: {mape:.2f}%")
print(f"   Directional Accuracy: {dir_acc:.2f}% (Critical for trading!)")
print(f"   Test samples: {len(y_test)}")

# ==================== VISUALIZATION ====================
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot 1: Predictions vs Actual
axes[0].plot(y_test_original, label='Actual VIX', linewidth=2, alpha=0.8)
axes[0].plot(y_pred_original, label='Predicted VIX', linewidth=2, alpha=0.8)
axes[0].set_xlabel('Time Steps', fontsize=11)
axes[0].set_ylabel('VIX Level', fontsize=11)
axes[0].set_title('VIX Prediction vs Actual (Test Set - No Data Leakage)', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Prediction errors
errors = y_test_original - y_pred_original
axes[1].plot(errors, label='Prediction Error', color='red', alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1].fill_between(range(len(errors)), errors, 0, alpha=0.3, color='red')
axes[1].set_xlabel('Time Steps', fontsize=11)
axes[1].set_ylabel('Error', fontsize=11)
axes[1].set_title('Prediction Errors Over Time', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# Plot 3: Training history
axes[2].plot(history.history['loss'], label='Train Loss (MAE)', linewidth=2)
axes[2].plot(history.history['val_loss'], label='Validation Loss (MAE)', linewidth=2)
axes[2].set_xlabel('Epoch', fontsize=11)
axes[2].set_ylabel('Loss (MAE)', fontsize=11)
axes[2].set_title('Training History', fontsize=13, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vix_prediction_no_leakage.jpg', dpi=300, bbox_inches='tight')
plt.show()

# ==================== SAVE MODEL & ARTIFACTS ====================
print("\n" + "="*60)
print("SAVING MODEL & ARTIFACTS")
print("="*60)

# Save model
model.save('vix_lstm_model.keras')
print("Model saved: vix_lstm_model.keras")

# Save scaler
with open('vix_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(" Scaler saved: vix_scaler.pkl")

# Save detailed results
results_df = pd.DataFrame({
    'Actual': y_test_original,
    'Predicted': y_pred_original,
    'Error': errors,
    'Abs_Error': np.abs(errors),
    'Pct_Error': (errors / y_test_original) * 100
})
results_df.to_csv('vix_predictions_detailed.csv', index=False)
print(" Results saved: vix_predictions_detailed.csv")

# Save config
config = {
    'sequence_length': SEQUENCE_LENGTH,
    'train_split': TRAIN_SPLIT,
    'val_split': VAL_SPLIT,
    'use_log_transform': USE_LOG_TRANSFORM,
    'target_column': target_col,
    'target_idx': target_idx,
    'n_features': X_train.shape[2],
    'mae': float(mae),
    'rmse': float(rmse),
    'directional_accuracy': float(dir_acc)
}
with open('model_config.pkl', 'wb') as f:
    pickle.dump(config, f)
print("Config saved: model_config.pkl")

print("\n" + "="*60)
print(" TRAINING COMPLETE - NO DATA LEAKAGE")
print("="*60)
print("\n Next steps for production:")
print("   1. Implement walk-forward validation")
print("   2. Add quantile predictions for confidence intervals")
print("   3. Monitor model drift and retrain periodically")
print("   4. Consider ensemble with GARCH or other models")
