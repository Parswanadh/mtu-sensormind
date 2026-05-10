import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# Constants
WINDOW_SIZE = 30
ROLLING_WINDOW = 10
MAX_RUL = 125
KEY_SENSORS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
OS_SETTINGS = ['os1', 'os2', 'os3']
FEATURES = OS_SETTINGS + KEY_SENSORS

COLUMNS = ['engine_id', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]

# LSTM Autoencoder Model
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)
        
    def forward(self, x):
        # Encoder
        _, (hidden, _) = self.encoder(x)
        # Repeat hidden state for the sequence length
        hidden = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        # Decoder
        decoder_out, _ = self.decoder(hidden)
        # Output
        out = self.output_layer(decoder_out)
        return out

def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i + window_size])
    return np.array(sequences)

def train_autoencoder(train_df, scaler, device):
    print("Preparing data for LSTM Autoencoder...")
    # Filter only healthy cycles (first 70% of each engine's life)
    healthy_data = []
    for engine_id, group in train_df.groupby('engine_id'):
        max_cycle = group['cycle'].max()
        healthy_cutoff = int(max_cycle * 0.7)
        healthy_group = group[group['cycle'] <= healthy_cutoff]
        healthy_data.append(healthy_group)
    
    healthy_df = pd.concat(healthy_data)
    
    # Scale data
    scaled_data = scaler.transform(healthy_df[FEATURES])
    healthy_df_scaled = healthy_df.copy()
    healthy_df_scaled[FEATURES] = scaled_data
    
    # Create sequences per engine
    X_train = []
    for engine_id, group in healthy_df_scaled.groupby('engine_id'):
        data = group[KEY_SENSORS].values # Only reconstruct key sensors
        if len(data) >= WINDOW_SIZE:
            seqs = create_sequences(data, WINDOW_SIZE)
            X_train.extend(seqs)
            
    X_train = np.array(X_train)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    
    print(f"Training Autoencoder on {len(X_train)} sequences. Shape: {X_train.shape}")
    
    model = LSTMAutoencoder(seq_len=WINDOW_SIZE, n_features=len(KEY_SENSORS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    epochs = 15
    batch_size = 256
    
    dataset = torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
        
    return model

def prepare_xgboost_data(df, scaler):
    print("Preparing data for XGBoost RUL Predictor...")
    df_scaled = df.copy()
    df_scaled[FEATURES] = scaler.transform(df[FEATURES])
    
    # Calculate rolling features
    roll_mean = df_scaled.groupby('engine_id')[FEATURES].rolling(ROLLING_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
    roll_std = df_scaled.groupby('engine_id')[FEATURES].rolling(ROLLING_WINDOW, min_periods=1).std().reset_index(level=0, drop=True)
    roll_std.fillna(0, inplace=True)
    
    roll_mean.columns = [f"{col}_mean" for col in FEATURES]
    roll_std.columns = [f"{col}_std" for col in FEATURES]
    
    df_features = pd.concat([df_scaled[['engine_id', 'cycle', 'RUL']], roll_mean, roll_std], axis=1)
    
    X = df_features.drop(['engine_id', 'cycle', 'RUL'], axis=1)
    y = df_features['RUL']
    return X, y

def train_xgboost(X_train, y_train):
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_FD001.txt')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please download CMAPSS dataset as per data/README.md")
        return
        
    print("Loading data...")
    train_df = pd.read_csv(data_path, sep='\s+', header=None, names=COLUMNS)
    
    # Calculate RUL
    rul_per_engine = train_df.groupby('engine_id')['cycle'].max()
    train_df['max_cycle'] = train_df['engine_id'].map(rul_per_engine)
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
    train_df['RUL'] = train_df['RUL'].clip(upper=MAX_RUL)
    train_df.drop('max_cycle', axis=1, inplace=True)
    
    # Fit Scaler
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES])
    
    # Train Models
    autoencoder = train_autoencoder(train_df, scaler, device)
    X_xgb, y_xgb = prepare_xgboost_data(train_df, scaler)
    xgb_model = train_xgboost(X_xgb, y_xgb)
    
    # Save Artifacts
    models_dir = os.path.dirname(__file__)
    print("Saving models...")
    
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    torch.save(autoencoder.state_dict(), os.path.join(models_dir, 'lstm_autoencoder.pt'))
    
    with open(os.path.join(models_dir, 'rul_xgboost.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)
        
    print("Training complete! Models saved in 'models/' directory.")

if __name__ == "__main__":
    main()
