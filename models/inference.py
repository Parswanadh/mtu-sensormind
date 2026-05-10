import os
import torch
import pandas as pd
import numpy as np
import pickle
from .train_models import LSTMAutoencoder, WINDOW_SIZE, ROLLING_WINDOW, KEY_SENSORS, FEATURES

class MTUEnginePredictor:
    def __init__(self):
        self.models_dir = os.path.dirname(__file__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        
        self.scaler = None
        self.autoencoder = None
        self.xgb_model = None
        
        self.ANOMALY_THRESHOLD = 0.5 
        
    def load_models(self):
        if self.is_loaded:
            return
            
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        ae_path = os.path.join(self.models_dir, 'lstm_autoencoder.pt')
        xgb_path = os.path.join(self.models_dir, 'rul_xgboost.pkl')
        
        if not all(os.path.exists(p) for p in [scaler_path, ae_path, xgb_path]):
            raise FileNotFoundError("Models not found. Please run train_models.py first.")
            
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        self.autoencoder = LSTMAutoencoder(seq_len=WINDOW_SIZE, n_features=len(KEY_SENSORS)).to(self.device)
        self.autoencoder.load_state_dict(torch.load(ae_path, map_location=self.device, weights_only=True))
        self.autoencoder.eval()
        
        with open(xgb_path, 'rb') as f:
            self.xgb_model = pickle.load(f)
            
        self.is_loaded = True

    def check_sensor_sanity(self, latest_window_df):
        """
        Data Quality Gate: Checks for common industrial sensor faults (e.g., frozen sensors)
        before passing data to the ML models.
        """
        warnings = []
        for sensor in KEY_SENSORS:
            # Check for frozen sensor (0 variance over the window)
            if latest_window_df[sensor].var() == 0:
                warnings.append(f"DATA FAULT: {sensor} is FROZEN (0 variance over {WINDOW_SIZE} cycles). Inspect sensor hardware.")
        return warnings

    def get_engine_status(self, engine_id, latest_window_df):
        if not self.is_loaded:
            self.load_models()
            
        if len(latest_window_df) < WINDOW_SIZE:
            raise ValueError(f"Insufficient data. Expected at least {WINDOW_SIZE} cycles, got {len(latest_window_df)}")
            
        # 1. Data Quality Check
        dq_warnings = self.check_sensor_sanity(latest_window_df)
            
        # 2. Scale data
        df_scaled = latest_window_df.copy()
        df_scaled[FEATURES] = self.scaler.transform(latest_window_df[FEATURES])
        
        # 3. Anomaly Detection (LSTM)
        ae_data = df_scaled.iloc[-WINDOW_SIZE:][KEY_SENSORS].values
        ae_input = torch.tensor(ae_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            ae_output = self.autoencoder(ae_input)
            mse_loss = torch.nn.functional.mse_loss(ae_output, ae_input, reduction='none')
            anomaly_score = mse_loss.mean().item()
            
        is_anomaly = bool(anomaly_score > self.ANOMALY_THRESHOLD)
        
        latest_cycle_true = ae_input[0, -1, :].cpu().numpy()
        latest_cycle_pred = ae_output[0, -1, :].cpu().numpy()
        deviations = np.abs(latest_cycle_true - latest_cycle_pred)
        
        sensor_deviations = {
            KEY_SENSORS[i]: float(deviations[i]) 
            for i in range(len(KEY_SENSORS))
        }
        
        # 4. RUL Prediction (XGBoost)
        df_roll_data = df_scaled.iloc[-ROLLING_WINDOW:]
        roll_mean = df_roll_data[FEATURES].mean().values
        roll_std = df_roll_data[FEATURES].std().values
        roll_std = np.nan_to_num(roll_std)
        
        xgb_input = np.concatenate([roll_mean, roll_std]).reshape(1, -1)
        rul_prediction = max(0, int(self.xgb_model.predict(xgb_input)[0]))
        
        # 5. Risk Level Assessment
        risk_level = "LOW"
        if len(dq_warnings) > 0:
            risk_level = "DATA_FAULT" # Sensor issue, not necessarily engine failure yet
        elif rul_prediction < 20 or (is_anomaly and anomaly_score > self.ANOMALY_THRESHOLD * 2):
            risk_level = "CRITICAL"
        elif rul_prediction < 50 or is_anomaly:
            risk_level = "HIGH"
        elif rul_prediction < 80:
            risk_level = "MEDIUM"
            
        return {
            "engine_id": engine_id,
            "anomaly_score": round(anomaly_score, 4),
            "is_anomaly": is_anomaly,
            "rul_prediction": rul_prediction,
            "risk_level": risk_level,
            "sensor_deviations": sensor_deviations,
            "data_quality_warnings": dq_warnings
        }
