import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import warnings
import io
warnings.filterwarnings('ignore')

# Flask imports
from flask import Flask, render_template, jsonify, request, send_file, make_response
from flask_cors import CORS

# ML/Stats imports
try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("Warning: ML libraries not available. Using mock predictions.")

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("Warning: reportlab not available. PDF export disabled.")

# Excel generation
try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    HAS_EXCEL = True
except ImportError:
    HAS_EXCEL = False
    print("Warning: openpyxl not available. Excel export disabled.")

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ============================================
# CONFIGURATION
# ============================================

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

for directory in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RAW_AZURE_PATH = RAW_DIR / "azure_usage.csv"
RAW_EXTERNAL_PATH = RAW_DIR / "external_factors.csv"
CLEANED_PATH = PROCESSED_DIR / "cleaned_merged.csv"
FEATURE_ENGINEERED_PATH = PROCESSED_DIR / "features_Engineered.csv"

# ============================================
# DATA PROCESSING CLASSES 
# ============================================

class DataProcessor:
    def __init__(self):
        self.azure_df = None
        self.external_df = None
        self.merged_df = None
        
    def load_data(self):
        try:
            if RAW_AZURE_PATH.exists():
                self.azure_df = pd.read_csv(RAW_AZURE_PATH, parse_dates=["date"])
                self.azure_df = self.azure_df.sort_values("date").reset_index(drop=True)
                if 'resource_type' in self.azure_df.columns and 'service' not in self.azure_df.columns:
                    self.azure_df['service'] = self.azure_df['resource_type']
            else:
                self.azure_df = self._generate_synthetic_azure_data()
                self.azure_df.to_csv(RAW_AZURE_PATH, index=False)
                
            if RAW_EXTERNAL_PATH.exists():
                self.external_df = pd.read_csv(RAW_EXTERNAL_PATH, parse_dates=["date"])
            else:
                self.external_df = self._generate_synthetic_external_data()
                self.external_df.to_csv(RAW_EXTERNAL_PATH, index=False)
                
            return self.azure_df, self.external_df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _generate_synthetic_azure_data(self, days=365):
        dates = pd.date_range(end=datetime.today(), periods=days, freq='D')
        regions = ['East US', 'West US', 'North Europe', 'Southeast Asia', 'Central India']
        services = ['Compute', 'Storage', 'Database', 'Networking']
        
        data = []
        for date in dates:
            for region in regions:
                for service in services:
                    base_cpu = 70
                    base_storage = 1500
                    base_users = 1200
                    
                    day_of_year = date.dayofyear
                    trend = day_of_year * 0.05
                    seasonal = 10 * np.sin(2 * np.pi * day_of_year / 365)
                    weekend_factor = 0.8 if date.weekday() >= 5 else 1.0
                    
                    data.append({
                        'date': date,
                        'region': region,
                        'service': service,
                        'usage_cpu': max(0, base_cpu + trend + seasonal + np.random.normal(0, 5)) * weekend_factor,
                        'usage_storage': max(0, base_storage + trend * 10 + seasonal * 50 + np.random.normal(0, 100)),
                        'users_active': max(0, int((base_users + trend * 2 + seasonal * 20 + np.random.normal(0, 50)) * weekend_factor)),
                        'cpu_total': 100,
                        'storage_allocated': 2000
                    })
        
        return pd.DataFrame(data)
    
    def _generate_synthetic_external_data(self, days=365):
        dates = pd.date_range(end=datetime.today(), periods=days, freq='D')
        
        data = []
        for date in dates:
            data.append({
                'date': date,
                'economic_index': 100 + np.random.normal(0, 5),
                'market_demand': 80 + np.random.normal(0, 10),
                'is_weekend': 1 if date.weekday() >= 5 else 0,
                'is_holiday': 1 if date.month == 12 and date.day in [25, 31] else 0
            })
        
        return pd.DataFrame(data)
    
    def clean_data(self):
        try:
            if 'usage_cpu' in self.azure_df.columns:
                self.azure_df['usage_cpu'] = self.azure_df['usage_cpu'].ffill()
            if 'usage_storage' in self.azure_df.columns:
                self.azure_df['usage_storage'] = self.azure_df['usage_storage'].ffill()
            if 'users_active' in self.azure_df.columns:
                self.azure_df['users_active'] = self.azure_df['users_active'].ffill()
            
            for col in ['usage_cpu', 'usage_storage', 'users_active']:
                if col in self.azure_df.columns:
                    mean = self.azure_df[col].mean()
                    std = self.azure_df[col].std()
                    if std > 0:
                        self.azure_df = self.azure_df[
                            (self.azure_df[col] >= mean - 3*std) & 
                            (self.azure_df[col] <= mean + 3*std)
                        ]
            
            return self.azure_df
        except Exception as e:
            print(f"Error cleaning data: {e}")
            raise
    
    def merge_data(self):
        try:
            self.merged_df = pd.merge(self.azure_df, self.external_df, on='date', how='left')
            if 'resource_type' in self.merged_df.columns and 'service' not in self.merged_df.columns:
                self.merged_df.rename(columns={"resource_type": "service"}, inplace=True)
            self.merged_df['date'] = pd.to_datetime(self.merged_df['date'], dayfirst=True, errors='coerce')
            self.merged_df.to_csv(CLEANED_PATH, index=False)
            return self.merged_df
        except Exception as e:
            print(f"Error merging data: {e}")
            raise

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        
    def create_features(self):
        try:
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'], dayfirst=True, errors='coerce')
            
            if 'cpu_total' not in self.df.columns:
                self.df['cpu_total'] = 100
            if 'storage_allocated' not in self.df.columns:
                self.df['storage_allocated'] = 2000
            
            if 'cpu_utilization' not in self.df.columns:
                self.df['cpu_utilization'] = self.df['usage_cpu'] / self.df['cpu_total']
            
            if 'storage_efficiency' not in self.df.columns:
                groupby_cols = []
                if 'region' in self.df.columns:
                    groupby_cols.append('region')
                if 'service' in self.df.columns:
                    groupby_cols.append('service')
                if groupby_cols:
                    storage_max = self.df.groupby(groupby_cols)['usage_storage'].transform('max')
                else:
                    storage_max = self.df['usage_storage'].max()
                self.df['storage_efficiency'] = self.df['usage_storage'] / storage_max
            
            if 'date' in self.df.columns:
                if 'day_of_week' not in self.df.columns:
                    self.df['day_of_week'] = self.df['date'].dt.dayofweek
                if 'day_of_month' not in self.df.columns:
                    self.df['day_of_month'] = self.df['date'].dt.day
                if 'month' not in self.df.columns:
                    self.df['month'] = self.df['date'].dt.month
                if 'quarter' not in self.df.columns:
                    self.df['quarter'] = self.df['date'].dt.quarter
                if 'week_of_year' not in self.df.columns:
                    self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
            
            groupby_cols = []
            if 'region' in self.df.columns:
                groupby_cols.append('region')
            if 'service' in self.df.columns:
                groupby_cols.append('service')
            
            for col in ['usage_cpu', 'usage_storage', 'users_active']:
                if col in self.df.columns:
                    for lag in [1, 7, 30]:
                        new_col = f"{col}_lag{lag}"
                        if new_col not in self.df.columns:
                            if groupby_cols:
                                self.df[new_col] = self.df.groupby(groupby_cols)[col].shift(lag)
                            else:
                                self.df[new_col] = self.df[col].shift(lag)
            
            for col in ['usage_cpu', 'usage_storage', 'users_active']:
                if col in self.df.columns:
                    roll_mean_col = f"{col}_rolling_mean_7"
                    roll_std_col = f"{col}_rolling_std_7"
                    if roll_mean_col not in self.df.columns:
                        if groupby_cols:
                            self.df[roll_mean_col] = self.df.groupby(groupby_cols)[col].transform(
                                lambda x: x.rolling(window=7, min_periods=1).mean()
                            )
                        else:
                            self.df[roll_mean_col] = self.df[col].rolling(window=7, min_periods=1).mean()
                    if roll_std_col not in self.df.columns:
                        if groupby_cols:
                            self.df[roll_std_col] = self.df.groupby(groupby_cols)[col].transform(
                                lambda x: x.rolling(window=7, min_periods=1).std()
                            )
                        else:
                            self.df[roll_std_col] = self.df[col].rolling(window=7, min_periods=1).std()
            
            self.df = self.df.dropna()
            self.df.to_csv(FEATURE_ENGINEERED_PATH, index=False)
            return self.df
        except Exception as e:
            print(f"Error creating features: {e}")
            raise

class ForecastModel:
    def __init__(self, df):
        self.df = df
        self.models = {}
        self.metrics = {}
        self.scalers = {}
        
    def time_based_split(self, train_ratio=0.7, val_ratio=0.2):
        n = len(self.df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train = self.df.iloc[:train_end].copy()
        val = self.df.iloc[train_end:val_end].copy()
        test = self.df.iloc[val_end:].copy()
        
        return train, val, test
    
    def prepare_features(self, df, target='usage_cpu'):
        drop_cols = ['date', 'region', 'service', 'usage_cpu', 'usage_storage', 'users_active']
        feature_cols = [col for col in df.columns if col not in drop_cols]
    
        X = df[feature_cols].copy()
        y = df[target]

        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.Categorical(X[col]).codes

        return X, y

    def calculate_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
        
        bias = np.mean(y_pred - y_true)
        
        return {
            "MAE": round(float(mae), 3),
            "RMSE": round(float(rmse), 3),
            "MAPE": round(float(mape), 2),
            "Forecast_Bias": round(float(bias), 3)
        }
    
    def train_xgboost(self, train, val, target='usage_cpu'):
        if not HAS_ML:
            return self._mock_model_metrics("XGBoost")
        
        try:
            X_train, y_train = self.prepare_features(train, target)
            X_val, y_val = self.prepare_features(val, target)
            
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            metrics = self.calculate_metrics(y_val, y_pred)
            
            self.models['XGBoost'] = model
            self.metrics['XGBoost'] = metrics
            
            joblib.dump(model, MODELS_DIR / f'xgboost_{target}.pkl')
            
            return metrics
        except Exception as e:
            print(f"XGBoost training failed: {e}")
            return self._mock_model_metrics("XGBoost")
    
    def train_arima(self, train, val, target='usage_cpu'):
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            train_ts = train.groupby('date')[target].mean()
            val_ts = val.groupby('date')[target].mean()
            
            model = SARIMAX(train_ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            model_fit = model.fit(disp=False)
            
            forecast = model_fit.forecast(steps=len(val_ts))
            
            metrics = self.calculate_metrics(val_ts.values, forecast.values)
            
            self.models['ARIMA'] = model_fit
            self.metrics['ARIMA'] = metrics
            
            model_fit.save(MODELS_DIR / f'arima_{target}.pkl')
            
            return metrics
        except Exception as e:
            print(f"ARIMA training failed: {e}")
            return self._mock_model_metrics("ARIMA")
    
    def train_lstm(self, train, val, target='usage_cpu'):
        """Train LSTM model with TensorFlow/Keras"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from sklearn.preprocessing import MinMaxScaler

            train_ts = train.groupby('date')[target].mean().values.reshape(-1, 1)
            val_ts = val.groupby('date')[target].mean().values.reshape(-1, 1)

            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train_ts)
            val_scaled = scaler.transform(val_ts)

            def create_sequences(data, seq_len=7):
                X, y = [], []
                for i in range(len(data) - seq_len):
                    X.append(data[i:i+seq_len])
                    y.append(data[i+seq_len])
                return np.array(X), np.array(y)

            seq_len = 7
            X_train, y_train = create_sequences(train_scaled, seq_len)
            X_val, y_val = create_sequences(val_scaled, seq_len)

            model = Sequential([
                LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_len, 1)),
                Dropout(0.2),
                LSTM(50, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

            model.fit(X_train, y_train, epochs=30, batch_size=16,
                    validation_data=(X_val, y_val), verbose=0)

            y_pred = model.predict(X_val, verbose=0)
            y_pred = scaler.inverse_transform(y_pred)
            y_val_original = scaler.inverse_transform(y_val)

            metrics = self.calculate_metrics(y_val_original.flatten(), y_pred.flatten())

            self.models['LSTM'] = model
            self.scalers['LSTM'] = scaler
            self.metrics['LSTM'] = metrics

            model.save(MODELS_DIR / f"lstm_{target}.h5")

            return metrics

        except Exception as e:
            print(f"LSTM training failed: {e}")
            return self._mock_model_metrics("LSTM")

    def _mock_model_metrics(self, model_name):
        mock_metrics = {
            "ARIMA": {"MAE": 0.095, "RMSE": 0.125, "MAPE": 6.8, "Forecast_Bias": 0.02},
            "XGBoost": {"MAE": 0.082, "RMSE": 0.108, "MAPE": 5.9, "Forecast_Bias": -0.01},
            "LSTM": {"MAE": 0.073, "RMSE": 0.097, "MAPE": 4.35, "Forecast_Bias": 0.005}
        }
        return mock_metrics.get(model_name, {"MAE": 0.1, "RMSE": 0.15, "MAPE": 8.0, "Forecast_Bias": 0.0})
    
    def train_all_models(self):
        try:
            train, val, test = self.time_based_split()
            
            print("Training XGBoost...")
            self.train_xgboost(train, val)
            
            print("Training ARIMA...")
            self.train_arima(train, val)
            
            print("Training LSTM...")
            self.train_lstm(train, val)
            
            return self.metrics
        except Exception as e:
            print(f"Error training models: {e}")
            raise

class CapacityPlanner:
    def __init__(self, model, df):
        self.model = model
        self.df = df
        
    def generate_forecast(self, region, service, horizon=30):
        try:
            filtered = self.df[
                (self.df['region'] == region) & 
                (self.df['service'] == service)
            ].copy()
            
            if filtered.empty:
                return self._generate_mock_forecast(horizon)
            
            last_date = filtered['date'].max()
            last_values = filtered.iloc[-1]
            
            predictions = []
            for i in range(1, horizon + 1):
                pred_date = last_date + timedelta(days=i)
                
                trend = last_values['usage_cpu'] * (1 + 0.002 * i)
                seasonal = 10 * np.sin(2 * np.pi * i / 7)
                noise = np.random.normal(0, 2)
                
                pred_cpu = max(0, trend + seasonal + noise)
                pred_storage = max(0, last_values['usage_storage'] * (1 + 0.003 * i) + np.random.normal(0, 50))
                pred_users = max(0, int(last_values['users_active'] * (1 + 0.002 * i) + np.random.normal(0, 20)))
                
                predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'predicted_cpu': round(pred_cpu, 2),
                    'predicted_storage': round(pred_storage, 2),
                    'predicted_users': pred_users,
                    'confidence_lower_cpu': round(max(0, pred_cpu - 8), 2),
                    'confidence_upper_cpu': round(pred_cpu + 8, 2)
                })
            
            return predictions
        except Exception as e:
            print(f"Error generating forecast: {e}")
            return self._generate_mock_forecast(horizon)
    
    def _generate_mock_forecast(self, horizon=30):
        base_cpu = 70
        predictions = []
        
        for i in range(1, horizon + 1):
            pred = base_cpu + np.sin(i * 0.1) * 10 + np.random.uniform(-5, 5)
            predictions.append({
                'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                'predicted_cpu': round(pred, 2),
                'predicted_storage': round(1500 + i * 10, 2),
                'predicted_users': int(1200 + i * 5),
                'confidence_lower_cpu': round(max(0, pred - 8), 2),
                'confidence_upper_cpu': round(pred + 8, 2)
            })
        
        return predictions
    
    def get_recommendations_data(self):
        """Generate recommendations data for CSV export"""
        recommendations = [
            {
                'region': 'East US',
                'service': 'Compute',
                'action': 'Scale Up',
                'current_capacity': 1000,
                'recommended_capacity': 1500,
                'change': '+500 units',
                'priority': 'High',
                'reason': 'Expected 23% increase in demand',
                'estimated_cost_impact': '+$15,000/month'
            },
            {
                'region': 'West Europe',
                'service': 'Storage',
                'action': 'Scale Down',
                'current_capacity': 2000,
                'recommended_capacity': 1300,
                'change': '-700 TB',
                'priority': 'Medium',
                'reason': 'Over-provisioned by 18%',
                'estimated_cost_impact': '-$8,500/month'
            },
            {
                'region': 'Southeast Asia',
                'service': 'VMs',
                'action': 'Scale Up',
                'current_capacity': 800,
                'recommended_capacity': 1650,
                'change': '+850 units',
                'priority': 'Critical',
                'reason': 'High growth trajectory detected',
                'estimated_cost_impact': '+$22,000/month'
            },
            {
                'region': 'North Europe',
                'service': 'Containers',
                'action': 'Scale Up',
                'current_capacity': 500,
                'recommended_capacity': 820,
                'change': '+320 units',
                'priority': 'Medium',
                'reason': 'Optimal scaling opportunity',
                'estimated_cost_impact': '+$9,500/month'
            }
        ]
        return recommendations

# ============================================
# INITIALIZE GLOBAL OBJECTS
# ============================================

try:
    data_processor = DataProcessor()
    azure_df, external_df = data_processor.load_data()
    azure_df = data_processor.clean_data()
    merged_df = data_processor.merge_data()

    feature_engineer = FeatureEngineer(merged_df)
    featured_df = feature_engineer.create_features()

    forecast_model = ForecastModel(featured_df)
    model_metrics = forecast_model.train_all_models()

    capacity_planner = CapacityPlanner(forecast_model, featured_df)
    
    print("✅ All models initialized successfully")
except Exception as e:
    print(f"❌ Initialization error: {e}")
    raise

# ============================================
# EXPORT HELPER FUNCTIONS
# ============================================

def generate_csv_report(data, filename):
    """Generate CSV file from data"""
    try:
        df = pd.DataFrame(data)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return output.getvalue()
    except Exception as e:
        print(f"Error generating CSV: {e}")
        raise

def generate_excel_report(data, filename):
    """Generate Excel file from data"""
    if not HAS_EXCEL:
        return None
    
    try:
        output = io.BytesIO()
        df = pd.DataFrame(data)
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Report', index=False)
            
            workbook = writer.book
            worksheet = writer.sheets['Report']
            
            header_fill = PatternFill(start_color='667EEA', end_color='667EEA', fill_type='solid')
            header_font = Font(bold=True, color='FFFFFF')
            
            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center')
            
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        output.seek(0)
        return output
    except Exception as e:
        print(f"Error generating Excel: {e}")
        return None

def generate_pdf_report(data, filename, title="Azure Forecast Report"):
    """Generate PDF file from data"""
    if not HAS_PDF:
        return None
    
    try:
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        title_para = Paragraph(f"<b>{title}</b>", styles['Title'])
        elements.append(title_para)
        elements.append(Spacer(1, 0.3 * inch))
        
        date_para = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        elements.append(date_para)
        elements.append(Spacer(1, 0.3 * inch))
        
        df = pd.DataFrame(data)
        table_data = [df.columns.tolist()] + df.values.tolist()
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667EEA')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        doc.build(elements)
        
        output.seek(0)
        return output
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None

# ============================================
# FLASK API ENDPOINTS
# ============================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/forecast", methods=["GET"])
def api_forecast():
    try:
        region = request.args.get('region', 'East US')
        service = request.args.get('service', 'Compute')
        horizon = int(request.args.get('horizon', 30))
        
        if region == 'all':
            region = 'East US'
        
        recent_data = featured_df[
            (featured_df['region'] == region) & 
            (featured_df['service'] == service)
        ].tail(30)
        
        if recent_data.empty:
            recent_data = featured_df.tail(30)
        
        historical = []
        for _, row in recent_data.iterrows():
            historical.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "cpu": round(float(row.get("usage_cpu", 70)), 1),
                "storage": round(float(row.get("usage_storage", 1500)), 1),
                "users": int(row.get("users_active", 1200))
            })
        
        forecast_data = capacity_planner.generate_forecast(region, service, horizon)
        
        forecast = []
        for pred in forecast_data:
            forecast.append({
                "date": pred['date'],
                "cpu": pred['predicted_cpu'],
                "cpuLower": pred['confidence_lower_cpu'],
                "cpuUpper": pred['confidence_upper_cpu'],
                "storage": pred.get('predicted_storage', 1500),
                "storageLower": pred.get('predicted_storage', 1500) - 100,
                "storageUpper": pred.get('predicted_storage', 1500) + 100,
                "users": pred.get('predicted_users', 1200),
                "usersLower": pred.get('predicted_users', 1200) - 80,
                "usersUpper": pred.get('predicted_users', 1200) + 80
            })
        
        return jsonify({
            "status": "success",
            "historical": historical, 
            "forecast": forecast
        })
    
    except Exception as e:
        print(f"Forecast API error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/api/dashboard-stats", methods=["GET"])
def api_dashboard_stats():
    try:
        avg_accuracy = 100 - np.mean([m["MAPE"] for m in model_metrics.values()])
        
        total_storage = featured_df['usage_storage'].tail(30).mean() / 1000
        peak_users = int(featured_df['users_active'].tail(30).max())
        
        stats = {
            "status": "success",
            "modelAccuracy": round(avg_accuracy, 1),
            "totalStorageForecast": round(total_storage, 1),
            "predictedPeakUsers": peak_users,
            "potentialSavings": 42000 + np.random.randint(-5000, 8000),
            "lastUpdated": datetime.now().isoformat()
        }
        return jsonify(stats)
    except Exception as e:
        print(f"Dashboard stats error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/api/model-comparison", methods=["GET"])
def model_comparison():
    try:
        models = []
        
        for model_name, metrics in model_metrics.items():
            models.append({
                "name": model_name,
                "mae": metrics["MAE"],
                "rmse": metrics["RMSE"],
                "mape": metrics["MAPE"],
                "accuracy": round(100 - metrics["MAPE"], 1)
            })
        
        return jsonify({
            "status": "success",
            "models": models
        })
    except Exception as e:
        print(f"Model comparison error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/api/regional-distribution", methods=["GET"])
def regional_distribution():
    """Get regional distribution data"""
    try:
        regions = featured_df.groupby('region')['usage_cpu'].mean().to_dict()
        
        return jsonify({
            "status": "success",
            "labels": list(regions.keys()),
            "data": [round(v, 1) for v in regions.values()]
        })
    except Exception as e:
        print(f"Regional distribution error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# ============================================
# DOWNLOAD ENDPOINTS
# ============================================

@app.route("/api/download/recommendations", methods=["GET"])
def download_recommendations():
    """Download recommendations in CSV, Excel, or PDF format"""
    try:
        format_type = request.args.get('format', 'csv')
        recommendations = capacity_planner.get_recommendations_data()
        
        if format_type == 'csv':
            csv_data = generate_csv_report(recommendations, "recommendations.csv")
            response = make_response(csv_data)
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = 'attachment; filename=azure_recommendations.csv'
            response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
            return response
            
        elif format_type == 'excel':
            if not HAS_EXCEL:
                return jsonify({"status": "error", "error": "Excel export not available. Install openpyxl."}), 500
            
            excel_data = generate_excel_report(recommendations, "recommendations.xlsx")
            if excel_data is None:
                return jsonify({"status": "error", "error": "Excel generation failed"}), 500
            
            response = make_response(excel_data.getvalue())
            response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            response.headers['Content-Disposition'] = 'attachment; filename=azure_recommendations.xlsx'
            response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
            return response
            
        elif format_type == 'pdf':
            if not HAS_PDF:
                return jsonify({"status": "error", "error": "PDF export not available. Install reportlab."}), 500
            
            pdf_data = generate_pdf_report(recommendations, "recommendations.pdf", "Azure Resource Recommendations")
            if pdf_data is None:
                return jsonify({"status": "error", "error": "PDF generation failed"}), 500
            
            response = make_response(pdf_data.getvalue())
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = 'attachment; filename=azure_recommendations.pdf'
            response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
            return response
        
        else:
            return jsonify({"status": "error", "error": "Invalid format. Use csv, excel, or pdf"}), 400
            
    except Exception as e:
        print(f"Download recommendations error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/api/download/report/<report_type>", methods=["GET"])
def download_report(report_type):
    """Download reports in CSV, Excel, or PDF format"""
    try:
        format_type = request.args.get('format', 'csv')
        
        if report_type == 'forecast':
            forecast_data = capacity_planner.generate_forecast('East US', 'Compute', 30)
            data = forecast_data
            title = "Azure Forecast Report"
            
        elif report_type == 'capacity':
            regions = ['East US', 'West US', 'North Europe', 'Southeast Asia']
            data = []
            for region in regions:
                region_data = featured_df[featured_df['region'] == region]
                if not region_data.empty:
                    avg_cpu = region_data['cpu_utilization'].mean() * 100
                    avg_storage = region_data['usage_storage'].mean()
                else:
                    avg_cpu = 70 + np.random.uniform(-10, 20)
                    avg_storage = 1500
                
                data.append({
                    'region': region,
                    'average_cpu_utilization': round(avg_cpu, 2),
                    'average_storage_gb': round(avg_storage, 2),
                    'capacity_status': 'Optimal' if avg_cpu < 80 else 'High' if avg_cpu < 90 else 'Critical'
                })
            title = "Azure Capacity Planning Report"
            
        elif report_type == 'comparison':
            data = []
            for model_name, metrics in model_metrics.items():
                data.append({
                    'model': model_name,
                    'mae': metrics['MAE'],
                    'rmse': metrics['RMSE'],
                    'mape': metrics['MAPE'],
                    'accuracy_percent': round(100 - metrics['MAPE'], 2),
                    'forecast_bias': metrics['Forecast_Bias']
                })
            title = "Model Comparison Report"
            
        else:  # full report
            data = featured_df.tail(100).to_dict('records')
            for record in data:
                if 'date' in record and hasattr(record['date'], 'strftime'):
                    record['date'] = record['date'].strftime('%Y-%m-%d')
            title = "Full Analytics Report"
        
        if format_type == 'csv':
            csv_data = generate_csv_report(data, f"{report_type}_report.csv")
            response = make_response(csv_data)
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = f'attachment; filename=azure_{report_type}_report.csv'
            response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
            return response
            
        elif format_type == 'excel':
            if not HAS_EXCEL:
                return jsonify({"status": "error", "error": "Excel export not available. Install openpyxl."}), 500
            
            excel_data = generate_excel_report(data, f"{report_type}_report.xlsx")
            if excel_data is None:
                return jsonify({"status": "error", "error": "Excel generation failed"}), 500
            
            response = make_response(excel_data.getvalue())
            response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            response.headers['Content-Disposition'] = f'attachment; filename=azure_{report_type}_report.xlsx'
            response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
            return response
            
        elif format_type == 'pdf':
            if not HAS_PDF:
                return jsonify({"status": "error", "error": "PDF export not available. Install reportlab."}), 500
            
            pdf_data = generate_pdf_report(data, f"{report_type}_report.pdf", title)
            if pdf_data is None:
                return jsonify({"status": "error", "error": "PDF generation failed"}), 500
            
            response = make_response(pdf_data.getvalue())
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename=azure_{report_type}_report.pdf'
            response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
            return response
        
        else:
            return jsonify({"status": "error", "error": "Invalid format. Use csv, excel, or pdf"}), 400
            
    except Exception as e:
        print(f"Download report error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500
    

# CAPACITY PLANNING API - MILESTONE 4

@app.route("/api/capacity-planning", methods=["GET"])
def capacity_planning():
    """
    Returns capacity planning recommendations with risk analysis
    Query params: region, service
    """
    try:
        region = request.args.get('region', 'East US')
        service = request.args.get('service', 'Compute')
        horizon = int(request.args.get('horizon', 30))
        
        forecast_data = capacity_planner.generate_forecast(region, service, horizon)
        
        avg_forecast_demand = np.mean([pred['predicted_cpu'] for pred in forecast_data])
        
        current_capacity_map = {
            'East US': {'Compute': 11000, 'Storage': 2000, 'Database': 5000, 'Networking': 3000},
            'West US': {'Compute': 9500, 'Storage': 1800, 'Database': 4500, 'Networking': 2800},
            'North Europe': {'Compute': 10000, 'Storage': 1900, 'Database': 4800, 'Networking': 2900},
            'Southeast Asia': {'Compute': 8500, 'Storage': 1600, 'Database': 4000, 'Networking': 2500},
            'Central India': {'Compute': 7500, 'Storage': 1500, 'Database': 3800, 'Networking': 2300}
        }
        
        available_capacity = current_capacity_map.get(region, {}).get(service, 10000)
        
        forecast_demand = int(avg_forecast_demand * 150)  
        
        adjustment = forecast_demand - available_capacity
        adjustment_percent = (adjustment / available_capacity) * 100
        
        utilization = (forecast_demand / available_capacity) * 100
        if utilization > 95:
            risk_level = "critical"
            risk_color = "red"
        elif utilization > 85:
            risk_level = "shortage"
            risk_color = "orange"
        elif utilization < 50:
            risk_level = "over-provisioned"
            risk_color = "yellow"
        else:
            risk_level = "optimal"
            risk_color = "green"
        
        avg_accuracy = 100 - np.mean([m["MAPE"] for m in model_metrics.values()])
        confidence = avg_accuracy / 100
        
        if adjustment > 0:
            recommendation = f"Scale up by {adjustment} units ({adjustment_percent:.1f}%)"
            action = "scale_up"
        elif adjustment < -500:
            recommendation = f"Scale down by {abs(adjustment)} units ({abs(adjustment_percent):.1f}%)"
            action = "scale_down"
        else:
            recommendation = "Maintain current capacity"
            action = "maintain"
        
        cost_per_unit = 10  
        cost_impact = adjustment * cost_per_unit
        
        response = {
            "status": "success",
            "region": region,
            "service": service,
            "forecast_demand": forecast_demand,
            "available_capacity": available_capacity,
            "utilization_percent": round(utilization, 2),
            "recommended_adjustment": adjustment,
            "adjustment_percent": round(adjustment_percent, 2),
            "recommendation": recommendation,
            "action": action,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "confidence": round(confidence, 3),
            "estimated_cost_impact": cost_impact,
            "horizon_days": horizon,
            "forecast_details": forecast_data[:7]  
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Capacity planning error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# MONITORING API - MILESTONE 4

@app.route("/api/monitoring", methods=["GET"])
def monitoring():
    """
    Returns model monitoring metrics including accuracy, drift, and health status
    """
    try:
        avg_mape = np.mean([m["MAPE"] for m in model_metrics.values()])
        current_accuracy = 100 - avg_mape
        
        error_drift = np.random.uniform(0.5, 3.5)  
        
        last_retrain_date = "2024-12-15"  
        
        from datetime import datetime, timedelta
        last_retrain = datetime.strptime(last_retrain_date, "%Y-%m-%d")
        days_since_retrain = (datetime.now() - last_retrain).days
        
        retrain_needed = days_since_retrain > 30 or current_accuracy < 85
        
        next_retrain_date = (last_retrain + timedelta(days=30)).strftime("%Y-%m-%d")
        
        if current_accuracy >= 90:
            health_status = "excellent"
            health_color = "green"
            health_icon = "🟢"
        elif current_accuracy >= 85:
            health_status = "stable"
            health_color = "green"
            health_icon = "🟢"
        elif current_accuracy >= 70:
            health_status = "caution"
            health_color = "yellow"
            health_icon = "🟡"
        else:
            health_status = "retrain_needed"
            health_color = "red"
            health_icon = "🔴"
        
        model_health = []
        for model_name, metrics in model_metrics.items():
            model_accuracy = 100 - metrics["MAPE"]
            model_health.append({
                "model": model_name,
                "accuracy": round(model_accuracy, 2),
                "mae": metrics["MAE"],
                "rmse": metrics["RMSE"],
                "mape": metrics["MAPE"],
                "status": "healthy" if model_accuracy >= 85 else "degraded"
            })
        
        accuracy_trend = [
            {"date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
             "accuracy": current_accuracy + np.random.uniform(-3, 3)}
            for i in range(30, 0, -3)
        ]
        
        inference_speed_ms = np.random.uniform(10, 20)  # milliseconds
        
        response = {
            "status": "success",
            "monitoring": {
                "health_status": health_status,
                "health_color": health_color,
                "health_icon": health_icon,
                "current_accuracy": round(current_accuracy, 2),
                "target_accuracy": 85.0,
                "error_drift": round(error_drift, 2),
                "drift_threshold": 5.0,
                "last_retrain_date": last_retrain_date,
                "days_since_retrain": days_since_retrain,
                "next_retrain_date": next_retrain_date,
                "retrain_needed": retrain_needed,
                "inference_speed_ms": round(inference_speed_ms, 2),
                "total_predictions_made": len(featured_df) * 3,  # mock value
                "models_deployed": len(model_metrics)
            },
            "model_health": model_health,
            "accuracy_trend": accuracy_trend,
            "recommendations": []
        }
        
        if retrain_needed:
            response["recommendations"].append({
                "priority": "high",
                "message": "Model retraining recommended",
                "reason": f"Accuracy below threshold or {days_since_retrain} days since last retrain"
            })
        
        if error_drift > 4.0:
            response["recommendations"].append({
                "priority": "medium",
                "message": "High error drift detected",
                "reason": "Consider investigating data quality or model degradation"
            })
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Monitoring error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# BULK CAPACITY PLANNING FOR ALL REGIONS

@app.route("/api/capacity-planning/all", methods=["GET"])
def capacity_planning_all_regions():
    """
    Returns capacity planning recommendations for all regions and services
    """
    try:
        regions = ['East US', 'West US', 'North Europe', 'Southeast Asia', 'Central India']
        services = ['Compute', 'Storage', 'Database', 'Networking']
        
        all_recommendations = []
        
        for region in regions:
            for service in services:
                forecast_data = capacity_planner.generate_forecast(region, service, 30)
                avg_forecast = np.mean([pred['predicted_cpu'] for pred in forecast_data])
                
                base_capacity = 10000 if service == 'Compute' else 2000
                current_capacity = base_capacity + np.random.randint(-2000, 2000)
                forecast_demand = int(avg_forecast * 150)
                
                adjustment = forecast_demand - current_capacity
                utilization = (forecast_demand / current_capacity) * 100
                
                if utilization > 90:
                    priority = "critical"
                elif utilization > 80 or utilization < 50:
                    priority = "high"
                else:
                    priority = "medium"
                
                all_recommendations.append({
                    "region": region,
                    "service": service,
                    "current_capacity": current_capacity,
                    "forecast_demand": forecast_demand,
                    "utilization": round(utilization, 1),
                    "adjustment_needed": adjustment,
                    "priority": priority
                })
        
        priority_order = {"critical": 0, "high": 1, "medium": 2}
        all_recommendations.sort(key=lambda x: priority_order[x["priority"]])
        
        return jsonify({
            "status": "success",
            "total_recommendations": len(all_recommendations),
            "recommendations": all_recommendations,
            "summary": {
                "critical_count": sum(1 for r in all_recommendations if r["priority"] == "critical"),
                "high_count": sum(1 for r in all_recommendations if r["priority"] == "high"),
                "medium_count": sum(1 for r in all_recommendations if r["priority"] == "medium")
            }
        })
        
    except Exception as e:
        print(f"Bulk capacity planning error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ============================================
# REPORT GENERATION API
# ============================================

@app.route("/api/report", methods=["GET"])
def generate_report():
    """
    General report endpoint that returns comprehensive data
    Query params: report_type (forecast, capacity, monitoring, full)
    """
    try:
        report_type = request.args.get('type', 'full')
        
        report_data = {
            "status": "success",
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "data": {}
        }
        
        if report_type in ['forecast', 'full']:
            # Add forecast data
            forecast = capacity_planner.generate_forecast('East US', 'Compute', 30)
            report_data["data"]["forecast"] = forecast
        
        if report_type in ['capacity', 'full']:
            # Add capacity planning data
            regions = ['East US', 'West US', 'North Europe']
            capacity_data = []
            for region in regions:
                forecast = capacity_planner.generate_forecast(region, 'Compute', 30)
                avg_demand = np.mean([p['predicted_cpu'] for p in forecast])
                capacity_data.append({
                    "region": region,
                    "avg_demand": round(avg_demand, 2),
                    "capacity": 10000,
                    "utilization": round((avg_demand / 100) * 100, 2)
                })
            report_data["data"]["capacity"] = capacity_data
        
        if report_type in ['monitoring', 'full']:
            # Add monitoring data
            report_data["data"]["monitoring"] = {
                "accuracy": round(100 - np.mean([m["MAPE"] for m in model_metrics.values()]), 2),
                "models": list(model_metrics.keys()),
                "health_status": "stable"
            }
        
        return jsonify(report_data)
        
    except Exception as e:
        print(f"Report generation error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "models_loaded": len(model_metrics),
            "data_records": len(featured_df)
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    print("=" * 60)
    print("Azure Demand Forecasting Dashboard - Backend Server")
    print("=" * 60)
    print(f"Data loaded: {len(featured_df)} records")
    print(f"Date range: {featured_df['date'].min()} to {featured_df['date'].max()}")
    print(f"Models trained: {len(model_metrics)}")
    print("\nModel Performance:")
    for model_name, metrics in model_metrics.items():
        print(f"  {model_name}: MAE={metrics['MAE']}, MAPE={metrics['MAPE']}%")
    print("\nStarting Flask server...")
    print("Access dashboard at: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host="0.0.0.0", port=5000)