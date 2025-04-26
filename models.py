import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Models:
    @staticmethod
    def XGBoost(X_train, y_train, X_test, y_test, *args):
        """Train and evaluate XGBoost model"""
        model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Evaluate
        metrics = Models.evaluate_model(y_test, predictions, "XGBoost")
        return metrics

    @staticmethod
    def AR(X_train, y_train, X_test, y_test, *args):
        """Train and evaluate Autoregressive model"""
        model = AutoReg(y_train, lags=10).fit()
        predictions = model.predict(start=len(y_train), end=len(y_train)+len(y_test)-1)
        
        # Evaluate
        metrics = Models.evaluate_model(y_test, predictions, "AR")
        return metrics

    @staticmethod
    def ARIMA(X_train, y_train, X_test, y_test, *args):
        """Train and evaluate ARIMA model"""
        model = ARIMA(y_train, order=(5, 0, 1)).fit()
        predictions = model.forecast(steps=len(y_test))
        
        # Evaluate
        metrics = Models.evaluate_model(y_test, predictions, "ARIMA")
        return metrics

    @staticmethod
    def SARIMAX(X_train, y_train, X_test, y_test, *args):
        """Train and evaluate SARIMAX model"""
        model = SARIMAX(y_train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0)).fit(disp=False)
        predictions = model.forecast(steps=len(y_test))
        
        # Evaluate
        metrics = Models.evaluate_model(y_test, predictions, "SARIMAX")
        return metrics

    @staticmethod
    def Prophet(X_train, y_train, X_test, y_test, date_col_train, date_col_test):
        """Train and evaluate Prophet model"""
        prophet_train = pd.DataFrame({'ds': date_col_train, 'y': y_train})
        prophet_model = Prophet(daily_seasonality=True)
        prophet_model.fit(prophet_train)
        
        future = pd.DataFrame({'ds': date_col_test})
        forecast = prophet_model.predict(future)
        predictions = forecast['yhat'].values
        
        # Evaluate
        metrics = Models.evaluate_model(y_test, predictions, "Prophet")
        return metrics

    @staticmethod
    def NeuralProphet(X_train, y_train, X_test, y_test, date_col_train, date_col_test):
        """Train and evaluate NeuralProphet model"""
        try:
            np_train = pd.DataFrame({'ds': date_col_train, 'y': y_train})
            np_model = NeuralProphet()
            np_model.fit(np_train)
            
            future = pd.DataFrame({'ds': date_col_test, 'y': [np.nan] * len(date_col_test)})
            forecast = np_model.predict(future)
            predictions = forecast['yhat1'].values
            
            # Evaluate
            metrics = Models.evaluate_model(y_test, predictions, "NeuralProphet")
            return metrics
        except Exception as e:
            return None

    @staticmethod
    def evaluate_model(y_test, predictions, model_name):
        """Evaluate the model and return metrics"""
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        return {"model": model_name, "MAE": mae, "RMSE": rmse, "R^2": r2}
