from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import numpy as np
import pandas as pd
from preprocess import Preprocess, Config

class Models:
    @staticmethod
    def XGBoost(X_train, y_train, X_test, y_test, *args):
        """Train and evaluate XGBoost model"""
        model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = Models.evaluate_model(y_test, predictions, "XGBoost")
        return metrics

    @staticmethod
    def AR(X_train, y_train, X_test, y_test, *args):
        """Train and evaluate Autoregressive model"""
        model = AutoReg(y_train, lags=10).fit()
        predictions = model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
        metrics = Models.evaluate_model(y_test, predictions, "AR")
        return metrics

    @staticmethod
    def ARIMA(X_train, y_train, X_test, y_test, *args):
        """Train and evaluate ARIMA model"""
        model = ARIMA(y_train, order=(5, 0, 1)).fit()
        predictions = model.forecast(steps=len(y_test))
        metrics = Models.evaluate_model(y_test, predictions, "ARIMA")
        return metrics

    @staticmethod
    def SARIMAX(X_train, y_train, X_test, y_test, *args):
        """Train and evaluate SARIMAX model"""
        model = SARIMAX(y_train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0)).fit(disp=False)
        predictions = model.forecast(steps=len(y_test))
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
        metrics = Models.evaluate_model(y_test, predictions, "Prophet")
        return metrics

    @staticmethod
    def evaluate_model(y_test, predictions, model_name):
        """Evaluate the model and return metrics"""
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        return {
            "model": model_name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R^2": round(r2, 4)
        }

class AutoGluonModel:
    @staticmethod
    def train_and_evaluate(train_data, test_data, target_column, prediction_length):
        # Проверка формата данных
        if not isinstance(train_data, TimeSeriesDataFrame):
            train_data = TimeSeriesDataFrame.from_data_frame(
                train_data,
                id_column="item_id",
                timestamp_column="timestamp"
            )
        if not isinstance(test_data, TimeSeriesDataFrame):
            test_data = TimeSeriesDataFrame.from_data_frame(
                test_data,
                id_column="item_id",
                timestamp_column="timestamp"
            )

        # Проверка длины test_data
        if len(test_data) <= prediction_length:
            raise ValueError(
                f"Тестовая выборка слишком короткая: {len(test_data)} <= {prediction_length}. "
                f"Убедитесь, что test_data содержит исторические данные + {prediction_length} шагов."
            )

        predictor = TimeSeriesPredictor(
            prediction_length=prediction_length,
            target=target_column,
            freq='T',  # Минутная частота
            path="autogluon_model",
            eval_metric="MASE",
            verbosity=3
        )
        predictor.fit(
            train_data=train_data,
            presets="medium_quality",
            time_limit=60  # 1 минута (увеличьте для больших датасетов)
        )

        # Получаем предсказания
        predictions = predictor.predict(train_data)  # Предсказываем на train_data для валидации

        # Извлекаем истинные значения из test_data для горизонта прогнозирования
        # Берем последние prediction_length значений для каждого item_id
        y_true = []
        y_pred = []
        for item_id in test_data.item_ids:
            test_series = test_data.loc[item_id]
            if len(test_series) >= prediction_length:
                y_true.extend(test_series[target_column].values[-prediction_length:])
                # Предсказания для последнего горизонта
                pred_series = predictions.loc[item_id]
                y_pred.extend(pred_series["mean"].values[-prediction_length:])
            else:
                print(f"Пропущен item_id {item_id}: недостаточно данных.")

        # Вычисляем метрики
        if not y_true or not y_pred:
            raise ValueError("Не удалось извлечь данные для оценки метрик.")

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Получаем лучшую модель
        leaderboard = predictor.leaderboard(test_data, silent=True)

        return {
            "model": "AutoGluon",
            "best_model": leaderboard.iloc[0]["model"] if not leaderboard.empty else "Unknown",
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R^2": round(r2, 4),
            "future_forecast": predictions
        }