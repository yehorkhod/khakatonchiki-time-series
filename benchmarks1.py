import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame
from models import AutoGluonModel
from preprocess import Preprocess, Config

def prepare_data(data, target_column):
    df = data[[target_column]].copy()
    try:
        df['timestamp'] = pd.to_datetime(data['Date'])
    except Exception as e:
        print(f"Ошибка преобразования Date: {e}")
        df['timestamp'] = pd.date_range(start='2012-01-01', periods=len(df), freq='T')

    time_diffs = df['timestamp'].diff().dropna()
    if not (time_diffs == pd.Timedelta(minutes=1)).all():
        print("Обнаружены нерегулярные временные метки. Преобразуем...")
        df = df.set_index('timestamp').resample('T').interpolate(method='linear').reset_index()

    df['item_id'] = 'main_series'
    df[target_column] = df[target_column].interpolate(method='linear')

    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp"
    )
    return ts_df

def main():
    try:
        data = pd.read_parquet("data.parquet")
        print(f"Загружено {len(data)} записей")
        if data.empty:
            raise ValueError("Данные не загружены")

        target_column = "Series1"
        if target_column not in data.columns:
            raise ValueError(f"Целевая колонка {target_column} не найдена")

        processed_data = prepare_data(data, target_column)
        prediction_length = 240  # 4 часа = 240 минут
        split_idx = int(len(processed_data) * 0.8)

        train_data = processed_data.iloc[:split_idx]
        test_data = processed_data.iloc[split_idx - 2 * prediction_length:]

        print("\nРазмеры данных:")
        print(f"Обучающая выборка: {len(train_data)} записей")
        print(f"Тестовая выборка: {len(test_data)} записей")

        if len(test_data) <= prediction_length:
            raise ValueError(f"Тестовая выборка слишком короткая: {len(test_data)} <= {prediction_length}")

        results = AutoGluonModel.train_and_evaluate(
            train_data=train_data,
            test_data=test_data,
            target_column=target_column,
            prediction_length=prediction_length
        )

        print("\nРезультаты:")
        print(f"Модель: {results['model']}")
        print(f"Лучшая модель: {results['best_model']}")
        print(f"MAE: {results['MAE']:.4f}")
        print(f"RMSE: {results['RMSE']:.4f}")
        print(f"R^2: {results['R^2']:.4f}")
        print("\nПрогноз на 4 часа:")
        print(results['future_forecast'])

        # Сохранение прогноза в CSV
        forecast_df = results['future_forecast'].reset_index()  # Преобразуем в pandas.DataFrame
        forecast_df.to_csv("forecast_4_hours.csv", index=False)
        print("\nПрогноз сохранен в 'forecast_4_hours.csv'")

        # Сохранение метрик в CSV
        metrics_df = pd.DataFrame({
            "Model": [results['model']],
            "Best_Model": [results['best_model']],
            "MAE": [results['MAE']],
            "RMSE": [results['RMSE']],
            "R^2": [results['R^2']]
        })
        metrics_df.to_csv("metrics.csv", index=False)
        print("Метрики сохранены в 'metrics.csv'")

    except Exception as e:
        print(f"\nКритическая ошибка: {str(e)}")
        if 'data' in locals():
            print("\nПервые 5 строк данных:")
            print(data.head())
            print("\nИнформация о данных:")
            print(data.info())

if __name__ == "__main__":
    main()