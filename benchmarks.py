import pandas as pd
from preprocess import Preprocess, Config
from models import Models
import warnings
warnings.filterwarnings('ignore')

def main():
    # Read the data
    df = pd.read_parquet("data.parquet")
    
    # Config
    config = Config(
        NaNs="Linear interpolation",
        Smoothing="Rolling median",
        Lags=False,
        Decays=False,
        PCA=False,
        test_size=0.2
    )
    
    # Preprocess
    train, test = Preprocess(df, config).preprocess()
    # target_col: str = "Series1"
    
    for target_col in ["Series1", "Series2", "Series3", "Series4", "Series5", "Series6"]:
        print(f"Target column: {target_col}")
        # Identify feature columns - all numeric columns except the target and date
        date_col = df.columns[0]
        feature_cols = [col for col in train.columns if col != target_col and col != date_col]
        
        # Prepare data for models
        X_train = train[feature_cols]
        y_train = train[target_col]
        X_test = test[feature_cols]
        y_test = test[target_col]
        date_col_train = train[date_col]
        date_col_test = test[date_col]

        models: list = [
            Models.XGBoost,
            Models.AR,
            Models.ARIMA,
            Models.SARIMAX,
            Models.Prophet,
            Models.NeuralProphet
        ]

        metrics = [model(X_train, y_train, X_test, y_test, date_col_train, date_col_test) for model in models]
        print(metrics)

if __name__ == "__main__":
    main()
