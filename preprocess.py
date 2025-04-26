from typing import Literal, TypedDict
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
lowess = sm.nonparametric.lowess

class Config(TypedDict):
    NaNs: Literal["Linear interpolation", "Forward fill", "Time-aware interpolation", "Drop"]
    Smoothing: Literal["Rolling median", "Rolling mean", "Lowess smoothing", "Skip"]
    Lags: bool 
    Decays: bool
    PCA: bool
    test_size: float

class Preprocess:
    columns: list[str]
    data: pd.DataFrame
    config: Config
    lags_list: list[int] = [1, 2, 5, 10, 30, 60, 120, 1440]  # minutes
    decays_list: list[int] = [5, 60, 360]  # minutes

    def __init__(self, data: pd.DataFrame, config: Config) -> None:
        self.data = data 
        self.columns = data.columns.tolist() # Date, Series1, Series2, Series3, Series4, Series5, Series6 (Date, is per minute feature)
        self.config = config

    def preprocess(self):
        self.nans()

        if self.config["PCA"]:
            self.pca()  # PCA on Series2, Series3, Series4

        self.smoothing()

        if self.config["Lags"]:
            self.lags()

        if self.config["Decays"]:
            self.decays()

        return self.split()

    def nans(self) -> None:
        date_col = self.columns[0]

        # Проверяем, является ли колонка датой/временем
        is_datetime = pd.api.types.is_datetime64_any_dtype(self.data[date_col])

        match self.config["NaNs"]:
            case "Linear interpolation":
                self.data.interpolate(method='linear', inplace=True)

            case "Forward fill":
                self.data.fillna(method='ffill', inplace=True)

            case "Time-aware interpolation":
                if not is_datetime:
                    # Если колонка не datetime, преобразуем её
                    try:
                        self.data[date_col] = pd.to_datetime(self.data[date_col])
                    except:
                        # Если не получается преобразовать, используем линейную интерполяцию
                        self.data.interpolate(method='linear', inplace=True)
                        return

                # Теперь безопасно использовать time-based интерполяцию
                self.data = self.data.set_index(date_col).interpolate(method='time').reset_index()

            case "Drop":
                self.data.dropna(inplace=True)

            case _:
                # По умолчанию используем линейную интерполяцию
                self.data.interpolate(method='linear', inplace=True)

    def pca(self) -> None:
        pca_cols = ['Series2', 'Series3', 'Series4']
        
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(self.data[pca_cols])
        
        self.data = self.data.drop(columns=pca_cols)
        
        # Add PCA component as a new column
        self.data['Series_PCA'] = pca_result
        
        # Update columns list
        self.columns = self.data.columns.tolist()

    def smoothing(self) -> None:
        # Identify numeric columns (exclude Date column)
        numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        
        match self.config["Smoothing"]:
            case "Rolling median":
                window_size = 3  # Can be parameterized
                for col in numeric_cols:
                    self.data[col] = self.data[col].rolling(window=window_size, center=True).median()
                    # Handle boundary NaNs with forward and backward fill
                    self.data[col] = self.data[col].fillna(method='ffill').fillna(method='bfill')
            
            case "Rolling mean":
                window_size = 3  # Can be parameterized
                for col in numeric_cols:
                    self.data[col] = self.data[col].rolling(window=window_size, center=True).mean()
                    # Handle boundary NaNs with forward and backward fill
                    self.data[col] = self.data[col].fillna(method='ffill').fillna(method='bfill')
            
            case "Lowess smoothing":
                frac = 0.3  # Fraction of data used for each local regression
                for col in numeric_cols:
                    y = self.data[col].values
                    x = np.arange(len(y))
                    smoothed = lowess(y, x, frac=frac)
                    self.data[col] = smoothed[:, 1]
            
            case "Skip":
                pass  # Skip smoothing

    def lags(self) -> None:
        """
        Add lagged features for each numeric column using the specified time lags.
        Each lag represents a time shift backward in minutes.
        """
        # Identify the date column and numeric columns
        date_col = self.columns[0]
        columns = self.data.columns[self.data.columns.str.startswith("Series")].tolist()
        
        # Ensure date column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
            self.data[date_col] = pd.to_datetime(self.data[date_col])
        
        # Sort by date to ensure proper lag calculation
        self.data = self.data.sort_values(by=date_col)
        
        # Create lag features for each numeric column
        for col in columns:
            for lag in self.lags_list:
                # Create a new column with the lagged values
                lag_col_name = f"{col}_lag_{lag}"
                # Using shift for lag calculation
                self.data[lag_col_name] = self.data[col].shift(lag)
        
        # Handle NaNs created by lag operations
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        
        # Update columns list
        self.columns = self.data.columns.tolist()

    def decays(self) -> None:
        """
        Add exponential moving averages (EMA) for each numeric column
        using the specified decay periods.
        """
        # Identify numeric columns
        columns = self.data.columns[self.data.columns.str.startswith("Series")].tolist()
        
        # Calculate EMA for each span (decay period)
        for col in columns:
            for span in self.decays_list:
                # Create EMA column
                ema_col_name = f"{col}_ema_{span}"
                # Calculate EMA using pandas
                self.data[ema_col_name] = self.data[col].ewm(span=span).mean()
        
        # Update columns list
        self.columns = self.data.columns.tolist()

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing sets based on the configured test_size.
        
        Returns:
            A tuple containing (train_data, test_data)
        """
        date_col = self.columns[0]
        
        # Make a copy to avoid modifying the original data
        data_copy = self.data.copy()
        
        # If date column exists, use it for chronological splitting
        if date_col in data_copy.columns:
            # Sort by date
            data_copy = data_copy.sort_values(by=date_col)
            
            # Calculate split point
            split_idx = int(len(data_copy) * (1 - self.config["test_size"]))
            
            # Split data
            train_data = data_copy.iloc[:split_idx].copy()
            test_data = data_copy.iloc[split_idx:].copy()
        else:
            # Fallback to random split if no date column
            train_data, test_data = train_test_split(
                data_copy, 
                test_size=self.config["test_size"], 
                random_state=42
            )
        
        return train_data, test_data
