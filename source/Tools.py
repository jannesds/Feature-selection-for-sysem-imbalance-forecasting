import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import h5py
import random
from pprint import pprint
import os
import matplotlib.pyplot as plt
from scipy.stats import t


def setup_device_and_seed(seed=987):
    """
    Sets up the random seed for reproducibility and configures the computing device (CPU/GPU).

    Parameters:
        seed (int): Random seed value. Default is 987.

    Returns:
        dict: A dictionary containing device information.
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # For GPU-based operations

    # Setup CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_index = torch.cuda.current_device() if torch.cuda.is_available() else None
    gpu_name = torch.cuda.get_device_name(device_index) if device_index is not None else "No GPU"

    print(f"Using device: {device}")
    print(f"Current device index: {device_index}, GPU name: {gpu_name}")

    return {"device": device, "device_index": device_index, "gpu_name": gpu_name}

class ResultsAnalyzer:
    def __init__(self, model, X_train, y_train, X_test, y_test, X_val, y_val, feature_names, device, date_index):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.feature_names = feature_names
        self.device = device
        self.date_index = date_index

    def plot_training_loss(self, loss_values, val_loss_values):
        # Convert loss_values to a pandas Series
        loss_series = pd.Series(loss_values, name="Training Loss")
        val_loss_series = pd.Series(val_loss_values, name="Validation Loss")
        print(val_loss_series)

        # Apply rolling mean (3-period window, change as necessary)
        smoothed_loss = loss_series.rolling(window=3, min_periods=1, center=True).mean()
        smoothed_val_loss = val_loss_series.rolling(window=3, min_periods=1, center=True).mean()

        plt.figure(figsize=(10, 6))
        plt.plot(smoothed_loss, label='Training Loss')
        plt.plot(smoothed_val_loss, label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()

    def plot_backtest_results(self, backtest_df):
        """
        Plot the aggregated backtest predictions vs. actual values.
        The backtest_df should have a DateTime index and columns 'y_test' and 'y_pred'.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(backtest_df.index, backtest_df['y_test'], label="Actual")
        plt.plot(backtest_df.index, backtest_df['y_pred'], label="Predicted", linestyle='--')
        plt.xlabel("Date")
        plt.ylabel("Target")
        plt.title("Backtest: Predicted vs. Actual")
        plt.legend()
        plt.show()

    def plot_predictions(self):
        # Generate predictions for the test set
        y_pred_test = self.model(torch.FloatTensor(self.X_test.values).to(self.device)).cpu().detach().numpy().flatten()
        # Predict validation set values
        y_pred_val = self.model(torch.FloatTensor(self.X_val.values).to(self.device)).cpu().detach().numpy().flatten()

        # Print predictions for verification
        print("Predicted values (test set):", y_pred_test)
        print("Predicted values (validation set):", y_pred_val)

        # Ensure `self.date_index` is properly formatted as a pandas datetime index
        self.date_index = pd.to_datetime(self.date_index, unit='s')

        # Calculate split indices for training, validation, and test sets
        train_index = len(self.X_train)
        val_index = train_index + len(self.X_val)
        full_index = np.arange(len(self.X_train) + len(self.X_val) +len(self.X_test))

        # Plot actual values for training and test data with a timeline
        plt.figure(figsize=(14, 7))
        plt.plot(self.date_index[:train_index], self.y_train, label="Training Data (Actual)", color="blue")
        plt.plot(self.date_index[train_index:val_index], self.y_val, label="Validation Data (Actual)", color="orange")
        plt.plot(self.date_index[val_index:],self.y_test, label="Test Data (Actual)", color="lightblue")
        plt.plot(self.date_index[train_index:val_index],y_pred_val, label="Predicted Values (Validation)", color="green")
        plt.plot(self.date_index[val_index:], y_pred_test, label="Predicted Values (Test)", color="red", linestyle='--',
                 alpha=0.7)

        # Label the plot
        plt.xlabel('Time')
        plt.ylabel('System Imbalance (SI)')
        plt.title('Predicted vs. Actual System Imbalance Over Time (Point-Wise calculation)')
        plt.legend()
        plt.show()

    def evaluate_metrics(self):
        y_pred = self.model(torch.FloatTensor(self.X_test.values).to(self.device)).cpu().detach().numpy().flatten()
        y_test_tensor = torch.FloatTensor(self.y_test.values)
        y_pred_tensor = torch.FloatTensor(y_pred)  # Convert y_pred to a tensor

        # Mean Absolute Error (MAE)
        test_loss_mae = torch.mean(torch.abs(y_pred_tensor - y_test_tensor))
        print(f'Mean Absolute Error (MAE): {test_loss_mae.item():.4f}')

        # Root Mean Squared Error (RMSE)
        test_loss_rmse = torch.sqrt(torch.mean((y_pred_tensor - y_test_tensor) ** 2))
        print(f'Root Mean Squared Error (RMSE): {test_loss_rmse.item():.4f}')

        # R-Squared (R²)
        ss_total = torch.sum((y_test_tensor - torch.mean(y_test_tensor)) ** 2)
        ss_res = torch.sum((y_test_tensor - y_pred_tensor) ** 2)
        r2_score = 1 - ss_res / ss_total
        print(f'R-Squared (R²): {r2_score.item():.4f}')

        # Mean Absolute Percentage Error (MAPE)
        test_loss_mape = torch.mean(torch.abs((y_pred_tensor - y_test_tensor) / y_test_tensor)) * 100
        print(f'Mean Absolute Percentage Error (MAPE): {test_loss_mape.item():.2f}%')

        # Mean Absolute Scaled Error (MASE)
        naive_forecast_error = torch.mean(
            torch.abs(y_test_tensor[1:] - y_test_tensor[:-1]))  # Assuming time-series data
        mase_numerator = torch.mean(torch.abs(y_pred_tensor - y_test_tensor))
        mase = mase_numerator / naive_forecast_error

        print(f'Mean Absolute Scaled Error (MASE): {mase.item():.2f}')
        return     test_loss_mae.item(), test_loss_rmse.item(), r2_score.item(), test_loss_mape.item()
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def compute_permutation_importance(self, metric_fn=mean_absolute_error, n_repeats=30):
        # Base score with original (non-permuted) test set
        y_pred_test = self.model(torch.FloatTensor(self.X_test.values).to(self.device)).cpu().detach().numpy().flatten()
        base_score = metric_fn(self.y_test, y_pred_test)

        importances = np.zeros(self.X_test.shape[1])

        # Calculate permutation importance for each feature
        for col in range(self.X_test.shape[1]):
            score_diffs = []
            for _ in range(n_repeats):
                X_permuted = self.X_test.values.copy()
                np.random.shuffle(X_permuted[:, col])
                y_pred_permuted = self.model(
                    torch.FloatTensor(X_permuted).to(self.device)).cpu().detach().numpy().flatten()
                score = metric_fn(self.y_test, y_pred_permuted)
                score_diffs.append(score - base_score)
            importances[col] = np.mean(score_diffs)

        return pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

       # Function to load data from the HDF5 file
    def load_data_from_h5(h5_file_path):
        # Open the HDF5 file
        with h5py.File(h5_file_path, 'r') as file:
            # Read the data (assumes data is stored in datasets with columns)
            X_data = {key: file[key][...] for key in file.keys()}

            # Convert to DataFrame (assuming 'index' is the index and other keys are features)
            X_df = pd.DataFrame(X_data)

        return X_df

# ----------------------------------------------------------------------------------------------------------------------


def DM_test(
    y_test: pd.Series,
    y_pred_1: pd.Series,
    y_pred_2: pd.Series,
    h: int = 1,
    harvey_adj: bool = True,
):
    """
    -> Performs the Diebold-Mariano test to check for statistical significance in difference between two forecasts.

    Arguments:
        y_test -> real forecasted time-series
        y_pred_1 -> first forecast to compare
        y_pred_2 -> second forecast to compare
        h -> forecast horizon
        harvey_adj -> Harvey adjustment (leave as True)
    """

    e1_lst = []
    e2_lst = []
    d_lst = []

    y_test = y_test.tolist()
    y_pred_1 = y_pred_1.tolist()
    y_pred_2 = y_pred_2.tolist()

    # Length of forecasts
    T = float(len(y_test))

    # Construct loss differential according to error criterion (MSE)
    for real, p1, p2 in zip(y_test, y_pred_1, y_pred_2):
        e1_lst.append((real - p1) ** 2)
        e2_lst.append((real - p2) ** 2)
    for e1, e2 in zip(e1_lst, e2_lst):
        d_lst.append(e1 - e2)

    # Mean of loss differential
    mean_d = pd.Series(d_lst).mean()

    # Calculate autocovariance
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N - k):
            autoCov += ((Xi[i + k]) - Xs) * (Xi[i] - Xs)
        return (1 / (T)) * autoCov

    # Calculate the denominator of DM stat
    gamma = []
    for lag in range(0, h):
        gamma.append(autocovariance(d_lst, len(d_lst), lag, mean_d))  # 0, 1, 2
    V_d = (gamma[0] + 2 * sum(gamma[1:])) / T

    # Calculate DM stat
    DM_stat = V_d ** (-0.5) * mean_d

    # Calculate and apply Harvey adjustement
    # It applies a correction for small sample
    if harvey_adj is True:
        harvey_adj = ((T + 1 - 2 * h + h * (h - 1) / T) / T) ** (0.5)
        DM_stat = harvey_adj * DM_stat

    # Calculate p-value
    p_value = 2 * t.cdf(-abs(DM_stat), df=T - 1)

    print(f"DM Statistic: {DM_stat :.4f} | p-value: {p_value :.4f}")

@staticmethod
def add_lagged_features(df, lag_features, lags=0, leads=0):
    """
    Add lagged and lead features to the dataset.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        lag_features (list): List of feature names to lag or lead.
        lags (int): Number of lagged time steps.
        leads (int): Number of lead time steps.

    Returns:
        pd.DataFrame: DataFrame with lagged and lead features.
    """
    for feature in lag_features:
        # Add lagged features
        if lags > 0:
            for lag in range(1, lags + 1):
                df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
        # Add lead features
        if leads > 0:
            for lead in range(1, leads + 1):
                df[f'{feature}_lead{lead}'] = df[feature].shift(-lead)
    return df

def generate_lagged_features(data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    -> Takes in original dataframe, selects usable input features according to data availability, and includes the chosen lags accordingly.

    Arguments:
        data -> input pd.DataFrame
        parameters -> dictionary containing lags and real-time availability for each desired input feature
        minute -> minute at which predictions are made

    Returns:
        pd.DataFrame with the desired
    """
    # Only use columns that are in parameters dictionary:
    data = data[parameters.keys()].copy()
    output_df = []

    # Start by figuring out whether data is for a single minute in the qh or for all minutes:
    guess_index = np.random.randint(len(data) - 1)
    flag_minute_data = (
        abs(data.index[-guess_index].minute - data.index[-guess_index - 1].minute) < 15
    )
    granularity = "minute" if flag_minute_data else "qh"

    if data.index.freq is None:
        print("WARNING: Frequency for this dataframe is undefined!")
        print(f"Frequency identified as: {granularity}")

    for column in data:
        for lag in parameters[column]["lags"]:
            # Set a descriptive name for this column
            if lag < 0:
                column_name = f"{column}_from_{granularity}_minus_{abs(lag)}"
            elif lag == 0:
                column_name = f"{column}_current_{granularity}"
            elif lag > 0:
                column_name = f"{column}_from_{granularity}_plus_{abs(lag)}"

            output_df.append(data[column].shift(-lag).rename(column_name))

    return pd.concat(output_df, axis="columns")

def create_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    -> Generate datetime features from index
    """
    df = df.copy().assign(
        minute=df.index.minute.values.astype(np.int16),
        hour=df.index.hour.values.astype(np.int16),
        dayofmonth=df.index.day.values.astype(np.int16),
        month=df.index.month.values.astype(np.int16),
        year=df.index.year.values.astype(np.int16),
        # dayofweek_name = df.index.day_name(),
        dayofweek=df.index.dayofweek.values.astype(np.int16),
        dayofyear=df.index.dayofyear.values.astype(np.int16),
        weekofyear=df.index.isocalendar().week.values.astype(np.int16),
    )
    return df

def check_datetime_index(df: pd.DataFrame, freq: str | None = None):
    """
    -> Checks whether a *quarter-hourly* datetime index has missing entries
    """
    if freq is None:
        freq = "15min"
    missing = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq=freq, tz=df.index.tz
    ).difference(df.index)

    if missing.empty:
        print("No missing entries.")
    else:
        print(missing)

# ----------------------------------------------------------------------------------------------------------------------


# # System level utils:
# def get_root_dir() -> pathlib.Path:
#     """
#     -> Returns project root directory, regardless of where it's called from.
#     """
#     return pathlib.Path(__file__).resolve().parents[1]
#
#
# def show_python_env():
#     print(f"Python {python_version()}")
#     print(f"On path:\n{sys.path}")
#

# ----------------------------------------------------------------------------------------------------------------------


# def main():
#     print(f"Root directory: {get_root_dir()}")
#     show_python_env()


