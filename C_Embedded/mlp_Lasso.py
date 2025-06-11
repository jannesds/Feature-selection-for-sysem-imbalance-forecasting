import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import os,sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

from source.Tools import setup_device_and_seed, ResultsAnalyzer
from source.Cross_Validation import backtesting_CV, visualize_timeseries_splits, get_time_series_splits
from source.features_selection import get_feature_lag_config
from source.join_qh_min_data import join_qh_min_data
from B_wrappers.SBS import run_sbs
from comparison_elia import compare_with_elia_preds
import multiprocessing

print("Available CPUs:", multiprocessing.cpu_count())
from time import perf_counter
overall_start = perf_counter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")

def load_and_preprocess_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h5_dir = os.path.join(project_root, "h5")  # or os.path.join(os.environ["VSC_DATA"], "h5")
    h5_files = [
        r"quarter_hours_data_2022.01.01_to_2024.01.01.h5",
        r"minutes_data_2022.01.01_to_2024.01.01.h5",
        r"hours_data_2022.01.01_to_2024.01.01.h5"
    ]
    dataframes = {}
    print("Loading and preprocessing HDF5 files...")

    """Loads data from an HDF5 file, drops NaNs, and filters relevant years."""
    for h5_filename in h5_files:
        file_path = os.path.join(h5_dir, h5_filename)
        print("Loading data from:", file_path)

        df = ResultsAnalyzer.load_data_from_h5(file_path)

        # Identify & drop rows with NaN values
        dropped_rows = df[df.isna().any(axis=1)]
        df = df.dropna()
        if not dropped_rows.empty:
            print(f"Rows with NaN values in {h5_filename}:\n", dropped_rows)

        df['TO_DATE'] = pd.to_datetime(df['TO_DATE'], unit='s', utc=True)
        df.set_index('TO_DATE', inplace=True, drop=True)

        df = df[df.index.year.isin([2022, 2023])]

        dataframes[h5_filename] = df

    return dataframes[h5_files[0]], dataframes[h5_files[1]], dataframes[h5_files[2]]  # qh, minute, hour
# Flags and parameters
elia_comparison = False
sbs = True
n_features_to_select = 26
MINUTE = 3
forecast_horizon = 1  # 0 or 1

# ----------------------------
# 1. Define a scikit-learn wrapper for the PyTorch model
# ----------------------------
from sklearn.base import BaseEstimator, RegressorMixin

class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_dim=32, num_hidden_layers=1, epochs=700, lr=0.01, batch_size=256, device=None):
        if device is None:
            raise ValueError("Device must be provided externally.")
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.model = None  # built in fit()

    def fit(self, X, y):
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        input_dim = X.shape[1]
        self.model = RegressionNN(input_dim=input_dim, hidden_dim=self.hidden_dim, num_hidden_layers=self.num_hidden_layers).to(self.device)

        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y.values).view(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=(self.device.type == "cuda"))
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        self.model.train()

        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy().flatten()
        return y_pred

    def get_params(self, deep=True):
        return {
            "hidden_dim": self.hidden_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "device": self.device
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self



# ----------------------------
# 2. Define your PyTorch model (RegressionNN)
# ----------------------------
class RegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_hidden_layers=1):
        super().__init__()
        layers = []
        in_dim = input_dim

        # build num_hidden_layers of (Linear → ReLU)
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # final output layer
        layers.append(nn.Linear(in_dim, 1))

        # wrap as a single sequential module
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
# ----------------------------
# 3. RFE based on permutation
# ----------------------------
from sklearn.model_selection import cross_val_score
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm

def run_permutation_rfe(
    model_fn,
    data: pd.DataFrame,
    time_splits,
    features: list,
    target: str,
    min_features_to_select: int = 5,
    scoring=None,
    verbose: bool = True
):
    """
    Recursive Feature Elimination using permutation importance.
    Args:
        model_fn: function that returns a fresh estimator (like TorchRegressor)
        data: DataFrame with all input features and target
        time_splits: precomputed time series splits
        features: full list of feature names
        target: name of the target column
        min_features_to_select: stop once this many features remain
        scoring: function(y_true, y_pred) -> float (higher is better). Defaults to -MAE.
        verbose: print progress

    Returns:
        ranked_features: list of features, most important last
        history: list of tuples (remaining_features, avg_importance_scores)
    """
    remaining_features = features.copy()
    ranked_features = []
    history = []

    if scoring is None:
        scoring = lambda y_true, y_pred: -mean_absolute_error(y_true, y_pred)

    while len(remaining_features) > min_features_to_select:
        if verbose:
            print(f"\nPermutation step with {len(remaining_features)} features...")

        # Train model on current features
        model = model_fn(len(remaining_features))
        backtest_df, _ = backtesting_CV(
            model=model,
            data=data,
            time_splits=time_splits,
            features=remaining_features,
            target=target,
            progress_bar=False,
            verbose=False,
            use_scaler=False,
        )

        y_true = backtest_df["y_test"].values
        y_pred = backtest_df["y_pred"].values
        base_score = scoring(y_true, y_pred)

        # Calculate permutation importances
        perm_scores = {}
        for feat in tqdm(remaining_features, desc="Permuting features", leave=False):
            df_copy = data.copy()
            shuffled = df_copy[feat].sample(frac=1.0, random_state=42).reset_index(drop=True)
            df_copy[feat] = shuffled.values

            model = model_fn(len(remaining_features))
            backtest_df_perm, _ = backtesting_CV(
                model=model,
                data=df_copy,
                time_splits=time_splits,
                features=remaining_features,
                target=target,
                progress_bar=False,
                verbose=False,
                use_scaler=False,
            )
            y_pred_perm = backtest_df_perm["y_pred"].values
            score_perm = scoring(y_true, y_pred_perm)
            importance = base_score - score_perm  # drop in performance
            perm_scores[feat] = importance

        # Remove least important
        worst_feature = min(perm_scores, key=perm_scores.get)
        remaining_features.remove(worst_feature)
        ranked_features.insert(0, worst_feature)

        avg_imp = np.mean(list(perm_scores.values()))
        history.append((remaining_features.copy(), avg_imp))

        if verbose:
            print(f"Removed '{worst_feature}'. Remaining: {len(remaining_features)}")

    return remaining_features + ranked_features, history

# ----------------------------
# 3. Main execution: data preparation, training via backtesting, and analysis
# ----------------------------
def main():
    # Load and join data
    qh, minute, hour = load_and_preprocess_data()
    MINUTE = 3  # You can also set a specific minute (e.g., 3), "all"

    qh_parameters= {'GDV': {'lags': [-1]}, 'GEN_DA': {'lags': []}, 'IP': {'lags': [-12]}, 'LOAD_ID': {'lags': [-3, 3]}, 'LOAD_RT': {'lags': [-4, -3]}, 'NETPOS_BE_ID': {'lags': [0, 1, 4, 8, 13]}, 'NRV': {'lags': [-3]}, 'SI': {'lags': [forecast_horizon, -96, -3, -1]}, 'SOLAR_ID': {'lags': [2, 4]}, 'SOLAR_RT': {'lags': [-2, -1]}, 'WIND_RT': {'lags': [-2, -1]}, 'XB_DA_NET_France': {'lags': [0]}, 'aFRR+': {'lags': [-1]}, 'aFRR-': {'lags': [-3]}}
    minute_parameters= {'IP_min': {'lags': [-2]}, 'SI_min': {'lags': [-3, -2]}}
    hour_parameters= None

    df = join_qh_min_data(
        qh_data=qh,
        minute_data=minute,
        qh_parameters=qh_parameters,
        minute_parameters=minute_parameters,
        minute=MINUTE,
        hour_data=hour,
        hour_parameters=hour_parameters
    )
    print(df.columns.tolist())
    df = df.dropna()
    df.info(memory_usage="deep", verbose=False)

    # Setup TimeSeriesSplit and visualize splits
    #tscv = get_time_series_splits()
    # visualize_timeseries_splits(df, tscv)
    tscv = TimeSeriesSplit(
        n_splits= 13 , #52 * 7 // 10, 13 splits (every 10 days a new split)
        max_train_size= 4 * 24 * 7 * 50,  # 50 weeks of training data
        test_size= 4 * 24 * 28,  # 28 days of test data
        gap=0 # No gap between train and test
    )
    # Dynamic target selection
    if forecast_horizon == 1:
        TARGET = "SI_from_qh_plus_1"
    else:
        TARGET = "SI_current_qh"
    FEATURES = [c for c in df.columns if c != TARGET]


    # for no tuning...
    best_params = {"epochs": 200, "lr": 0.01, "batch_size": 256, "hidden_dim": 32, "num_hidden_layers": 1}

    # 1) Now train your final wrapper with best_params:
    model_wrapper = TorchRegressor(
        hidden_dim=32, num_hidden_layers=1, epochs=best_params["epochs"], lr=best_params["lr"], batch_size=best_params["batch_size"], device=device  # Device is determined externally
    )
    start = perf_counter()
    # Run backtesting cross-validation using the TorchRegressor
    results = backtesting_CV(
        model=model_wrapper,
        data=df,
        time_splits=tscv,
        features=FEATURES,
        target=TARGET,
        progress_bar=True,
        verbose=True,
        use_scaler=False,
    )
    print(f"⏱️ Backtesting took {perf_counter() - start:.2f} seconds")


    # results[0] is the backtest DataFrame and results[1] contains metrics.
    backtest_df, metrics = results
    print("Backtest results:")
    print(backtest_df.head())
    print("Metrics:")
    print(metrics)

    def model_fn(n_features):
        return TorchRegressor(
            hidden_dim=32,
            num_hidden_layers=1,
            epochs=300,
            lr=0.01,
            batch_size=256,
            device=device  # or "cpu" if you want safety during SBS
        )

    if sbs is False:  # RFE alternative
        start = perf_counter()
        def model_fn(n_features):
            return TorchRegressor(
                hidden_dim=32,
                num_hidden_layers=1,
                epochs=200,
                lr=0.01,
                batch_size=256,
                device=device
            )
    
        ranked_feats, history = run_permutation_rfe(
            model_fn=model_fn,
            data=df,
            time_splits=tscv,
            features=FEATURES,
            target=TARGET,
            min_features_to_select=5,
            scoring=None,  # default = neg MSE
            verbose=True
        )
    
        print("Ranked features (most important last):")
        print("RFE took", perf_counter()-start)
        for i, feat in enumerate(ranked_feats[::-1]):
            print(f"{i+1}. {feat}")

    # 3) Compare with Elia forecasts
    if elia_comparison:
        compare_with_elia_preds(
            y_true=backtest_df_orig["y_test"],
            y_pred_model=backtest_df_orig["y_pred"],
            elia_forecasts={
                "QH+0": df["si_forecast_qh_current_current_minute"],
                "QH+1": df["si_forecast_qh_plus1_current_minute"]
            }
        )



if __name__ == "__main__":
    main()
