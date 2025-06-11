from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from source.Tools import setup_device_and_seed
from source.Cross_Validation import backtesting_CV, visualize_timeseries_splits, get_time_series_splits
from source.join_qh_min_data import join_qh_min_data
from source.features_selection import get_feature_lag_config
from source.Tools import ResultsAnalyzer

from B_wrappers.SBS import run_sbs
import time
from time import perf_counter

import psutil
#from comparison_elia import compare_with_elia_preds
# At start
start_mem = psutil.Process().memory_info().rss / (1024 ** 2)  # in MB
start_time = perf_counter()


# Initialize device and seed
device_info = setup_device_and_seed(seed=987)
device = device_info["device"]

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

# Load data
qh, minute, hour = load_and_preprocess_data()
elia_comparison = False
sbs = True
n_features_to_select = 150
# Parameters
MINUTE = 3
forecast_horizon = 1 # qh+0 = 0; qh+1 = 1, ...

qh_parameters = {
    "SI": {"lags": [forecast_horizon, -1, -2, -3, -4, -5, -21, -25, -92, -93, -96, -97, -192, -193]},
    "LOAD_RT": {"lags": [-3, -4, -5]},
    "LOAD_ID": {"lags": [3, 2, 1, 0, -1, -2, -3]},
    "LOAD_ID_P90": {"lags": [3, 2, 1, 0, -1, -2, -3]},
    "LOAD_DA": {"lags": [3, 2, 1, 0, -1, -2, -3]},
    "NRV": {"lags": [-1, -2, -3, -4]},
    "NETPOS_BE_ID": {"lags": [22, 14, 13, 9, 8, 4, 1, 0, -1 ]},
    "IP": {"lags": [ -1, -2, -3, -4, -5, -6, -7, -8, -9, -11, -12, -13, -92, -96]},
    "MIP": {"lags": [-1, -2, -3, -4, -6, -96] },
    "MDP": {"lags": [-1, -2, -3, -4, -6, -96] },
    "Hour_sin": {"lags": [0]},
    "Hour_cos": {"lags": [0]},
    "DayOfWeek_sin": {"lags": [0]},
    "DayOfWeek_cos": {"lags": [0]},
    "Month_sin": {"lags": [0]},
    "Month_cos": {"lags": [0]},
    "WIND_ID":  {"lags": [ 11, 4, 3, 2, 1, 0, -1, -2,  -14]},
    "WIND_P90":  {"lags": [ 4, 3, 2, 0, -1, -2,  -10]},
    "WIND_RT":  {"lags": [-1, -2,  -5, -6, -7, -10, -11, -12]},
    "SOLAR_ID":  {"lags": [4, 3, 2, 0, -1, -2, -30]},
    "SOLAR_P90":  {"lags": [ 3, 2, -30]},
    "SOLAR_RT":  {"lags": [-1, -2, -29 ]},
    "GEN_DA": {"lags": [-4, -26, -262]},
    "GDV": {"lags": [-1, -2, -3, -4, -5, -6, -7, -8, -96, -192]},
    "GUV": {"lags": [-1, -2, -3]},
    "IGCC+": {"lags": [-1, -2, -3]},
    "aFRR+": {"lags": [-1, -2, -3, -6, -7, -8, -96, -192]},
    "mFRR+": {"lags": [-1, -2, -3]},
    "IGCC-": {"lags": [-1, -2, -3]},
    "aFRR-": {"lags": [-1, -2, -3, -6, -7, -8, -96, -192]},
    'XB_DA_EXP_France':     {"lags": [ 1, 0, -1, -2, -6, -7]},
    'XB_DA_EXP_Germany':    {"lags": [ 1, 0, -1, -2, -6, -7]},
    'XB_DA_EXP_Netherlands': {"lags": [ 1, 0, -1, -2, -38, -45]},
    'XB_DA_EXP_UnitedKingdom': {"lags": [54, 1, 0, -1, -2]},
    'XB_DA_IMP_France':     {"lags": [ 1, 0, -1, -2, -6, -7, -14]},
    'XB_DA_IMP_Germany':    {"lags": [ 1, 0, -1, -2, -6, -7, -10]},
    'XB_DA_IMP_Netherlands': {"lags": [62, 1, 0, -1, -2, -34]},
    'XB_DA_IMP_UnitedKingdom': {"lags": [39, 1, 0, -1, -2]},
    'XB_DA_NET_France':     {"lags": [ 1, 0, -1, -2, -6, -7]},
    'XB_DA_NET_Germany':    {"lags": [1, 0, -1, -2, -6, -10]},
    'XB_DA_NET_Netherlands': {"lags": [ 1, 0, -1, -34, -45]},
    'XB_DA_NET_UnitedKingdom': {"lags": [ 42, 1, 0, -1, -2]},
    'XB_RT':                {"lags": [ -3, -4, -5, -6, -7, -8, -9, -10]},
    'XB_RT_France':         {"lags": [ -3, -4, -5]},
    'XB_RT_Germany':        {"lags": [ -3, -4, -5, -11]},
    'XB_RT_Luxembourg':     {"lags": [ -3, -4, -5, -14]},
    'XB_RT_Netherlands':    {"lags": [ -3, -4, -5, -33]},
    'XB_RT_UnitedKingdom':  {"lags": [ -3, -4, -5, -6]}
}
minute_parameters = {
    "SI_min": {"lags": [ -2, -3,-4,-5, -16, -31, -46, -61]},
    "NRV_min": {"lags": [-2, -3,-4,-5]},
    # "si_forecast_qh_current": {"lags": [0]},
    # "si_forecast_qh_plus1": {"lags": [0]},
    "IP_min": {"lags": [-2,-3,-4]},
    "MIP_min": {"lags":  [-2,-3]},
    "MDP_min": {"lags":  [-2,-3]},
    "GUV_min": {"lags":  [-2,-3]},
    "IGCC+_min": {"lags":  [-2,-3]},
    "aFRR+_min": {"lags":  [-2,-3]},
    "GDV_min": {"lags":  [-2,-3]},
    "IGCC-_min": {"lags":  [-2,-3]},
    "aFRR-_min": {"lags":  [-2,-3]}

}
hour_parameters = {
    "NETPOS_GB_DA": {"lags": [1, 0, -1]},
    "NETPOS_GB_ID": {"lags": [1, 0, -1]},
}
#qh_parameters, minute_parameters, hour_parameters = get_feature_lag_config(forecast_horizon)

df = join_qh_min_data(
    qh_data=qh,
    minute_data=minute,
    qh_parameters=qh_parameters,
    minute_parameters=minute_parameters,
    minute=MINUTE,
    hour_data=hour,
    hour_parameters=hour_parameters
)

df = df.dropna()
df.info(memory_usage="deep", verbose=False)

# Cross-validation
tscv = get_time_series_splits()
# visualize_timeseries_splits(df,tscv)

# Model
model_linear = LinearRegression()
if forecast_horizon == 1:
    TARGET = "SI_from_qh_plus_1"
if forecast_horizon == 0:
    TARGET = "SI_current_qh"
FEATURES = [feature for feature in df.columns if feature != TARGET]

# ----------------------
# Original dataset
# ----------------------

backtest_results = backtesting_CV(
    model=model_linear,
    data=df,
    time_splits=tscv,
    features=FEATURES,
    target=TARGET,
    progress_bar=True,
    verbose=True,
    analyze_results=False,
    forecast_horizon_qh=forecast_horizon+1
)

# ----------------------
# Sequential Backward Selection (with mlxtend)
# ----------------------
if sbs:
    selected_features, df_metric = run_sbs(
        model=lambda n_features: LinearRegression(),
        df=df,
        target=TARGET,
        features=FEATURES,
        cv=tscv,
        n_features_to_select=n_features_to_select
    )

# ----------------------
# Retrain on selected features
# ----------------------
    results_sbs = backtesting_CV(
        model=model_linear,
        data=df,
        time_splits=tscv,
        features=selected_features,
        target=TARGET,
        progress_bar=True,
        verbose=False,
        analyze_results=True,
        forecast_horizon_qh=forecast_horizon+1
    )

# After run
end_mem = psutil.Process().memory_info().rss / (1024 ** 2)
print(f"Memory usage: {end_mem - start_mem:.2f} MB")
print(f"Total execution time: {perf_counter() - start_time:.2f} s")

# ----------------------
# Compare with Elia
# ----------------------
if elia_comparison:
    compare_with_elia_preds(
        y_true=backtest_results[0]["y_test"],
        y_pred_model=backtest_results[0]["y_pred"],
        elia_forecasts={
            "QH+0": df["si_forecast_qh_current_current_minute"],
            "QH+1": df["si_forecast_qh_plus1_current_minute"]
        }
    )
