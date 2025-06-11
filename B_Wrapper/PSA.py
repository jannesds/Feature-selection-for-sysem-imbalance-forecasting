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
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
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
# Parameters
MINUTE = 3
forecast_horizon = 1 # qh+0 = 0; qh+1 = 1, ...

qh_parameters= {'DayOfWeek_sin': {'lags': [0]}, 'GDV': {'lags': [-192]}, 'GEN_DA': {'lags': [-262]}, 'GUV': {'lags': [-1]}, 'IGCC-': {'lags': [-3, -1]}, 'IP': {'lags': [-12, -11, -4, -3, -2]}, 'LOAD_DA': {'lags': [-3, -1, 0, 2]}, 'LOAD_ID_P90': {'lags': [-3, -1, 0, 1, 2]}, 'LOAD_ID': {'lags': [-3, 0, 1, 2, 3]}, 'LOAD_RT': {'lags': [-5, -4, -3]}, 'MDP': {'lags': [-6, -4, -3]}, 'MIP': {'lags': [-4, -3]}, 'Month_cos': {'lags': [0]}, 'NETPOS_BE_ID': {'lags': [0, 1, 4, 8, 9, 13, 14]}, 'NRV': {'lags': [-4, -3, -2, -1]}, 'SI': {'lags': [-193, -192, -97, -96, -25, -21, -3, -1, 1]}, 'SOLAR_ID': {'lags': [-1, 0, 2, 3, 4]}, 'SOLAR_RT': {'lags': [-2, -1]}, 'WIND_P90': {'lags': [0, 1]}, 'WIND_RT': {'lags': [-10, -6, -4, -3, -2, -1]}, 'XB_DA_EXP_France': {'lags': [0]}, 'XB_DA_EXP_Germany': {'lags': [0, 1]}, 'XB_DA_EXP_Netherlands': {'lags': [0, 1]}, 'XB_DA_EXP_UnitedKingdom': {'lags': [-1, 0]}, 'XB_DA_IMP_France': {'lags': [0]}, 'XB_DA_IMP_Germany': {'lags': [1]}, 'XB_DA_IMP_UnitedKingdom': {'lags': [-1, 0]}, 'XB_DA_NET_France': {'lags': [-1, 1]}, 'XB_DA_NET_Germany': {'lags': [0]}, 'XB_DA_NET_Netherlands': {'lags': [-38]}, 'XB_DA_NET_UnitedKingdom': {'lags': [1]}, 'XB_RT_Germany': {'lags': [-11, -3]}, 'XB_RT_Luxembourg': {'lags': [-3]}, 'aFRR+': {'lags': [-192, -96, -6, -1]}, 'aFRR-': {'lags': [-7, -3]}}
minute_parameters= {'GDV_min': {'lags': [-3, -2]}, 'GUV_min': {'lags': [-3, -2]}, 'IGCC+_min': {'lags': [-3, -2]}, 'IGCC-_min': {'lags': [-3, -2]}, 'MDP_min': {'lags': [-3, -2]}, 'MIP_min': {'lags': [-2]}, 'SI_min': {'lags': [-31, -5, -3, -2]}, 'aFRR+_min': {'lags': [-3, -2]}, 'aFRR-_min': {'lags': [-2]}}
hour_parameters= {'NETPOS_GB_DA': {'lags': [1, 2]}}

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
# PSA
# ----------------------

X = df[FEATURES]
y = df[TARGET]
num_features = len(FEATURES)

# Convert column names to array indices
feature_names = FEATURES.copy()

import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

# PSO needs data as NumPy array
X_np = X.values
y_np = y.values
feature_names = FEATURES.copy()
num_features = X_np.shape[1]
best_score = None
best_features = None

# Objective function for PSO
def pso_objective_function(particles):
    """
    particles: np.ndarray of shape (n_particles, n_features), where each entry is in [0,1]
               indicating feature selection (1 = include, 0 = exclude)
    Returns:
        np.ndarray of shape (n_particles,) with negative MAE for each particle
    """
    global best_score
    global best_features
    scores = []
    for particle in particles:
        selected_features_idx = np.where(particle >= 0.5)[0]
        if len(selected_features_idx) == 0:
            scores.append(0)  # Penalize empty feature set
            continue

        selected_features = [feature_names[i] for i in selected_features_idx]

        try:
            _, metrics = backtesting_CV(
                model=LinearRegression(),
                data=df,
                time_splits=tscv,
                features=selected_features,
                target=TARGET,
                analyze_results=False,
            )
            score = -metrics['MAE']  # Minimize MAE → maximize negative MAE
        except Exception as e:
            print("Error during fitness eval:", e)
            score = 0
            
        # Track best feature set found so far
        if best_score is None or score > best_score:
            best_score = score
            best_features = selected_features
        scores.append(score)

    return np.array(scores)

# Initialize the PSO optimizer
options = {
    'c1': 2.0,   # cognitive parameter
    'c2': 2.0,   # social parameter
    'w': 0.9,    # inertia weight
    'k': 5,      # number of neighbors (used for ring topology)
    'p': 2       # L2-norm for distance
}



# Binary PSO
optimizer = ps.discrete.BinaryPSO(
    n_particles=40,
    dimensions=num_features,
    options=options
)


# Run optimization
best_cost, best_pos = optimizer.optimize(pso_objective_function, iters=30)
print("\n--- Best particle according to PSO ---")
selected_features = [f for i, f in enumerate(feature_names) if best_pos[i] == 1]
print("Selected features (best_pos):", selected_features)
print("Best fitness from PSO (negative MAE):", best_cost)

print("\n--- Best features actually observed during search ---")
print("Best features:", best_features)
print("Best observed fitness (negative MAE):", best_score)


# Get selected features
selected_features = [f for i, f in enumerate(feature_names) if best_pos[i] == 1]

print("\n--- PSO RESULTS ---")
print("Selected features:", selected_features)
print("Best fitness (negative MAE):", best_cost)

# Re-evaluate with selected features
_, final_metrics = backtesting_CV(
    model=LinearRegression(),
    data=df,
    time_splits=tscv,
    features=selected_features,
    target=TARGET
)
print("Final metrics with selected features:", final_metrics)


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
