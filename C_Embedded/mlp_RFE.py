import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import os, sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from copy import deepcopy
from tqdm import tqdm
from time import perf_counter

# Project imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from source.Tools import setup_device_and_seed, ResultsAnalyzer
from source.join_qh_min_data import join_qh_min_data

# Device setup
print("Available CPUs:", os.cpu_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")

# Load data
def load_and_preprocess_data():
    h5_dir = os.path.join(project_root, "h5")
    h5_files = [
        "quarter_hours_data_2022.01.01_to_2024.01.01.h5",
        "minutes_data_2022.01.01_to_2024.01.01.h5",
        "hours_data_2022.01.01_to_2024.01.01.h5"
    ]
    dataframes = {}
    print("Loading and preprocessing HDF5 files...")

    for h5_filename in h5_files:
        file_path = os.path.join(h5_dir, h5_filename)
        print("Loading data from:", file_path)
        df = ResultsAnalyzer.load_data_from_h5(file_path)
        df = df.dropna()
        df['TO_DATE'] = pd.to_datetime(df['TO_DATE'], unit='s', utc=True)
        df.set_index('TO_DATE', inplace=True)
        df = df[df.index.year.isin([2022, 2023])]
        dataframes[h5_filename] = df

    return dataframes[h5_files[0]], dataframes[h5_files[1]], dataframes[h5_files[2]]
def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    numerator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)

    smape_values = 2 * numerator / denominator
    result = np.mean(smape_values)*100
    return result
def backtesting_CV_alt(
    model,
    data: pd.DataFrame,
    time_splits: TimeSeriesSplit,
    features: list,
    target: str,
    use_scaler: bool = True,
    n_jobs: int = -1,
    return_per_fold: bool = False,
):
    """
    Alternative backtesting using sklearn.cross_validate.

    Parameters:
      model        : sklearn estimator
      data         : DataFrame containing features+target
      time_splits  : TimeSeriesSplit instance
      features     : list of feature column names
      target       : name of target column
      use_scaler   : whether to wrap model in a StandardScaler pipeline
      n_jobs       : parallel jobs for cross_validate
      return_per_fold: if True, returns a DataFrame of per-sample predictions

    Returns:
      backtest_summary: DataFrame with one row per fold:
        [split, train_MAE, test_MAE, train_RMSE, test_RMSE, train_MAPE, test_MAPE]
      metrics: overall metrics dict same as your original API
      per_sample_df (optional): full y_test vs y_pred for all folds
    """
    start = perf_counter()
    X = data[features]
    y = data[target]

    # build pipeline
    estimator = make_pipeline(StandardScaler(), clone(model)) if use_scaler else clone(model)

    # define scorers
    def rmse(y_true, y_pred): return root_mean_squared_error(y_true, y_pred)
    def mape(y_true, y_pred):
        eps = np.finfo(float).eps
        return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

    scorers = {
        'mae': make_scorer(mean_absolute_error, greater_is_better=True),
        'rmse': make_scorer(lambda y, y_pred: root_mean_squared_error(y, y_pred), greater_is_better=True),
        'mape': make_scorer(lambda y, y_pred: np.mean(np.abs((y - y_pred) / np.maximum(np.abs(y), np.finfo(float).eps))) * 100, greater_is_better=True),
        'smape': make_scorer(smape, greater_is_better=False),

    }

    cv_results = cross_validate(
        estimator, X, y,
        cv=time_splits,
        scoring=scorers,
        return_train_score=True,
        n_jobs=n_jobs,
        return_estimator=return_per_fold
    )
    calc_time = perf_counter() - start
    # Assemble a per-fold summary
    summary = []
    n_splits = time_splits.n_splits
    for i in range(n_splits):
        summary.append({
            'split': i,
            'train_MAE': cv_results['train_mae'][i],
            'test_MAE':  cv_results['test_mae'][i],
            'train_RMSE': cv_results['train_rmse'][i],
            'test_RMSE':  cv_results['test_rmse'][i],
            'train_MAPE': cv_results['train_mape'][i],
            'test_MAPE':  cv_results['test_mape'][i],
            'test_SMAPE': cv_results['test_smape'][i],
            'train_SMAPE': cv_results['train_smape'][i],
            'fit_time': cv_results['fit_time'][i],
            'score_time': cv_results['score_time'][i],
        })

    backtest_summary = pd.DataFrame(summary)
    # Overall metrics
    metrics = {
        'train_MAE':  backtest_summary['train_MAE'].mean(),
        'test_MAE':   backtest_summary['test_MAE'].mean(),
        'train_RMSE': backtest_summary['train_RMSE'].mean(),
        'test_RMSE':  backtest_summary['test_RMSE'].mean(),
        'train_MAPE': backtest_summary['train_MAPE'].mean(),
        'test_MAPE':  backtest_summary['test_MAPE'].mean(),
        'test_SMAPE': backtest_summary['test_SMAPE'].mean(),
        'train_SMAPE': backtest_summary['train_SMAPE'].mean(),
        'n_splits':   n_splits,
        'runtime_s': calc_time,
        'total_time_s': calc_time,
        'total_time_fit': backtest_summary['fit_time'].sum(),
        'total_time_score': backtest_summary['score_time'].sum(),
        'total_fold_cpu_time' : backtest_summary['fit_time'].sum() + backtest_summary['score_time'].sum(),
        'efficiency_estimate' : backtest_summary['fit_time'].sum() + backtest_summary['score_time'].sum() / (calc_time * n_cores)
    }

    if return_per_fold:
        # Collect per-sample predictions from each estimator
        per_sample = []
        for i, est in enumerate(cv_results['estimator']):
            train_idx, test_idx = list(time_splits.split(X))[i]
            y_pred = est.predict(X.iloc[test_idx])
            per_sample.append(pd.DataFrame({
                'y_test': y.iloc[test_idx].values,
                'y_pred': y_pred,
                'split': i
            }, index=y.iloc[test_idx].index))
        per_sample_df = pd.concat(per_sample)

        # ✅ Now it's safe to call smape and print detailed values
        print("SMAPE (manual):")
        manual_smape = smape(per_sample_df["y_test"], per_sample_df["y_pred"])
        print(manual_smape)

        return backtest_summary, metrics, per_sample_df



    return backtest_summary, metrics
# Recursive feature elimination using permutation importance
def rfe_with_permutation_importance(model_fn, X, y, scoring, min_features=10, step=1):
    features = list(X.columns)
    eliminated_features = []
    history = []
    timings = []

    while len(features) > min_features:
        model = model_fn(len(features))  # new model instance per iteration
        start = perf_counter()
        model.fit(X[features], y)
        result = permutation_importance(model, X[features], y, scoring=scoring, n_repeats=5, random_state=42, n_jobs=-1)
        importances = pd.Series(result.importances_mean, index=features)
        end = perf_counter()

        history.append((list(features), importances.copy()))
        timings.append(end - start)

        to_drop = importances.nsmallest(step).index.tolist()
        for feat in to_drop:
            features.remove(feat)
            eliminated_features.append(feat)

    return history, eliminated_features, timings


# Main
if __name__ == "__main__":
    qh, minute, hour = load_and_preprocess_data()

    forecast_horizon = 1
    MINUTE = 3

    qh_parameters= {'DayOfWeek_sin': {'lags': [0]}, 'GDV': {'lags': [ -1]}, 'IGCC-': {'lags': [ -2, -1]}, 'IP': {'lags': [-12,  -3, -2]}, 'LOAD_DA': {'lags': [ -1, 2]}, 'LOAD_ID_P90': {'lags': [-1, 2, 3]}, 'LOAD_ID': {'lags': [0]}, 'LOAD_RT': {'lags': [-4, -3]}, 'MDP': {'lags': [-3]}, 'MIP': {'lags': [-4]}, 'NETPOS_BE_ID': {'lags': [0, 1, 4, ]}, 'NRV': {'lags': [-4, -3]}, 'SI': {'lags': [-97, -5, -3, -1, 1]}, 'SOLAR_ID': {'lags': [2, 4]}, 'SOLAR_RT': {'lags': [-2, -1]}, 'WIND_P90': {'lags': [0, 1]}, 'WIND_RT': {'lags': [ -2, -1]}, 'XB_DA_EXP_Germany': {'lags': [0, 1]}, 'XB_DA_EXP_Netherlands': {'lags': [ 0, 1]}, 'XB_DA_EXP_UnitedKingdom': {'lags': [0]}, 'XB_DA_IMP_UnitedKingdom': {'lags': [0]}, 'XB_DA_NET_France': {'lags': [0, 1]}, 'XB_DA_NET_UnitedKingdom': {'lags': [-2, 1]}, 'XB_RT_Germany': {'lags': []}, 'XB_RT_UnitedKingdom': {'lags': [-3]}, 'aFRR+': {'lags': [ -6, -1]}, 'aFRR-': {'lags': [-3, -1]}}
    minute_parameters= {'IGCC+_min': {'lags': [-2]}, 'MDP_min': {'lags': [-2]}, 'MIP_min': {'lags': [-2]}, 'NRV_min': {'lags': [-4]}, 'SI_min': {'lags': [ -3, -2]}}
    hour_parameters= None

    tscv = TimeSeriesSplit(
        n_splits= 13 , #52 * 7 // 10, 13 splits (every 10 days a new split)
        max_train_size= 4 * 24 * 7 * 50,  # 50 weeks of training data
        test_size= 4 * 24 * 28,  # 28 days of test data
        gap=0 # No gap between train and test
    )


    start= perf_counter()
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

    # Example target setup (customize as needed)
    target_var = "SI_from_qh_plus_1"
    features = [f for f in df.columns if f != target_var]
    X = df[features]
    y = df[target_var]



    # Example: simple model to use with permutation importance
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=42)


    # Define your model factory
    def model_fn(n_features):
        return MLPRegressor(
            hidden_layer_sizes=(32,),
            activation='relu',
            solver='adam',
            learning_rate_init=0.01,
            max_iter=500,
            batch_size=256,
            random_state=42,
            early_stopping=False
        )


    results = backtesting_CV_alt(
        model=model_fn(len(features)),  # ✅ Call the function to get the model instance
        data=df,
        time_splits=tscv,
        features=features,
        target=target_var
    )
    # Run RFE
    history, eliminated, timings = rfe_with_permutation_importance(
        model_fn=model_fn,
        X=X,
        y=y,
        scoring='neg_mean_absolute_error',
        min_features=1,
        step=1
    )

    print("Timings per RFE iteration:", timings)
    print("Total RFE time:", sum(timings), "seconds")
    print("Everything total time:", perf_counter()-start)

    from sklearn.metrics import mean_absolute_error

    mae_per_step = []
    backtest_results = []

    for feature_subset, _ in history:
        filtered_df = df[feature_subset + [target_var]]

        backtest_summary, metrics = backtesting_CV_alt(
            model=model_fn(len(features)),
            data=filtered_df,
            time_splits=tscv,
            features=feature_subset,
            target=target_var,
            use_scaler=True,
            n_jobs=-1,
            return_per_fold=False
        )

        mae_per_step.append(metrics['test_MAE'])
        backtest_results.append((metrics, backtest_summary))
    best_idx = np.argmin(mae_per_step)
    best_features = history[best_idx][0]
    best_mae = mae_per_step[best_idx]
    best_metrics, best_summary = backtest_results[best_idx]

    print(f"\n✅ Best feature count: {len(best_features)}")
    print(f"✅ Best CV MAE: {best_mae:.4f}")
    print("✅ Selected features:\n", best_features)
    print("✅ Other metrics:\n", best_metrics)

