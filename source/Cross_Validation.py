# -*- coding: utf-8 -*-
"""
-> Script to train and cross-validate a forecasting model (scikit-learn compatible) on a provided TimeSeriesSplit.
"""
from time import perf_counter

import numpy as np
import pandas as pd
from mpmath.libmp import round_up
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm import tqdm
from source.Tools import ResultsAnalyzer

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math



def get_time_series_splits(number_splits = 5, training_weeks: int = None):
    """
    Returns a configured TimeSeriesSplit object.

    Parameters:
        number_splits: Number of splits in cross validation
        training_weeks (int, optional): Number of weeks to include in the training set.
                                        If None, training size is unlimited (grows over time).

    Returns:
        TimeSeriesSplit: Configured splitter with optional training window.
    """

    if training_weeks is not None:
        max_train_size = (4*24*7) * training_weeks  # 4 data points/hour/day/week
        test_size = 4 * 24 * round(training_weeks * 7 * 0.1 / 0.8)  # 4 days of test data (15-min intervals)
    else:
        max_train_size = (4*24*7) * 8
        test_size = 4 * 24 * round(8*7 * 0.2 / 0.8)


    return TimeSeriesSplit(
        n_splits= number_splits, #52 * 7 // 4, 91 splits (every 4 days a new split)
        max_train_size= max_train_size, #80% train
        test_size=test_size,  # 10% test
        gap=0 # No gap between train and test
    )


def backtesting_CV(
        model,
        data: pd.DataFrame,
        time_splits: TimeSeriesSplit,
        features: list | str,
        target: list | str,
        use_scaler: bool | None = True,
        progress_bar: bool | None = True,
        print_error_metrics: bool | None = True,
        verbose: bool | None = False,
        analyze_results: bool = True,
        forecast_horizon_qh: int = 1,
):
    start = perf_counter()
    model_name = type(model).__name__
    print(f"MODEL: {model_name}")

    guess_index = np.random.randint(len(data) - 1)
    all_minutes = (
            abs(data.index[-guess_index].minute - data.index[-guess_index - 1].minute) < 15
    )

    if all_minutes:
        print(
            f"Minutes: Time configuration: {time_splits.n_splits} splits, {time_splits.test_size // 15 // (4 * 24)} testing days, "
            f"{time_splits.max_train_size // 15 // (4 * 24 * 7)} training weeks. "
            f"Total predicted time: {time_splits.n_splits * time_splits.test_size // 15 // (4 * 24)} days.\n"
        )
        print(
            f"{time_splits.n_splits},{time_splits.max_train_size // 15 // (4 * 24 * 7)},{time_splits.test_size // 15 // (4 * 24)},{time_splits.n_splits * time_splits.test_size // 15 // (4 * 24)}")
    else:
        print(
            f"Else: Time configuration: {time_splits.n_splits} splits, {time_splits.test_size // 4 // 24} testing days, "
            f"{time_splits.max_train_size // 4 // 24 // 7} training weeks. "
            f"Total predicted time: {time_splits.n_splits * time_splits.test_size // 4 // 24} days.\n"
        )

    backtest, predictions, split_numbers = pd.DataFrame(), np.array([]), np.array([])
    y_train_full, y_train_pred_full = np.array([]), np.array([])
    cutting_train_size = False

    for ii, (train_idx, test_idx) in tqdm(
            enumerate(time_splits.split(data)),
            total=time_splits.n_splits,
            disable=(not progress_bar) or verbose,
    ):
        if features is not None and len(features) > 0:
            X_train = data[features].iloc[train_idx]
            X_test = data[features].iloc[test_idx]
        else:
            X_train = np.arange(len(data))[train_idx].reshape(-1, 1)
            X_test = np.arange(len(data))[test_idx].reshape(-1, 1)
        y_train = data[target].iloc[train_idx]
        y_test = data[target].iloc[test_idx]
        if (
                train_idx.shape[0] != time_splits.max_train_size
                and cutting_train_size is False
        ):
            cutting_train_size = True
            print(
                "WARNING: there isn't enough data to fulfill the desired max_train_size for all time splits!"
            )

        if use_scaler:
            model_pipeline = make_pipeline(StandardScaler(), clone(model))
        else:
            model_pipeline = clone(model)

        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)
        y_train_pred = model_pipeline.predict(X_train)
        last_trained_model = model_pipeline  # stores the model from the current fold

        mae_test = mean_absolute_error(y_test, y_pred)
        # RMSE should be the square root of the MSE:
        rmse_test = root_mean_squared_error(y_test, y_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        rmse_train = root_mean_squared_error(y_train, y_train_pred)

        if verbose:
            if time_splits.n_splits > 10:
                if (ii % 10 == 0) or (ii + 1 == time_splits.n_splits):
                    tqdm.write(
                        f"Train MAE|RMSE for fold {ii} is {mae_train:.2f} | {rmse_train:.2f} MW\n"
                        f"Test  MAE|RMSE for fold {ii} is {mae_test:.2f} | {rmse_test:.2f} MW"
                    )
            else:
                tqdm.write(
                    f"Train MAE|RMSE for fold {ii} is {mae_train:.2f} | {rmse_train:.2f} MW\n"
                    f"Test  MAE|RMSE for fold {ii} is {mae_test:.2f} | {rmse_test:.2f} MW"
                )

        backtest = pd.concat([backtest, y_test], axis=0)
        predictions = np.append(predictions, y_pred)
        split_numbers = np.append(
            split_numbers, ii * np.ones(len(y_test), dtype=np.int16)
        )
        y_train_full = np.append(y_train_full, y_train)
        y_train_pred_full = np.append(y_train_pred_full, y_train_pred)

    print("backtest_columns: ",backtest.columns.tolist())  # This will show the current column names
    # Rename column: Ensure you rename the correct column name if needed
    if backtest.shape[1] == 1:
        backtest.columns = ["y_test"]
    else:
        raise ValueError("Expected backtest to have only one column.")

    print(backtest.columns)  # This will show the current column names
    backtest = backtest.assign(y_pred=predictions.astype(backtest["y_test"].dtype))
    backtest = backtest.assign(
        error=(backtest["y_test"] - backtest["y_pred"]).abs()
    ).assign(split_number=split_numbers.astype(np.int16))

    total_mae = mean_absolute_error(backtest["y_test"], backtest["y_pred"])
    # Use square root of MSE for RMSE
    total_rmse = root_mean_squared_error(backtest["y_test"], backtest["y_pred"])

    diff_period = forecast_horizon_qh
    if all_minutes:
        diff_period *= 15  # convert QH → minutes

    total_mase = (
            total_mae / backtest["y_test"].diff(periods=diff_period).dropna().abs().mean()
    )

    max_error = (backtest["y_test"] - backtest["y_pred"]).abs().max()
    max_error_dt = (backtest["y_test"] - backtest["y_pred"]).abs().idxmax()
    P90_error = backtest["error"].quantile(q=0.90)

    total_mae_train = mean_absolute_error(y_train_full, y_train_pred_full)
    total_rmse_train = root_mean_squared_error(y_train_full, y_train_pred_full)

    end = perf_counter()
    if print_error_metrics:
        print(
            f"\nTrain set average error:\n"
            f"MAE: {total_mae_train:.2f} MW | RMSE: {total_rmse_train:.2f} MW"
        )
        print(
            f"Test set average error:\n"
            f"MAE: {total_mae:.2f} MW | RMSE: {total_rmse:.2f} MW | MASE: {total_mase:.4f} | "
            f"P90 Error: {P90_error:.2f} MW | Max Error: {max_error:.2f} MW\nMax Error Date: {max_error_dt}"
        )
    print(f"Total execution time: {round(end - start, 2)} seconds")
    # Cleaned print output (removed duplicate RMSE)
    print(f"{total_mae_train:.2f}".replace('.', ','))
    print(f"{total_rmse_train:.2f}".replace('.', ','))
    print(f"{total_mae:.2f}".replace('.', ','))
    print(f"{total_rmse:.2f}".replace('.', ','))
    print(f"{total_mase:.2f}".replace('.', ','))
    print(f"{P90_error:.2f}".replace('.', ','))
    print(f"{max_error:.2f}".replace('.', ','))
    print(max_error_dt)

    if analyze_results:
        analyzer = ResultsAnalyzer(
            model=last_trained_model,
            X_train=None, y_train=None, X_test=None, y_test=None, X_val=None, y_val=None,
            feature_names=features if isinstance(features, list) else [features],
            device=None,
            date_index=data.index  # assumes DateTimeIndex
        )
        analyzer.plot_backtest_results(backtest)

    return [
        backtest,
        {
            "model": model,
            "tscv": time_splits,
            "MASE": total_mase,
            "MAE": total_mae,
            "MAE_train": total_mae_train,
            "RMSE": total_rmse,
            "max_error": max_error,
            "P90_error": P90_error,
            "trained_model": last_trained_model,
            "y_true": backtest["y_test"],
            "y_pred": backtest["y_pred"],
        },
    ]

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

def visualize_timeseries_splits(df, tscv):
    """
    Visualizes the train-test splits for a time series dataset using TimeSeriesSplit.

    Parameters:
        df (pd.DataFrame): DataFrame with a DateTime index.
        tscv (TimeSeriesSplit): An instance of sklearn's TimeSeriesSplit.
    """
    color = "#1FABD5"
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

    # Ensure white background
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (train_idx, test_idx) in enumerate(tscv.split(df)):
        ax.plot(df.index[train_idx], [i] * len(train_idx), color=color, lw=2, markersize=2)
        ax.plot(df.index[test_idx], [i] * len(test_idx), 'r.', markersize=4)

    # Labels
    ax.set_xlabel("Date", fontsize=20)
    ax.set_ylabel("Split number", fontsize=20)

    # Custom legend with proper size
    custom_lines = [
        Line2D([0], [0], color=color, lw=2, label='Train'),
        Line2D([0], [0], color='r', marker='.', linestyle='None', markersize=8, label='Test')
    ]
    ax.legend(handles=custom_lines, fontsize=20)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.autofmt_xdate()

    plt.show()
    # Print first two and last two split indices
    num_splits = len(list(tscv.split(df)))

    for i, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_start = df.index[train_idx[0]].strftime('%Y-%m-%d %H:%M:%S')
        train_end = df.index[train_idx[-1]].strftime('%Y-%m-%d %H:%M:%S')
        test_start = df.index[test_idx[0]].strftime('%Y-%m-%d %H:%M:%S')
        test_end = df.index[test_idx[-1]].strftime('%Y-%m-%d %H:%M:%S')

        if i < 2 or i >= num_splits - 2:
            print(f"Split {i + 1}:")
            print(f"  Train range: {train_start} to {train_end} ({len(train_idx)} samples)")
            print(f"  Test range: {test_start} to {test_end} ({len(test_idx)} samples)\n")
    return fig


def main():
    pass


if __name__ == "__main__":
    main()
