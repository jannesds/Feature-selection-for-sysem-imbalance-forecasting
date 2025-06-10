
import pandas as pd
from source import Tools


def join_qh_min_data(
    minute: int | str,
    qh_data: pd.DataFrame,
    qh_parameters: dict,
    minute_data: pd.DataFrame | None = None,
    minute_parameters: dict | None = None,
    hour_data: pd.DataFrame | None = None,
    hour_parameters: dict | None = None
) -> pd.DataFrame:
    """
    -> Merge quarter-hourly and minute-wise extracted datasets
    """

    if type(minute) is str and minute != "all":
        raise ValueError("ERROR: Provided a string for a numerical value of MINUTE!")

    # Generate lagged features for QH data:
    qh_data = Tools.generate_lagged_features(data=qh_data, parameters=qh_parameters)
    # Turn quarter-hourly dataset into "fake" minute frequency for merging with minute-wise dataset:
    qh_data = qh_data.asfreq("1min", method="ffill")

    if (minute_data is not None) and (minute_parameters is not None):
        minute_data = Tools.generate_lagged_features(data=minute_data, parameters=minute_parameters)
        # Merge them:
        df = pd.concat([qh_data, minute_data], axis="columns", join="inner")
    else:
        df = qh_data

    # Process hour-level data if available
    if (hour_data is not None) and (hour_parameters is not None):
        hour_data = Tools.generate_lagged_features(data=hour_data, parameters=hour_parameters)
        # Ensure hour_data is at 1-minute frequency for merging
        hour_data = hour_data.asfreq("1min", method="ffill")
        df = df.merge(hour_data, how="inner", right_index=True, left_index=True)

    # Filter for just the current qh minute:
    if minute != "all":
        df = df.loc[df.index.minute % 15 == minute]

    return df


def main():
    pass


if __name__ == "__main__":
    main()