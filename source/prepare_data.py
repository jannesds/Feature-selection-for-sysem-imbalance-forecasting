import os
import datetime as dt
import numpy as np
import pandas as pd
import h5py
import yaml

from source.Elia_data_fetcher import EliaDataFetcher
from source.Elia_data_standarization import EliaPowerSystemData
from source.Tools import ResultsAnalyzer

#general function that is called in each model, easy to change dataset from here
def load_and_preprocess_data(system_imbalance_analysis=False):
    h5_dir = r"C:\Users\jds\OneDrive - KU Leuven\Ku Leuven\Master Semester 5\Thesis\Elia\CSV"
    h5_files = [
        r"quarter_hours_data_2022.01.01_to_2024.01.01.h5",
        r"minutes_data_2022.01.01_to_2024.01.01.h5",
        r"hours_data_2022.01.01_to_2024.01.01.h5"
    ]
    if system_imbalance_analysis:
        h5_files = [ r"quarter_hours_data_2020.01.01_to_2023.12.31.h5",
                     r"minutes_data_2022.01.01_to_2023.12.31.h5",                      #for system imbalance analysis over 4 years
                     r"quarter_hours_data_2020.01.01_to_2023.12.31.h5",
                     ]
    dataframes = {}

    """Loads data from an HDF5 file, drops NaNs, and filters relevant years."""
    for h5_filename in h5_files:
        file_path = os.path.join(h5_dir, h5_filename)
        print("Loading data from:", file_path)

        df = ResultsAnalyzer.load_data_from_h5(file_path)

        # Identify rows with NaN values
        dropped_rows = df[df.isna().any(axis=1)]
        # Drop rows with NaN values
        df = df.dropna()
        if not dropped_rows.empty:
            print(f"Rows with NaN values in {h5_filename}:\n", dropped_rows)

        # Convert timestamps
        df['TO_DATE'] = pd.to_datetime(df['TO_DATE'], unit='s', utc=True)
        df.set_index('TO_DATE', inplace=True, drop=True)

        # Filter for years 2022-2023
        # df = df[df.index.year.isin([2022, 2023])]
        if not system_imbalance_analysis:
            df = df[df.index.year.isin([2022, 2023])]

        # Store the processed dataframe in a dictionary
        dataframes[h5_filename] = df

    return dataframes[h5_files[0]], dataframes[h5_files[1]], dataframes[h5_files[2]]  # qh, minute, hour

class DataPreparer:
    def __init__(self, dataset_name, raw_data_path):
        self.fetcher = EliaDataFetcher()
        self.data = None
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path


    def fetch_data(self, dataset_name: str, start: dt.datetime, end: dt.datetime):
        print("Fetching data...")
        try:
            self.data = self.fetcher.get_elia_data_as_df(dataset_name, start, end)
            print("Data fetched successfully.", self.data.columns.tolist())
            # Sort the data by 'datetime' in ascending order
            if 'datetime' in self.data.columns:
                self.data.sort_values(by='datetime', ascending=True, inplace=True)
                print("Data sorted by datetime.")
            else:
                print("Warning: 'datetime' column not found in the dataset.")
                self.data.sort_values(by='predictiontimeutc',ascending=True, inplace=True)
                self.data.rename(columns={'predictiontimeutc': 'datetime'}, inplace=True)
        except Exception as e:
            print(f"An error occurred while fetching data: {e}")
            raise

    def prepare_data(self):
        """
        Prepare the data by transforming it from the Elia format to the custom format.
        This includes loading, cleaning, and transforming the data.

        Returns:
            pd.DataFrame: The cleaned and transformed data.
        """
        # Load the raw data
        raw_data = self.data

        # Create an instance of EliaPowerSystemData to handle transformations
        self.elia_data = EliaPowerSystemData(self.dataset_name, raw_data, format="Elia")

        # Transform the data
        self.elia_data.transform_from_elia_format_to_db_format()

        # Return the transformed data
        return self.elia_data.df

    def save_transformed_data(self, dataset_name, save_path):
        """
        Save the transformed data to the specified HDF5 file using h5py.

        Args:
            dataset_name (str): The name of the dataset (file name).
            save_path (str): The directory path to save the transformed data.
        """
        if self.elia_data is None:
            raise ValueError("Data has not been processed. Call 'prepare_data' first.")

        if self.elia_data is not None:
            df = self.elia_data.df.copy()  # Make a copy to avoid modifying the original DataFrame
            print("column headers:")
            print(df.columns.tolist())

            # Save to CSV file
            csv_file_path = os.path.join(save_path, f"{dataset_name}.csv")
            try:
                df.to_csv(csv_file_path, float_format='%.16f', index=False, decimal=',')  # Save without the index
                print(f"Data successfully saved to {csv_file_path}")
            except Exception as e:
                print(f"An error occurred while saving to CSV: {e}")

            # Prepare HDF5-compatible data types, # Convert datetime to Unix timestamps in seconds, # Encode strings as UTF-8
            def convert_column(data):
                if pd.api.types.is_datetime64_any_dtype(data):
                    return data.astype('int64') // 10 ** 9
                elif pd.api.types.is_object_dtype(data):
                    return data.astype(str)
                return data  # Leave numeric columns as-is

            # Apply conversion to all columns
            transformed_columns = {col: convert_column(df[col]) for col in df.columns}

            # Save the transformed data to an HDF5 file
            with h5py.File(os.path.join(save_path, f"{dataset_name}.h5"), "w") as file:
                for col, data in transformed_columns.items():
                    dtype = h5py.string_dtype() if data.dtype.kind in {'O', 'S', 'U'} else data.dtype
                    file.create_dataset(col, data=data.values, dtype=dtype)

                # Save the index as a dataset (convert to string if necessary)
                file.create_dataset("index", data=df.index.astype(str).values, dtype=h5py.string_dtype())

            print(f"Data successfully saved to {os.path.join(save_path, f'{dataset_name}.h5')}")
        else:
            raise ValueError("Data has not been processed. Call 'prepare_data' first.")

    def load_transformed_data(self, load_path):
        """
        Load the transformed data from an HDF5 file using h5py.

        Args:
            load_path (str): The path from where to load the transformed data.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        with h5py.File(load_path, "r") as h5file:
            data = {key: h5file[key][...] for key in h5file.keys()}
            # Reconstruct DataFrame
            df = pd.DataFrame(data)
            # Restore index if saved
            if "index" in df:
                df.set_index("index", inplace=True)
        print(f"Data successfully loaded from {load_path}")
        return df

    def apply_cyclical_encoding(self, df):
        # Predefined max values for cyclical features, 0 (Monday) to 6 (Sunday), 0 (January) to 11 (December)
        time_columns = {'Hour': 23, 'DayOfWeek': 6,   'Month': 11}
        for col, max_value in time_columns.items():
            if col in df.columns:
                if f'{col}_sin' not in df.columns and f'{col}_cos' not in df.columns:
                    df[f'{col}_sin'] = np.round(np.sin(2 * np.pi * df[col] / max_value), decimals=3)
                    df[f'{col}_cos'] = np.round(np.cos(2 * np.pi * df[col] / max_value), decimals=3)
        df.drop(columns=time_columns, inplace=True)
        return df

    @staticmethod
    def join_qh_min_data(datasets: dict, frequency: str = "1min") -> pd.DataFrame:
        """
            Merge multiple datasets with varying frequencies into a single DataFrame.

            Parameters:
            ----------
            datasets: dict
                A dictionary where keys are dataset names and values are pandas DataFrames.
            frequency: str
                The frequency to resample datasets for merging (default is "1min").

            Returns:
            -------
            pd.DataFrame
                A merged DataFrame combining all datasets on their datetime index.
            """
        # Resample datasets to the desired frequency and store them in a list
        resampled_datasets = []
        for name, df in datasets.items():
            df = df.set_index("TO_DATE")
            print(df.index)
            print(df.index.dtype)
            if df is not None:
                df_resampled = df.asfreq(frequency, method="ffill")
                resampled_datasets.append(df_resampled)

        # Merge all resampled datasets on their index
        if resampled_datasets:
            merged_df = pd.concat(resampled_datasets, axis="columns", join="inner")
        else:
            raise ValueError("No valid datasets provided for merging.")

        # Ensure the resulting DataFrame is sorted by the datetime index
        merged_df.sort_index(inplace=True)
        # Drop duplicated columns (keeping only the first occurrence)
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        merged_df.reset_index('TO_DATE',inplace=True)
        print("Merged DataFrame - Head (First 20 Rows):"), print(merged_df.head(20))
        print("\nMerged DataFrame - Data Types:"), print(merged_df.dtypes)
        print("\nMerged DataFrame - Index:"), print(merged_df.index)
        print("\nMerged DataFrame - Columns:"), print(merged_df.columns)
        print("\nType of merged_df:"), print(type(merged_df))
        print("\nChecking for NaN values:"), print(merged_df.isna().sum())

        return merged_df
def fix_nan_values(df):
    """
    Fix NaN values and print where they occurred (timestamp + column).

    Args:
        df (pd.DataFrame): DataFrame to clean.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    for col in df.columns:
        if df[col].isna().any():
            # Log NaN positions
            nan_entries = df[df[col].isna()]
            print(f"\n⚠️ Missing values detected in column: '{col}'")
            print("Timestamps with NaNs:")
            print(nan_entries.index.tolist())

            # Fix numeric columns with interpolation and fill
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            else:
                # Fix categorical/object columns with forward/backward fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

    print("\n✅ NaN fixing complete.\n")
    return df

def generate_time_chunks(start: dt.datetime, end: dt.datetime, delta_days: int):
    current = start
    while current < end:
        chunk_end = min(current + dt.timedelta(days=delta_days), end)
        yield current, chunk_end
        current = chunk_end


def main():
    # Define file paths and parameters
    raw_data_path = r"C:\Users\jds\OneDrive - KU Leuven\Ku Leuven\Master Semester 5\Thesis\Elia\CSV"
    save_path = r"C:\Users\jds\OneDrive - KU Leuven\Ku Leuven\Master Semester 5\Thesis\Elia\CSV"

    # Define start and end dates for the data fetching
    original_start = dt.datetime(2023, 1, 1)
    #^^depending on the lag, lag-values before this date & hour have to be added
    original_end = dt.datetime(2024, 1, 1, 4)

    # original_start = dt.datetime(2023, 10, 20); for Elia
    # original_end = dt.datetime(2024, 5, 2, 23)

    config = yaml.safe_load(open(
        os.path.join(r"C:\Users\jds\OneDrive - KU Leuven\Ku Leuven\Master Semester 5\Thesis\Elia",
                     "config.yml")))

    # Access specific configurations
    to_loop_elia = [
        #('minutes', config['minutes']),
        ('quarter_hours', config['quarter_hours']),
        ('hours', config['hours'])
    ]

    total_datasets = {}
    start_str, end_str = original_start.strftime('%Y.%m.%d'), original_end.strftime('%Y.%m.%d')

    for config_name, DATA in to_loop_elia:
        dataset_names = DATA['dataset_names']
        interval_minutes = DATA['interval_minutes']

        combined_dataset_name = f"{config_name}"

        # Define the filename with dates included
        combined_dataset_name = f"{combined_dataset_name}_data_{start_str}_to_{end_str}"

        # Initialize a list to hold the transformed DataFrames
        combined_data = []

        for dataset_name in dataset_names:
            print(f"Processing dataset: {dataset_name}")

            # Initialize DataPreparer
            data_preparer = DataPreparer(dataset_name, raw_data_path)

            # Temporary list to hold chunks of transformed data
            chunked_dfs = []
            failed_chunks = []

            # Loop over time chunks (e.g. 30-day chunks to avoid API overload)
            for chunk_start, chunk_end in generate_time_chunks(original_start, original_end, delta_days=50):
                try:
                    print(f"⏳ Fetching data from {chunk_start} to {chunk_end}...")
                    data_preparer.fetch_data(dataset_name, chunk_start, chunk_end)
                    transformed_chunk = data_preparer.prepare_data()
                    chunked_dfs.append(transformed_chunk)
                except Exception as e:
                    print(f"⚠️ Failed to fetch or transform chunk {chunk_start} to {chunk_end}: {e}")
                    failed_chunks.append((chunk_start, chunk_end))
            if failed_chunks:
                print(f"⚠️ Failed chunks for {dataset_name}: {failed_chunks}")
            for chunk_start, chunk_end in failed_chunks:
                try:
                    print(f"🔁 Retrying chunk {chunk_start} to {chunk_end}...")
                    data_preparer.fetch_data(dataset_name, chunk_start, chunk_end)
                    transformed_chunk = data_preparer.prepare_data()
                    chunked_dfs.append(transformed_chunk)
                except Exception as e:
                    print(f"❌ Final failure for chunk {chunk_start} to {chunk_end}: {e}")

            # Combine all chunks into one DataFrame
            if chunked_dfs:
                dataset_df = pd.concat(chunked_dfs, ignore_index=True).sort_values(by=['TO_DATE'])
                combined_data.append(dataset_df)
            else:
                print(f"❌ No data fetched for dataset {dataset_name}.")

        # Check the type of combined_data to confirm it's a list of DataFrames
        print(f"Type of combined_data: {type(combined_data)}")

        # Start with the first DataFrame as the base, merge subsequent DataFrames on the 'TO_DATE' column
        for i, df_i in enumerate(combined_data):
            print(f"\n➡️ Processing DataFrame #{i}...")

            if 'TO_DATE' not in df_i.columns:
                print(f"❌ 'TO_DATE' column not found in DataFrame #{i}. Columns: {df_i.columns.tolist()}")
                continue

            print(f"🔍 Before conversion - sample TO_DATE values (#{i}):")
            print(df_i['TO_DATE'].head())

            try:
                df_i['TO_DATE'] = (
                    pd.to_datetime(df_i['TO_DATE'], dayfirst=True)
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                )
            except Exception as e:
                print(f"💥 Error parsing 'TO_DATE' in DataFrame #{i}: {e}")
                continue

            print(f"✅ After conversion - sample TO_DATE values (#{i}):")
            print(df_i['TO_DATE'].head())
        final_combined_data = combined_data[0]
        for df in combined_data[1:]:
            final_combined_data = pd.merge(final_combined_data, df, on='TO_DATE', how='outer')

        print("All datasets have been processed and combined.")
        print(final_combined_data.columns.tolist())
        # Ensure the TO_DATE column is in datetime format
        final_combined_data['TO_DATE'] = pd.to_datetime(final_combined_data['TO_DATE'])
        final_combined_data['Hour'] = final_combined_data['TO_DATE'].dt.hour
        final_combined_data['DayOfWeek'] = final_combined_data['TO_DATE'].dt.dayofweek
        final_combined_data['Month'] = final_combined_data['TO_DATE'].dt.month
        final_combined_data = data_preparer.apply_cyclical_encoding(final_combined_data)
        final_combined_data = fix_nan_values(final_combined_data)


        # Save the combined data to an HDF5 file
        data_preparer.elia_data.df = final_combined_data  # Set the combined data
        data_preparer.save_transformed_data(combined_dataset_name, save_path)

        # Append to the additional_datasets dictionary
        total_datasets[f"{config_name}"] = final_combined_data

    df = data_preparer.join_qh_min_data(total_datasets)
    df['TO_DATE'] = pd.to_datetime(df['TO_DATE'], unit='s', utc=True).dt.tz_localize(None)
    date_index = df.set_index('TO_DATE', inplace=True, drop=False)
    data_preparer.elia_data.df = df  # Set the combined data
    Name = f"total_datasets_data_{start_str}_to_{end_str}"
    data_preparer.save_transformed_data(Name,save_path)

if __name__ == "__main__":
    main()