from datetime import datetime

import pandas as pd
import yaml
import os
import pandas as pd
import copy

class EliaPowerSystemData:
    """
       A class to manage, transform, and process Elia power system data into a custom database format.

       Attributes:
           dataset_name (str): The name of the dataset being processed, this is an identifier
            that signals from which endpoint in the ELIA OPEN DATA API this data comes. Determines dataset-specific transformations.
           df (pd.DataFrame): The DataFrame that holds the relevant data.
           format (str): The format of the data. Must be either "Elia" (original format) or "Own" (transformed format).
           transform_params (dict): Transformation parameters loaded from a YAML file, including columns to keep and renaming rules.
       """

    def __init__(self, dataset_name, df: pd.DataFrame, format: str):
        """
                Initialize an instance of the EliaPowerSystemData class.

                Args:
                    dataset_name (str): The name of the dataset being processed.
                    df (pd.DataFrame): The input DataFrame in Elia format.
                    format (str): The format of the data. Must be either "Elia" or "Own".
                    transform_params (dict): Parameters specifying the transformation from Elia to Own format. Includes columns_kept and rename_dict.

                Raises:
                    ValueError: If the provided format is not "Elia" or "Own".
                """
        dict_params = yaml.safe_load(open(os.path.join(r"C:\Users\jds\OneDrive - KU Leuven\Ku Leuven\Master Semester 5\Thesis\Elia", "params_elia_to_own_db.yml")))
        self.dataset_name = dataset_name
        print(dict_params["transform_params"][dataset_name])
        try:
            self.transform_params = dict_params["transform_params"][dataset_name]
        except KeyError:
            raise KeyError(
                f"dataset_name: {dataset_name} not recognized in transformation parameters file. Possible dataset_names are: "
                f"{[n for n in dict_params['transform_params'].keys()]}")
        self.df = df
        self.format = format

    @property
    def format(self):
        """
        str: The current format of the data ("Elia" or "Own").
        """
        return self._format

    @format.setter
    def format(self, value):
        """
        Validates and sets the format of the data.

        Args:
            value (str): The new format. Must be either "Elia" or "Own".

        Raises:
            ValueError: If the provided format is not "Elia" or "Own".
        """
        if value in ["Elia", "Own"]:
            self._format = value
        else:
            raise ValueError(f"Format must be either Elia or Own, currrently :{value}")

    def transform_from_elia_format_to_db_format(self):
        """
        Transforms the DataFrame from Elia's format to the custom database format.

        Workflow:
            1. Validates that the current format is "Elia".
            2. Filters and deep-copies only relevant columns.
            3. Applies dataset-specific transformations by calling `_perform_dataset_specific_operations`.
            4. Renames columns using a predefined dictionary.
            5. Adjusts datetime columns to timezone "Etc/GMT-1".
            6. Adds a `TO_DATE` column by adding 15 minutes to the datetime. ???WHY???
            7. Sets the `datetime` column as the DataFrame index.
            8. Updates the `format` attribute to "Own".

        Raises:
            ValueError: If the current format is not "Elia".
            KeyError: If required columns are missing in the DataFrame.
        """
        assert (self.format == "Elia")
        print(self.df.head(20))
        # Extract the dataset-specific list of columns that will be kept and the dictionary for renaming
        cols_kept = self.transform_params["columns_kept"]
        rename_dict = self.transform_params["rename_dict"]

        # Keep only the columns that you want and make a copy
        df_relevant = self.df[cols_kept].copy()

        df_relevant = self._perform_dataset_specific_operations(df_relevant)

        # Rename columns
        df_relevant = df_relevant.rename(rename_dict, axis=1)
        print("df_relevant",df_relevant.columns.tolist())
        # Datetime operations
        df_relevant['datetime'] = pd.to_datetime(df_relevant['datetime'])
        if df_relevant['datetime'].dt.tz is None:
            df_relevant['datetime'] = df_relevant['datetime'].dt.tz_localize('Etc/GMT-1')
        df_relevant['datetime'] = df_relevant['datetime'].dt.tz_convert('Etc/GMT-1')

        df_relevant["TO_DATE"] = pd.to_datetime(df_relevant['datetime'])
        df_relevant.set_index("datetime", inplace=True)
        # if not df_relevant.index.is_unique:
        #     print("⚠️ Index contains duplicate datetimes!")

        self.df = df_relevant
        self.format = "Own"

    def _perform_dataset_specific_operations(self, df_relevant):
        """
                Applies dataset-specific transformations to the DataFrame. Before the renaming based on rename_dict

                Args:
                    df (pd.DataFrame): The DataFrame to transform.

                Returns:
                    pd.DataFrame: The transformed DataFrame.

                Raises:
                    ValueError: If the `dataset_name` is not recognized.

                Dataset-Specific Logic:
                    - imba_price_qh_hist_old: Adds `ACE` column as the sum of `netregulationvolume` and `systemimbalance`.
                    - imba_price_qh_hist_old: Adds `NRV` column as the difference of `ace` and `systemimbalance`.
                    - pv_prod_hist: filters based on 'region' column for value Belgium, then drops 'region' column
                    - wind_prod_hist: Groups by `datetime` and sums values.8
                    - DA_sched_FT_old: Pivots data on `fuelcode` with `dayaheadgenerationschedule` as values.
                    - DA_sched_FT Pivots data on `fueltypepublication` with `dayaheadschedule` as values.
                    - act_prod_FT_old: Pivots data on `fuelcode` with `generatedpower` as values.
                    - act_prod_FT: Pivots data on `fueltypepublication` with `generatedpower` as values.
                    - load_hist: No specific transformation.
                    - physical_flow_by_border: Pivots data on `controlarea` with `physicalflowatborder` as values, and adds `Net_Position`.
                """
        # Do the dataset-specific operations
        if self.dataset_name == "imba_price_qh_hist_old":
            # df_relevant["ACE"] = df_relevant["netregulationvolume"] + df_relevant["systemimbalance"]
            # ACE can be added if you want
            pass

        elif self.dataset_name == "imba_price_qh_hist":
            df_relevant["NRV"] = df_relevant["ace"] - df_relevant["systemimbalance"]

        elif self.dataset_name == "imba_price_qh_min_old":
            pass

        elif self.dataset_name == 'imbalance_tariff':
            pass

        elif self.dataset_name == "pv_prod_hist":
            df_relevant = df_relevant[df_relevant["region"] == "Belgium"].copy()
            df_relevant.drop("region", axis=1, inplace=True)

        elif self.dataset_name == "wind_prod_hist":
            df_relevant = df_relevant.groupby("datetime", as_index=False).sum()

        elif self.dataset_name == "intraday_impl_net_pos_BE":
            pass

        elif self.dataset_name == "intraday_impl_net_pos_GB":
            df_relevant["adjusted_schedule"] = df_relevant["nettransfercapacity"]*df_relevant["direction"].apply(lambda x: -1 if x== "Import" else 1)
            df_relevant.drop("direction", axis=1, inplace=True)
            df_relevant.drop("nettransfercapacity", axis=1, inplace=True)
            df_relevant = df_relevant.groupby("datetime", as_index=False).sum()
            df_relevant.rename(columns={"adjusted_schedule": "nettransfercapacity"}, inplace=True)

        elif self.dataset_name == "day_ahead_impl_net_pos_sum":
            # Step 1: Add signed schedule column (Import = negative)
            df_relevant["adjusted_schedule"] = df_relevant["commercialschedule"] * df_relevant["direction"].apply(
                lambda x: -1 if x == "Import" else 1)
            df_relevant["signed_schedule"] = df_relevant["adjusted_schedule"]

            # Step 2: Total net schedule per timestamp (all countries combined)
            df_total = df_relevant.groupby("datetime", as_index=False)["adjusted_schedule"].sum()
            df_total.rename(columns={"adjusted_schedule": "commercialschedule"}, inplace=True)

            # Step 3: Net schedule per country (Import = negative, Export = positive)
            df_country_net = df_relevant.pivot_table(
                index="datetime",
                columns="country",
                values="signed_schedule",
                aggfunc="sum"
            ).fillna(0)
            df_country_net.columns = [f"XB_DA_NET_{c.replace(' ', '')}" for c in df_country_net.columns]

            # Step 4: Import per country (always positive)
            df_import = df_relevant[df_relevant["direction"] == "Import"].pivot_table(
                index="datetime",
                columns="country",
                values="commercialschedule",
                aggfunc="sum"
            ).fillna(0)
            df_import.columns = [f"XB_DA_IMP_{c.replace(' ', '')}" for c in df_import.columns]

            # Step 5: Export per country (always positive)
            df_export = df_relevant[df_relevant["direction"] == "Export"].pivot_table(
                index="datetime",
                columns="country",
                values="commercialschedule",
                aggfunc="sum"
            ).fillna(0)
            df_export.columns = [f"XB_DA_EXP_{c.replace(' ', '')}" for c in df_export.columns]

            # Step 6: Combine all parts into final df
            df_final = df_total.set_index("datetime").join([df_country_net, df_import, df_export]).reset_index()

            # Step 7: Replace df_relevant with the final version
            df_relevant = df_final

        elif self.dataset_name == "final_com_sched":
            # Step 1: Total export per datetime
            df_export_total = df_relevant.groupby("datetime", as_index=False)["export_value"].sum()
            df_export_total.rename(columns={"export_value": "XB_ID_EXP"}, inplace=True)

            # Step 2: Total import per datetime
            df_import_total = df_relevant.groupby("datetime", as_index=False)["import_value"].sum()
            df_import_total.rename(columns={"import_value": "XB_ID_IMP"}, inplace=True)

            # Step 3: Pivot import values per country
            df_import = df_relevant.pivot_table(
                index="datetime",
                columns="country",
                values="import_value",
                aggfunc="sum"
            ).fillna(0)
            df_import.columns = [f"XB_ID_IMP_{c.replace(' ', '')}" for c in df_import.columns]

            # Step 4: Pivot export values per country
            df_export = df_relevant.pivot_table(
                index="datetime",
                columns="country",
                values="export_value",
                aggfunc="sum"
            ).fillna(0)
            df_export.columns = [f"XB_ID_EXP_{c.replace(' ', '')}" for c in df_export.columns]

            # Step 5: Merge all data
            df_total = pd.merge(df_export_total, df_import_total, on="datetime")
            df_final = df_total.set_index("datetime").join([ df_import, df_export]).reset_index()

            # Step 6: Store final result
            df_relevant = df_final


        elif self.dataset_name == "day_ahead_impl_net_pos_GB":
            df_relevant["adjusted_schedule"] = df_relevant["nettransfercapacity"]*df_relevant["direction"].apply(lambda x: -1 if x== "Import" else 1)
            df_relevant.drop("direction", axis=1, inplace=True)
            df_relevant.drop("nettransfercapacity", axis=1, inplace=True)
            df_relevant = df_relevant.groupby("datetime", as_index=False).sum()
            df_relevant.rename(columns={"adjusted_schedule": "nettransfercapacity"}, inplace=True)

        elif self.dataset_name in ["si_forecast_qh_0", "si_forecast_qh_1"]:
            # Standardize datetime to minute precision and sort
            df_relevant["datetime"] = pd.to_datetime(df_relevant["datetime"], errors="coerce").dt.floor("min")
            df_relevant = df_relevant.sort_values("datetime").reset_index(drop=True)

            # Identify all duplicate datetime rows (keep=False marks both)
            duplicate_mask = df_relevant["datetime"].duplicated(keep=False)
            duplicate_rows = df_relevant[duplicate_mask]

            # Print duplicate summary and rows
            if not duplicate_rows.empty:
                print("⚠️ Duplicate datetime values found:")
                print(duplicate_rows["datetime"].value_counts())
                print("\n🧾 Full rows with duplicate datetimes:")
                print(duplicate_rows.iloc[:, 2].to_string(index=False))  # no column name, no index
            else:
                print("✅ No duplicate datetime values found.")

            #
            # # Check for gaps in minute-level timestamps and shift rows forward if a duplicate exists later
            # i = 0
            # while i < len(df_relevant) - 1:
            #     current_time = df_relevant.loc[i, "datetime"]
            #     expected = current_time + pd.Timedelta(minutes=1)
            #     next_time = df_relevant.loc[i + 1, "datetime"]
            #
            #     if next_time != expected:
            #         j = i + 1
            #         temp_shifted_rows = []
            #
            #         # Shift following rows forward to fill the gap, until a duplicate timestamp is reached
            #         while j < len(df_relevant):
            #             new_time = expected + pd.Timedelta(minutes=(j - i - 1))
            #             if new_time in df_relevant["datetime"].values:
            #                 break
            #             temp_row = df_relevant.loc[j].copy()
            #             temp_row["datetime"] = new_time
            #             temp_shifted_rows.append(temp_row)
            #             j += 1
            #
            #         # Replace original rows with the shifted versions
            #         for k, row in enumerate(temp_shifted_rows):
            #             df_relevant.loc[i + 1 + k] = row
            #
            #         i += len(temp_shifted_rows)  # Skip to the end of the adjusted section
            #     else:
            #         i += 1

            # Final resort to ensure consistent order after adjustments
            df_relevant = df_relevant.sort_values("datetime").reset_index(drop=True)

        elif self.dataset_name == "DA_sched_FT_old":
            df_relevant = df_relevant.groupby("datetime", as_index=False).sum()



        elif self.dataset_name == "load_hist":
            pass

        elif self.dataset_name == "physical_flow_by_border":
            df_relevant = df_relevant.pivot(values="physicalflowatborder", columns="controlarea",
                                            index="datetime")
            df_relevant.columns = [f"XB_RT_{c.replace(' ', '')}" for c in df_relevant.columns]

            df_relevant["XB_RT"] = df_relevant.sum(axis=1)
            df_relevant["datetime"] = df_relevant.index

        elif self.dataset_name == "indiv_incr_bal_en_bids_hist_old":
            df_relevant["datetime"] = pd.to_datetime(df_relevant["datetime"])

            # STEP 2: Aggregate total bidding volume and average price
            base_features = df_relevant.groupby("datetime").agg(
                BID_V=( "energybidvolume", "sum"),
                BID_P=("energybidmarginalprice", "mean")
            )
            # STEP 3: Aggregate bid volume by reserve type (aFRR and mFRR)
            reserve_volumes = df_relevant.groupby(["datetime", "balancingreserve"])["energybidvolume"].sum().unstack(fill_value=0)

            # STEP 4: Rename columns for clarity
            reserve_volumes = reserve_volumes.rename(columns={
                "aFRR": "aFRR_volume",
                "mFRR": "mFRR_volume"
            })

            # STEP 5: Combine all features into a single DataFrame
            features_df = base_features.join(reserve_volumes, how="left").reset_index()

            # STEP 6: Ensure both aFRR and mFRR columns exist (fill with 0 if missing)
            for col in ["aFRR_volume", "mFRR_volume"]:
                if col not in features_df.columns:
                    features_df[col] = 0

        elif self.dataset_name == "act_bal_en_vol_qh_old":
            # Replace string "null" values and blank entries with proper NaN
            df_relevant.replace(['null', 'NaN', 'nan', '', 'None'], pd.NA, inplace=True)
            df_relevant.fillna(0, inplace=True)


        elif self.dataset_name == "act_bal_en_volumes":
            self.df.fillna(0, inplace=True)

        else:
            print("Type of data set not in list for dataset-specific transformations, returning original")

        return df_relevant





if __name__ == "__main__":
    elia_power = elia_power_system_data

