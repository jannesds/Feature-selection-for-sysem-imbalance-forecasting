import requests
import datetime as dt
import pandas as pd
import json
import yaml
import os
from datetime import datetime, timedelta

import pytz
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

class EliaDataFetcher:
    def __init__(self):
        dict_params = yaml.safe_load(
            open(os.path.join(r"C:\Users\jds\OneDrive - KU Leuven\Ku Leuven\Master Semester 5\Thesis\Elia", "params_elia_API.yml")))
        self.BASE_URL = dict_params["base_url"]
        self.ENDPOINT = dict_params["endpoint"]
        self.DATASET_MAP = dict_params["names_map"]
        pass
    def _execute_query(self,dataset: str, params: dict) -> pd.DataFrame:
        """Executes an API query to retrieve data from the specified dataset.

        Args:
            dataset (str): The dataset identifier for the specific Elia API dataset.
            params (dict): Query parameters for the request, typically including filters.

        Returns:
            pd.DataFrame: A DataFrame containing the queried data.

        Raises:
            HTTPError: If the request fails, raises an HTTPError with the response status.
        """
        response = requests.get(self.BASE_URL + self.ENDPOINT % dataset, params=params)
        if response.status_code == 200:
            json_data = json.loads(response.text)
            #print(response.text)
            df = pd.json_normalize(json_data)
            return df
        else:
            response.raise_for_status()

    @staticmethod
    def _convert_params_to_where_query_string(start:datetime,end:datetime,field:str="datetime") -> str:
        """Constructs a query string for date filtering.
        Necessary for the API "where" function, endpoint retrieval

        Args:
            **kwargs: Contains 'start' and 'end' as datetime objects representing
                      the date range for filtering.

        Returns:
            str: A formatted query string for the specified datetime range.

        Example:
            If start is '2022-11-20' and end is '2022-11-21', the output is:
            "datetime IN [date'2022-11-20 00:00:00'..date'2022-11-21 23:59:59'["
        """

        def adjust_time(time: datetime):
            if time.tzinfo:
                # shift timestamps by the local offset (assumes we want to query as UTC+0)
                offset_diff = time.utcoffset()
                adjusted_time = time - offset_diff
            else:
                # if no timezone info is provided, shift by 1 hour to simulate UTC+1 correction
                adjusted_time = time - timedelta(hours=1)
            return adjusted_time

        if start and end:
            start_ = adjust_time(start)
            end_ = adjust_time(end)
            return (
                f"{field} IN [date'{start_.strftime(DATETIME_FORMAT)}'"
                f"..date'{end_.strftime(DATETIME_FORMAT)}'["
            )
        else:
            return None


    def get_elia_data_as_df(self, dataset_name: str, start: dt.datetime, end: dt.datetime):
        """Retrieves and returns data from the Elia dataset within a specified date range.

        Args:
            dataset_name (str): The identifier for the dataset to query.
            start (datetime): The start datetime for the data query range.
            end (datetime): The end datetime for the data query range.

        Returns:
            pd.DataFrame: A DataFrame containing data from the dataset within the specified range.
        """
        try:
            dataset = self.DATASET_MAP[dataset_name]
        except KeyError:
            raise KeyError(
                f"dataset_name: {dataset_name} not recognized. Possible dataset_names are: {[n for n in self.DATASET_MAP.keys()]}")
        field = "datetime"
        # Only override field for forecast datasets, no datetime given
        if dataset_name in ["si_forecast_qh_0", "si_forecast_qh_1"]:
            field = "predictiontimeutc"
        else:
            field = "datetime"

        where_filter = self._convert_params_to_where_query_string(start, end, field)

        params = {"where": where_filter}
        response_json = self._execute_query(dataset, params)
        return response_json
    #   def get_elia_data_as_power_system_data(self,dataset_name:str,start:dt.datetime,end:dt.datetime):

# Example usage:
if __name__ == "__main__":
    fetcher = EliaDataFetcher()
    start = dt.datetime(2023, 1, 1)
    end = dt.datetime(2023, 12, 31,23)

    dataset_name = "imba_price_qh_hist_old"
    try:
        data_df = fetcher.get_elia_data_as_df(dataset_name,start, end)
        print(data_df.head())
    except Exception as e:
        print(f"An error occurred: {e}")