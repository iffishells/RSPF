import pandas as pd
import json


class DataLoader:

    def __init__(self):
        self.columns_name = ['Date','UNITS', 'DAILY_AQI_VALUE', 'Site Name',
                             'AQS_PARAMETER_DESC', 'COUNTY', 'SITE_LATITUDE',
                             'SITE_LONGITUDE']

    def save_json(self, data: dict):
        """

        :param data:
        """
        if isinstance(data, dict):

            with open("Datasets/meta_information.json", "w") as json_file:
                json.dump(data, json_file)
        else:
            print('Dataset is not found')

    def save_meta_information_of_data(self, df: pd.DataFrame):
        """

        :param df:
        """
        meta_information = {
            "columns_name": list(self.columns_name),
            "city_names": list(set(df['Site Name'])),
            'country_names': list(set(df['COUNTY'])),
            'air-pollutant': list(set(df['AQS_PARAMETER_DESC']))
        }
        self.save_json(meta_information)

    def __call__(self, file_path=None):
        if file_path is not None:
            df = pd.read_csv(file_path, usecols=self.columns_name)
            self.save_meta_information_of_data(df)
            print("file_path : ",file_path)
            df.to_csv('Datasets/preprocessed_data/air-pollution_preprocessed_data.csv', index=False)

            return df
