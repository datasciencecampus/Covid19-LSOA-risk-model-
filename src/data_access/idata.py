from abc import ABCMeta, abstractmethod
from google.cloud import bigquery
import pandas as pd

class IData(metaclass=ABCMeta):

    "The Data Access Object Interface"
    @staticmethod
    @abstractmethod
    def create_dataframe() -> pd.DataFrame:
        "An abstract interface method"
        
    @property
    def client(self):
        return bigquery.Client()
