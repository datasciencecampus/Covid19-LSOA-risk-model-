import os
import sys

import pytest

import pandas as pd

sys.path.append(os.getcwd() + '/src/data_access')

from prep_pipeline import read_data, geo_merge, normalise_data, ffill_cumsum, apply_timelag

import config as cf
from data_access.data_factory import DataFactory as factory


def test_read_data_static():
    static_test = read_data('static_test', table_dict = cf.read_data_table_dict)
    
    # read in dataframe to test function against
    static_df = factory.get('unit_test_static').create_dataframe()
    
    # reading in the dataframe will create a different datatype to the processing
    static_test['geometry'] = static_test['geometry'].astype(str)
    static_df['geometry'] = static_df['geometry'].astype(str)
    
    # sort values and reindex to allow assertion of equality
    static_test.sort_values(by='LSOA11CD', inplace=True)
    static_test.reset_index(drop=True, inplace=True)

    static_df.sort_values(by='LSOA11CD', inplace=True)
    static_df.reset_index(drop=True, inplace=True)
    
    pd.testing.assert_frame_equal(static_df, static_test)
    
def test_geo_merge_static():
    geom_df = factory.get('LSOA_2011').create_dataframe()
    
    geom_df = geo_merge(geom_df)
    
    geom_test = factory.get('unit_test_geometry').create_dataframe()
    
    pd.testing.assert_frame_equal(geom_df, geom_test)
    
    # I think geo merge should be replaced by a lookup for the Areas 
    # they're not going to change, and the current calculation uses geographic CRS which is slightly inaccurate when calculating areas
    # and the effort required to get an accurate projection is non-trivial

