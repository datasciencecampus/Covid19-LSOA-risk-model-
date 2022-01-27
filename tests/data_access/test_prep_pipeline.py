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
    
    # convert to strings as geom_test geometry column will be read in as object
    # whereas geom_df geometry column will be a geometry type
    geom_df['geometry'] = geom_df['geometry'].astype(str)
    geom_test['geometry'] = geom_test['geometry'].astype(str)
    
    pd.testing.assert_frame_equal(geom_df, geom_test)
    
    # I think geo merge should be replaced by a lookup for the Areas 
    # they're not going to change, and the current calculation uses geographic CRS which is slightly inaccurate when calculating areas
    # and the effort required to get an accurate projection is non-trivial

def test_normalise_data():
    # input dataframe
    df = factory.get('unit_test_normalise').create_dataframe()
    
    # target dataframe
    df_result = factory.get('unit_test_normalise_result').create_dataframe()
    
    df_normalised = normalise_data(df.copy(), 'test', cf.normalise_dic)
    
    # check that output of normlise_data function is as expected
    pd.testing.assert_frame_equal(df_normalised, df_result)
    
def test_ffill_cumsum():
    # input dataframe
    df = factory.get('unit_test_ffill_df').create_dataframe()
    
    # target dataframe for forward filling one column only
    df_one_test = factory.get('unit_test_ffill_1').create_dataframe()
    
    # target dataframe for forward filling two columns
    df_two_test = factory.get('unit_test_ffill_2').create_dataframe()
    
    df_one = ffill_cumsum(df.copy(), col_list = 'col1', sort_col = 'date', group_col ='col3')
    df_two = ffill_cumsum(df.copy(), col_list = ['col1', 'col2'], sort_col = 'date', group_col ='col3')
    
    # sort and reindex as GCP tables are automatically sorted upon saving
    for df in [df_one, df_two, df_one_test, df_two_test]:
        df.sort_values(by=['date', 'col3'], inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    pd.testing.assert_frame_equal(df_one, df_one_test)
    
    pd.testing.assert_frame_equal(df_two, df_two_test)
    
    