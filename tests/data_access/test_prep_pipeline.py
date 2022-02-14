import os
import sys

import pytest

import pandas as pd

sys.path.append(os.getcwd() + '/src/data_access')

import prep_pipeline as pp

import config as cf
from data_access.data_factory import DataFactory as factory


def test_read_data_static():
    # read in dataframe to test function against
    static_result = factory.get('unit_test_static').create_dataframe()
    
    static_df = pp.read_data('static_test', table_dict = cf.read_data_table_dict)
    
    # reading in the dataframe will create a different datatype to the processing
    static_df['geometry'] = static_df['geometry'].astype(str)
    static_result['geometry'] = static_result['geometry'].astype(str)
    
    # sort values and reindex to allow assertion of equality
    static_df.sort_values(by='LSOA11CD', inplace=True)
    static_df.reset_index(drop=True, inplace=True)

    static_result.sort_values(by='LSOA11CD', inplace=True)
    static_result.reset_index(drop=True, inplace=True)
    
    pd.testing.assert_frame_equal(static_result, static_df)
    
def test_geo_merge_static():
    geom_df = factory.get('LSOA_2011').create_dataframe()
    
    geom_df = pp.geo_merge(geom_df)
    
    # dataframe to test function against
    geom_result = factory.get('unit_test_geometry').create_dataframe()
    
    # convert to strings as geom_test geometry column will be read in as object
    # whereas geom_df geometry column will be a geometry type
    geom_df['geometry'] = geom_df['geometry'].astype(str)
    geom_result['geometry'] = geom_result['geometry'].astype(str)
    
    pd.testing.assert_frame_equal(geom_df, geom_result)
    
    # I think geo merge should be replaced by a lookup for the Areas 
    # they're not going to change, and the current calculation uses geographic CRS which is slightly inaccurate when calculating areas
    # and the effort required to get an accurate projection is non-trivial

def test_normalise_data():
    # input dataframe
    df = factory.get('unit_test_normalise').create_dataframe()
    
    # target dataframe
    df_result = factory.get('unit_test_normalise_result').create_dataframe()
    
    df_normalised = pp.normalise_data(df.copy(), 'test', cf.normalise_dic)
    
    # check that output of normlise_data function is as expected
    pd.testing.assert_frame_equal(df_normalised, df_result)
    
def test_ffill_cumsum():
    # input dataframe
    df = factory.get('unit_test_ffill_df').create_dataframe()
    
    # target dataframe for forward filling one column only
    df_one_result = factory.get('unit_test_ffill_1').create_dataframe()
    
    # target dataframe for forward filling two columns
    df_two_result = factory.get('unit_test_ffill_2').create_dataframe()
    
    df_one = pp.ffill_cumsum(df.copy(), col_list = 'col1', sort_col = 'date', group_col ='col3')
    df_two = pp.ffill_cumsum(df.copy(), col_list = ['col1', 'col2'], sort_col = 'date', group_col ='col3')
    
    # sort and reindex as GCP tables are automatically sorted upon saving
    for df in [df_one, df_two, df_one_result, df_two_result]:
        df.sort_values(by=['date', 'col3'], inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    pd.testing.assert_frame_equal(df_one, df_one_result)
    
    pd.testing.assert_frame_equal(df_two, df_two_result)
    
def test_sum_features():
    # input dataframe
    df = factory.get('unit_test_sum_features').create_dataframe()
    
    # target dataframe
    df_result = factory.get('unit_test_sum_features_result').create_dataframe()
    
    df = pp.sum_features(df, dic = cf.sum_dic)
    
    pd.testing.assert_frame_equal(df, df_result)
    
def test_apply_timelag():
    # input dataframes
    dynamic_df = factory.get('unit_test_timelag_dynamic').create_dataframe()
    dynamic_df_norm = factory.get('unit_test_timelag_dynamic_norm').create_dataframe()
    
    # target dataframe
    df_result = factory.get('unit_test_timelag_result').create_dataframe()
    
    df = pp.apply_timelag(dynamic_df, dynamic_df_norm, save_results=False)
    
    # fix datatypes and order of two dataframes is the same to ensure comparison
    df['Date'] = df['Date'].apply(lambda x: x.replace(tzinfo=None))
    df_result['Date'] = df_result['Date'].apply(lambda x: x.replace(tzinfo=None))
    
    df.sort_values(by=['LSOA11CD', 'Date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df_result.sort_values(by=['LSOA11CD', 'Date'], inplace=True)
    df_result.reset_index(drop=True, inplace=True)
    
    pd.testing.assert_frame_equal(df, df_result)