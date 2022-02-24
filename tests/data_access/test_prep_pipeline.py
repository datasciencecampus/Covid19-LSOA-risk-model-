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
    pp.sort_cols(static_df, 'LSOA11CD')
    pp.sort_cols(static_result, 'LSOA11CD')
    
    pd.testing.assert_frame_equal(static_result, static_df)
    
# read in dynamic test data to use in following two tests
@pytest.fixture
def dynamic_result():
    dynamic_df = factory.get('unit_test_dynamic').create_dataframe()
    
    return dynamic_df

# test default behaviour of read_data 
def test_read_data_dynamic_eng(dynamic_result):
    dynamic_df = pp.read_data('dynamic_test', table_dict = cf.read_data_table_dict, join_col=['LSOA11CD', 'Date'])
    
    # Filter out Welsh LSOAs from target dataframe
    dynamic_result = dynamic_result[dynamic_result['LSOA11CD'].str.startswith('E')]
    
    # sort to allow comparison
    pp.sort_cols(dynamic_result, ['LSOA11CD', 'Date'])
    pp.sort_cols(dynamic_df, ['LSOA11CD', 'Date'])
    
    pd.testing.assert_frame_equal(dynamic_result, dynamic_df)

# test England filter works as expected
def test_read_data_dynamic_all(dynamic_result):
    dynamic_df = pp.read_data('dynamic_test', table_dict = cf.read_data_table_dict, join_col=['LSOA11CD', 'Date'], england_only=False)
    
    # sort to allow comparison
    pp.sort_cols(dynamic_result, ['LSOA11CD', 'Date'])
    pp.sort_cols(dynamic_df, ['LSOA11CD', 'Date'])
    
    pd.testing.assert_frame_equal(dynamic_result, dynamic_df)
    
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
        pp.sort_cols(df, ['date', 'col3'])
    
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
    
    # sort to allow comparison
    pp.sort_cols(df, ['LSOA11CD', 'Date'])
    pp.sort_cols(df_result, ['LSOA11CD', 'Date'])
    
    pd.testing.assert_frame_equal(df, df_result)

# read in target dataframe for join_cases_to_static_data, and input dataframe for derive_week_number and create_test_data
@pytest.fixture
def cases_static():
    
    df = factory.get('unit_test_cases_static').create_dataframe()
    
    pp.sort_cols(df, ['LSOA11CD', 'Date'])
    
    return df
    
def test_join_cases_to_static_data(cases_static):
    # input dataframe
    static_df = factory.get('unit_test_static_for_cases').create_dataframe()

    df = pp.join_cases_to_static_data(static_df, table='unit_test_cases')
    
    # sort to allow comparison
    pp.sort_cols(df, ['LSOA11CD', 'Date'])
    
    pd.testing.assert_frame_equal(df, cases_static)
    

# target dataframe for derive_week_number
# and input dataframe for join_tranches_mobility_data
@pytest.fixture
def cases_static_week():
    
    df = factory.get('unit_test_cases_static_week').create_dataframe()
    
    pp.sort_cols(df, ['LSOA11CD', 'Date'])
    
    return df

def test_derive_week_number(cases_static, cases_static_week):
    
    # Ensure data types are as expected
    cases_static['Date'] = pd.to_datetime(cases_static['Date'].dt.date)
    
    df = pp.derive_week_number(cases_static)
    
    pd.testing.assert_frame_equal(df, cases_static_week)
    
# input for joint_tranches_mobility_data and create_test_data
@pytest.fixture
def deimos_footfall():
    
    df = factory.get('unit_test_deimos').create_dataframe()
    
    df['Date'] = pd.to_datetime(df['Date'].dt.date)
    
    pp.sort_cols(df, ['LSOA11CD', 'Date'])
    
    return df

# target dataframe for join_tranches_mobility_data
# and input for convert_units
@pytest.fixture
def tranches_mobility():
    
    df = factory.get('unit_test_deimos_cases').create_dataframe()
    
    pp.sort_cols(df, ['LSOA11CD', 'Date'])
    
    return df

# target dataframe for convert_units 
# and input for create_time_tranches and create_test_data
@pytest.fixture
def convert_units_df():
    
    df = factory.get('unit_test_convert_unit').create_dataframe()
    
    pp.sort_cols(df, ['LSOA11CD', 'Date'])
    
    return df

# target dataframe for create_time_tranches
# and input for derive_tranche_order
@pytest.fixture
def time_tranche_df():
    pass
    
def test_join_tranches_mobility_data(cases_static_week, deimos_footfall, tranches_mobility):
    
    df = pp.join_tranches_mobility_data(cases_static_week, deimos_footfall)
    
    pd.testing.assert_frame_equal(df, tranches_mobility)
    
# test default behaviour of function
def test_convert_units(tranches_mobility, convert_units_df):
    
    df = pp.convert_units(tranches_mobility,
                          'meat_and_fish_processing',
                          0.1)
    
    pd.testing.assert_frame_equal(df, convert_units_df)
    
# test non-default parameter behaviour
def test_convert_units_alt(tranches_mobility):
    
    # read in target dataframe
    df_result = factory.get('unit_test_convert_unit_alt').create_dataframe()
    
    pp.sort_cols(df_result, ['LSOA11CD', 'Date'])
    
    df = pp.convert_units(tranches_mobility,
                          'meat_and_fish_processing',
                          0.1,
                          new_colname='meat_and_fish_processing_alt')
    
    pd.testing.assert_frame_equal(df, df_result)
    

def test_create_test_data(convert_units_df, cases_static, deimos_footfall):
    pass

def test_create_time_tranches(convert_units_df, time_tranche_df):
    pass

def test_derive_tranche_order(time_tranche_df):
    # read in target here, not used elsewhere
    pass

    
    
    