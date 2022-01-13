# Import Packages
import os
import sys
from functools import reduce

import pandas as pd
import numpy as np
import pandas_gbq
import geopandas as gpd

# Import from local data files
current_path = os.path.abspath('.')
sys.path.append(os.path.dirname(current_path))

# this may or may not break normal useage of this module...
sys.path.append(os.getcwd() + '/src')

from data_access.data_factory import DataFactory as factory
from utils import data as dt
from utils import config as cf
import utils.dynamic as dyn

#############################

def read_data(table_type, table_dict = cf.data_tables, join_col = 'LSOA11CD', england_only = True):
    '''
    Read in and join a list of data tables on a common column.
    
    :param table_type: Type of tables to read in, must be one of 'static' or 'dynamic'. This is fed into the table_dict parameter and will determine the list of tables to read in. 
    :type table_type: string
    
    :param table_dict: Dictionary of tables to read in where the values are lists of strings of table names for this function to read in. These strings feed into the DataFactory.get() function within src/data_access/data_factory.py. Defaults to the data_tables variable within src/data_access/config.py
    :type table_dict: dictionary
    
    :param join_col: The common column on which to join ALL of the tables. Any common columns not specified here will be dropped from the right table in each join. Defaults to 'LSOA11CD'
    :type join_col: string or list of strings
    
    :param england_only: Whether to filter to English LSOAs only. Default True. 
    :type england_only: bool
    
    :return: DataFrame of joined datasets
    :rtype: Pandas DataFrame
    '''
    
    table_list = table_dict[table_type]
    
    df_final = pd.DataFrame()
    
    for table in table_list:
        df = factory.get(table).create_dataframe()
        if len(df_final) == 0:
            df_final = df
        else:
            df_final = df_final.merge(df, on=join_col, how='outer', suffixes=['', '_drop'])
            
    drop_cols = [col for col in df_final.columns if col.endswith('_drop')]
    df_final.drop(columns=drop_cols, inplace=True)
    
    if england_only:
         df_final = df_final[df_final['LSOA11CD'].str.startswith('E')]
            
    return df_final

def geo_merge(df, geo_col = 'geometry'):
    '''
    Add on geographic information with an 'Area' column, and combine intermediate travel clusters. 
    
    :param df: Input dataframe, from read_data function. Requires 'travel_cluster' column for travel cluster grouping function.
    :type df: Pandas DataFrame
    
    :param geo_col: Name of column from which to calculate the area
    :type geo_col: string
    
    :return: DataFrame with Area and combined and renamed travel cluster groupings
    :rtype: Pandas DataFrame
    
    '''
    
    # merge geo data to get areas of each LSOA
    gdf = gpd.GeoDataFrame(df, crs="EPSG:27700", geometry=df[geo_col])
    gdf.crs
    gdf['Area'] = gdf[geo_col].area/10**6

    # combine and rename travel clusters 
    df = dt.combining_and_remap_travel_cluster(gdf)
    
    return df

def normalise_data(df, flag, dic = cf.features_dict):
    
    '''
    Apply the normalise function from src/utils/data.py, which normalises 1 or more columns by the sum of those columns, or by the value of another supplied column, depending on whether the 'by' parameter is None or not. The details for each call to this function is held in the config file in the features_dict dictionary. The values to be used are dictionaries themselves and have a 'flag' key with value 'static', 'dynamic_norm', or 'dynamic', denoting where they are used.
    
    :param df: Input DataFrame, resulting from calls to the read_data and geo_merge functions.
    :type df: Pandas DataFrame
    
    :param flag: Which type of features to select - one of 'static', 'dynamic', or 'dynamic_norm'. 
    :type flag: string
    
    :param dic: Dictionary containing the normalise function call information. Defaults to src/utils/config.py features_dict.
    :type dic: dictionary
    
    :return: DataFrame with normalised columns. Original columns will be changed if no suffix is supplied in config file.
    :rtype: Pandas DataFrame
    
    '''
    
    keys = [key for key in dic.keys() if type(dic[key]) == dict and dic[key]['flag'] == flag]
    
    for key in keys:
        
        if not dic[key]['by']:
            norm_by = None
        else:
            norm_by = df[dic[key]['by']]
           
        df = dt.normalise(df, dic[key]['columns'], by = norm_by, suffix=dic[key]['suffix'])
    
    return df

def ffill_cumsum(df, col_list, sort_col='Date', group_col = 'LSOA11CD'):
    
    '''
    Perform forward fill on cumulative sum columns. This is needed because cumulative sums have been performed on the source data which does not have entries for every day.
    
    :param df: Input dataset, typically resulting from calls to read_data and other subsequent preprocessing functions.
    :type df: Pandas DataFrame
    
    :param col_list: List of columns to perform forward fill on
    :type col_list: List of strings
    
    :param sort_col: Column(s) by which to sort the dataframes prior to forward filling, defaults to 'Date'. 
    :type sort_col: string, or list of strings
    
    :param group_col: Column(s) to group by prior to forward filling. Defaults to 'LSOA11CD'.
    :type group_col: string, or list of strings
    
    :return: DataFrame with fixed cumulative sums and 0s in place of any starting NaNs should there be any.
    :rtype: Pandas DataFrame
    
    '''
    
    df.sort_values(by=sort_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.replace(0, np.nan, inplace=True)
    
    df[col_list] = df.groupby(group_col)[col_list].ffill().fillna(0).astype(float)
    
    df.fillna(0, inplace=True)
    
    return df

def apply_timelag(dynamic_df, dynamic_df_norm):
    '''
    Calculate appropriate time lag values to use and apply to dynamic dataset.
    
    :param dynamic_df: dynamic dataframe as produced by the dynamic dataset preprocessing, also accessed through DataFactory.get('lsoa_dynamic').create_datarame().
    :type dynamic_df: Pandas DataFrame
    
    :param dynamic_df_norm: normalised dynamic dataframe as produced by the dynamic dataset preprocessing, also accessed through DataFactory.get('dynamic_raw_norm_chosen_geo').create_dataframe().
    :type dynamic_df: Pandas DataFrame
    
    :return: Dataframe ready for modelling
    :rtype: Pandas DataFrame
    '''
    
    dynamic_df['Date'] = pd.to_datetime(dynamic_df['Date'])
    dynamic_df_norm['Date'] = pd.to_datetime(dynamic_df_norm['Date'])
    
    dynamic_df['Country'] = 'England'
    
    lag_granularity = cf.chosen_granularity_for_lag
    
    dynamic_df_norm_split = [pd.DataFrame(y) for x, y in \
                                      dynamic_df_norm.groupby(lag_granularity, as_index=False)]
    
    tl = dyn.TimeLag()
    
    # calculate and store lag values for mobility
    lag_values_mobility={}

    for c in cf.mobility_cols_to_lag:
        lag_values_mobility[f'{c}']=tl.get_time_lag_value(dfs = dynamic_df_norm_split, 
                                                          trgt = 'COVID_Cases_norm_lag_area',
                                                          vacc = 'total_vaccinated_first_dose_norm_lag_pop',
                                                          mobility = c, 
                                                          region = lag_granularity,
                                                          window_days = 0,
                                                          start_date = '2020-01-01',
                                                          n_lag=12, 
                                                          plt_flg=True, 
                                                          moblty_flag=True)
        
    # calculate and store lag values for vaccination
    # currently empty because cf.vacc_cols_to_lag is empty
    lag_values_vacc={}
    for c in cf.vacc_cols_to_lag:
        lag_values_vacc[f'{c}']=tl.get_time_lag_value(dfs = dynamic_df_norm_split, 
                                                      trgt = 'COVID_Cases_norm_lag_area',
                                                      vacc = c,
                                                      mobility = 'worker_visitor_footfall_sqkm', 
                                                      region = lag_granularity,
                                                      window_days = 30, 
                                                      start_date = '2020-01-01',
                                                      n_lag=12, 
                                                      plt_flg=True,
                                                      moblty_flag=False)
        
    # making stationary if wanted
    # split the data for each LSOA
    dynamic_df_lsoa = [pd.DataFrame(y) for x, y in dynamic_df.groupby('LSOA11CD', as_index=False)]

    # flag for whether to perform differencing on mobility and vaccination data
    # TODO add to config and as parameter here
    flg_stnrty_both=False  
    
    # fetch names of columns which were lagged
    mobility_vars=[s.replace('_norm_lag_area','') for s in  list(lag_values_mobility.keys())] 
    vacc_vars=[s.replace('_norm_lag_pop','') for s in  list(lag_values_vacc.keys())] 
    
    # TODO could replace this bit with a group by to avoid concat
    
    if flg_stnrty_both:
        # Perform first order differencing to achieve stationarity on mobility and vaccination data only
        dynamic_df_lsoa_diff=[x.sort_values(by='Date')[['Date','LSOA11CD']+mobility_vars+vacc_vars].\
                                     set_index(cf.cols_not_to_lag).\
                                     diff().dropna() for x in dynamic_df_lsoa]
    
    else:
        # Non-stationary versions of mobility and vaccination data
        dynamic_df_lsoa_diff=[x.sort_values(by='Date')[['Date','LSOA11CD']+mobility_vars+vacc_vars].set_index(cf.cols_not_to_lag)\
                                           for x in dynamic_df_lsoa]
    
    dynamic_df_lsoa_diff=pd.concat(dynamic_df_lsoa_diff,axis=0).reset_index()
    
    # Concat cases data for all LSOAs
    dynamic_df_lsoa = pd.concat(dynamic_df_lsoa,axis=0)[['Date','LSOA11CD','COVID_Cases','cases_per_person', 'pct_infected_all_time',
                                                                           'cumsum_divided_area']].reset_index(drop=True)

    # Join case, vaccination, and mobility data for all LSOAs
    dynamic_df_lsoa_diff = dynamic_df_lsoa_diff.merge(dynamic_df_lsoa,on=['Date','LSOA11CD'],how='inner')
    
    # applying time lag to variables

    dynamic_df_lagged = []

    for mob_col in mobility_vars:
        lag_key = mob_col + '_norm_lag_area'

        df_lagged = pd.concat(tl.split_df_apply_time_lag(dynamic_df_lsoa_diff, 
                                                         [mob_col],
                                               lag_values_mobility[lag_key],
                                               apply_lag=True),axis=0).reset_index()

        dynamic_df_lagged.append(df_lagged)

    # no lag applied to cases data
    df_lagged_cases = pd.concat(tl.split_df_apply_time_lag(dynamic_df_lsoa_diff,['COVID_Cases','cases_per_person', 'pct_infected_all_time','cumsum_divided_area'],
                                                               apply_lag=False),axis=0).reset_index()
    
    dynamic_df_lagged_merged = reduce(lambda left, right: pd.merge(left, right, on = ['LSOA11CD', 'Date'], how='inner'), dynamic_df_lagged) 

    dynamic_df_lagged_merged = dynamic_df_lagged_merged.merge(df_lagged_cases, on=['LSOA11CD', 'Date'], how='inner')

    # TODO change this to fetch the travel clusters from the static_df in the main script
    # columns produced are LSOA11CD and travel_cluster
    travel_clusters=factory.get('mobility_clusters_processed').create_dataframe()

    dynamic_df_lagged_merged = dynamic_df_lagged_merged.merge(travel_clusters, on = ['LSOA11CD'], how='inner')
    
    # change column names for subsequent scripts
    dynamic_df_lagged_merged = dynamic_df_lagged_merged.rename(columns={'COVID_Cases':'COVID_Cases_per_unit_area',
                "cases_per_person":"COVID_Cases_prop_population",
                "cumsum_divided_area":"COVID_Cases_per_unit_area_cumsum",
                "pct_infected_all_time":"COVID_Cases_prop_population_cumsum"})
    
    if flg_stnrty_both:
        dynamic_df_lagged_merged.to_gbq(cf.lagged_dynamic_stationary,\
                               project_id = cf.project_name,if_exists='replace')
    else:
        dynamic_df_lagged_merged.to_gbq(cf.lagged_dynamic_non_stationary,\
                                                 project_id=cf.project_name,if_exists='replace')
    
    return dynamic_df_lagged_merged
    
    
    
    

    
