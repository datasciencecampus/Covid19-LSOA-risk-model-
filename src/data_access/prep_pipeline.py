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
    Calculate the area of a geometry column in a dataframe and add to the dataframe as an 'Area' column.
    
    :param df: Input dataframe, from read_data function
    :type df: Pandas DataFrame
    
    :param geo_col: Name of column from which to calculate the area. This column must have a geometry dtype. 
    :type geo_col: str
    
    :return: DataFrame with Area
    :rtype: Pandas DataFrame
    
    '''
    
    # merge geo data to get areas of each LSOA
    gdf = gpd.GeoDataFrame(df, crs="EPSG:27700", geometry=df[geo_col])
    gdf.crs
    gdf['Area'] = gdf[geo_col].area/10**6
    
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
    
    :param col_list: Column(s) to perform forward fill on
    :type col_list: string, or list of strings
    
    :param sort_col: Column(s) by which to sort the dataframes prior to forward filling, defaults to 'Date'. 
    :type sort_col: string, or list of strings
    
    :param group_col: Column(s) to group by prior to forward filling. Defaults to 'LSOA11CD'.
    :type group_col: string, or list of strings
    
    :return: DataFrame with fixed cumulative sums and 0s in place of any starting NaNs should there be any.
    :rtype: Pandas DataFrame
    
    '''
    
    df.sort_values(by=sort_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df[col_list].replace(0, np.nan, inplace=True)
    
    df[col_list] = df.groupby(group_col)[col_list].ffill().fillna(0).astype(float)
    
    return df

def sum_features(df, dic = cf.static_cols_to_sum):
    '''
    Create new features by summing the values of other features in a dataframe. The definition for
    which features should be summed is provided as a dictionary 'static_cols_to_sum' from the config file.
    The columns containing the individual features before summation are removed from the dataframe.
    
    :param df: The dataframe of static features
    :type df: Pandas DataFrame
    
    :param dic: A dictionary where the keys are the names of new column to be created and the values are
    the individual columns to be summed to create them.
    :type dic: dictionary
    
    :return: A dataframe with new summed features added and the individual feature columns removed
    :rtype: Pandas DataFrame
    '''

    # create a copy of the DataFrame
    df_copy = df.copy()
    
    # for each new column, and list of columns to sum
    for new_col, cols_to_sum in dic.items():
        
        df_copy[new_col] = df_copy[cols_to_sum].sum(axis=1)
        
        # drop original columns
        df_copy.drop(columns=cols_to_sum, inplace=True)
    
    return df_copy

def apply_timelag(dynamic_df, dynamic_df_norm, save_results=True):
    '''
    Calculate appropriate time lag values to use and apply to dynamic dataset.
    
    :param dynamic_df: dynamic dataframe as produced by the dynamic dataset preprocessing, also accessed through DataFactory.get('lsoa_dynamic').create_datarame().
    :type dynamic_df: Pandas DataFrame
    
    :param dynamic_df_norm: normalised dynamic dataframe as produced by the dynamic dataset preprocessing, also accessed through DataFactory.get('dynamic_raw_norm_chosen_geo').create_dataframe().
    :type dynamic_df: Pandas DataFrame
    
    :param save_results: whether to save output to GCP or not, default True
    :type save_results: boolean
    
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
    
    if save_results:

        if flg_stnrty_both:
            dynamic_df_lagged_merged.to_gbq(cf.lagged_dynamic_stationary,\
                                   project_id = cf.project_name,if_exists='replace')
        else:
            dynamic_df_lagged_merged.to_gbq(cf.lagged_dynamic_non_stationary,\
                                                     project_id=cf.project_name,if_exists='replace')
    
    return dynamic_df_lagged_merged


def join_cases_to_static_data(static_df):
    """
    Prepare the Test & Trace data containing the number of positive cases at LSOA level. The number of cases
    is left joined to the exist static data to ensure that a record exists for every LSOA in every week.
    
    :param static_df: The processed static data set
    :type static_df: Pandas DataFrame
    
    :return: A Pandas DataFrame containing static variables and cases for each LSOA each week
    :rtype: Pandas DataFrame
    """
    
    # ingest cases data
    cases_df = factory.get('aggregated_tests_lsoa').create_dataframe()

    # sort cases data by date
    cases_df_datum = cases_df[['Date','LSOA11CD','COVID_Cases']].sort_values(by='Date').reset_index(drop=True)

    # create a list of dataframes of cases, one df for each week 
    cases_df_datum = [pd.DataFrame(y) for x, y in cases_df_datum.groupby('Date', as_index=False)]

    cases_df_datum_mrgd = []

    # The cases dataframe is split into different dates.
    # This splitting allows for each dataframe to be left joined to the static data
    # Therefore there will be a record for cases for every LSOA in every week
    # If no cases data is present for a given week in a given LSOA, the 'Date'
    # field is filled with the 'Date' value from that DataFrame

    # for each df
    for splt_df in cases_df_datum:

        # store the date for the given DataFrame
        datm = splt_df['Date'].unique()[0]

        # left-join cases onto the static data
        df = static_df.merge(splt_df,how='left',on=['LSOA11CD'])

        # fill any gaps in the cases data with the correct date
        df['Date'] = df['Date'].fillna(datm)

        # any dates that needed to be filled had zero cases for that week
        df['COVID_Cases'] = df['COVID_Cases'].fillna(0)

        # apply normalisation
        df['COVID_Cases'] = df['COVID_Cases'].div(df['Area'])

        cases_df_datum_mrgd.append(df)

    # stack the dataframes        
    cases_static_df = pd.concat(cases_df_datum_mrgd).reset_index(drop=True)

    # drop the area column
    cases_static_df.drop('Area', axis=1, inplace=True)

    # rename to reflect normalisation
    cases_static_df.rename(columns={'COVID_Cases':'COVID_Cases_per_unit_area'},inplace=True)
    
    return cases_static_df

def join_tranches_mobility_data(cases_all_weeks_df, deimos_footfall_df):
    '''
    Load the mobility data and join it to the static and cases dataframe
    
    :param cases_all_weeks_df: A dataframe containing static and cases data
    :type cases_all_weeks_df: Pandas DataFrame
    
    :param deimos_footfall_df: A dataframe containing mobility data
    :type deimos_footfall_df: Pandas DataFrame
    
    :return: A single dataframe containing static data, cases data and mobility data
    :rtype: Pandas DataFrame
    '''
    
    # drop the footfall columns that aren't used in the time tranches model
    deimos_footfall_df.drop(['msoa_people', 'resident_footfall_sqkm', 'total_footfall_sqkm', 'visitor_footfall_sqkm', 'worker_footfall_sqkm'], axis=1, inplace=True)
    
    # convert date to string for joining 
    deimos_footfall_df['Date'] = deimos_footfall_df['Date'].astype(str)
    
    # join mobility with the static and cases data
    cases_mobility_all_weeks_df = cases_all_weeks_df.merge(deimos_footfall_df, how='inner', on=['LSOA11CD','Date'])
    
    return cases_mobility_all_weeks_df


def derive_week_number(cases_static_df):
    '''
    Add a column to the data set which maps each week to a week number from the time
    that the first data is available. The function also asserts that the number of LSOAs
    in the data is equal to the number of LSOAs specified in the config file.
    
    :param cases_static_df: DataFrame containing the static data
    :type cases_static_df: Pandas DataFrame
    
    :return: A dataframe with a column of week numbers
    :rtype: Pandas DataFrame
    '''
    
    # Sort the unique dates 
    date_list = sorted(cases_static_df['Date'].dt.date.unique())

    # assign a number of each week and append it as a string
    week_list = ["week_" + str(x+1) for x in range(len(date_list))]

    # create a dictionary of dates and their corresponding string
    date_dict = dict(zip(date_list,week_list))

    # create a new column in the static dataframe containing the mapping of dates to their week number
    cases_static_df['week'] = cases_static_df['Date'].map(date_dict)
    
    # not sure why we do this, maybe drop?
    cases_static_df['Date']=cases_static_df['Date'].astype(str)

    # remove the index - again not sure why we do this
    cases_static_df = cases_static_df.reset_index(drop=True)

    # check that every LSOA is present in the data
    assert cases_static_df.groupby('Date')['LSOA11CD'].count().unique() == cf.n_lsoa
    
    return cases_static_df


def create_test_data(all_weeks_df, static_df, deimos_footfall_df, idbr_features = cf.tranche_model_idbr_features):
    '''
    Create a test data set which contains records for which mobility data
    is available but cases data is not available. 
    
    :param all_weeks_df: A DataFrame containing the static and cases data
    :type all_weeks_df: Pandas DataFrame
    
    :param static_df: A DataFrame containing the static data only
    :type static_df: Pandas DataFrame
    
    :param deimos_footfall_df: A DataFrame containing footfall data
    :type deimos_footfall_df: Pandas DataFrame
    
    :param idbr_features: List of column names of the IDBR features
    :type idbr_features: [str]
    
    :return : A DataFrame on which to test the trained model
    :rtype: Pandas DataFrame
    '''
    
    # merge footfall with the static data
    footfall_static_df = deimos_footfall_df.merge(static_df, how='inner', on=['LSOA11CD'])
    
    # find the max date for which cases data is avaiable
    date_cutoff = all_weeks_df['Date'].max()
    
    # the test set contains only timestamps where footfall data is available, but not cases data
    test_df = footfall_static_df[footfall_static_df['Date'] > date_cutoff].reset_index(drop=True)
    
    # convert units of mobility features to align with the training data
    test_df = convert_units(df = test_df, 
                            colname = 'worker_visitor_footfall_sqkm',
                            factor = 0.000001,
                            new_colname = 'worker_visitor_footfall_sqm')
    
    # convert units of IDBR features to align with the training data
    for feature in idbr_features:
        
        test_df = convert_units(df = test_df, 
                                colname = feature, 
                                factor = 0.01)
    
    # store the date range as a string
    test_data_range = test_df['Date'].min() + '-' + test_df['Date'].max()
    
    # collapse into one row per LSOA
    test_df = test_df.groupby(['LSOA11CD', 'travel_cluster'])[list(test_df.select_dtypes(include=np.number).columns)].mean().reset_index()
        
    # insert date range column
    test_df['Date'] = test_data_range
    
    return test_df


def create_time_tranches(all_weeks_df, 
                         tranche_dates = cf.tranche_dates, 
                         tranche_description = cf.tranche_description):
    '''
    Function to process all weeks of data into time tranches whose boundaries are
    defined by a list of dates 'tranche_dates'. The 'all_weeks_df' dataframe is split
    into individual time tranches dataframes between the boundary dates. Numeric features are 
    then aggregated for each time tranche before the individual tranche dataframes are concatenated
    into a single dataframe.

    :param all_weeks_df: A dataframe containing all weeks of static, dynamic and cases data
    :type all_weeks_df: Pandas DataFrame
    
    :param tranche_dates: A list of the boundary dates for the time tranches
    :type tranche_dates: [str]
    
    :param tranche_description: A list containing a short description of the conditions in each time tranche
    :type tranche_description: [str]
    
    :return: A dataframe containing features aggregated at time tranche level
    :rtype: Pandas DataFrame
    '''
    
    # list to store a dataframe for each tranche
    tranche_dfs = []

    # for each item in the list of tranche boundary dates
    for i, date in enumerate(tranche_dates):
        
        # t1 is the selected date
        t1 = date

        # if i is the final element of the list
        if i == len(tranche_dates) - 1:

            # subset for all dates after t1
            df_time = all_weeks_df[all_weeks_df['Date'] > t1]


        # if i is not the final element of the list of dates
        else:
            
            # t2 is the next date in the list
            t2 = tranche_dates[i + 1]

            # subset for all dates between t1 and t2
            df_time = dt.create_time_slice(all_weeks_df, t1, t2)

        # if the time tranche dataframe is not empty
        if not df_time.empty:

            # add a column describing the tranche
            df_time.loc[:, 'tranche_desc'] = tranche_description[i]
            
            # append the dataframe to the list of individual tranche dataframes
            tranche_dfs.append(df_time)

    # Perform aggregation of predictors and target variable for each tranche
    # Each tranche contains multiple weeks, aggregation results in mean of each of the numerical features
    # In practice, the static features are the same for each week, so we are averaging footfall over the tranche
    # Each individual tranche dataframe will have one unique record for each LSOA (because of averaging)
    
    tranche_dfs_agg = []

    for df in tranche_dfs:

        # make a copy of the dataframe
        tranche_df = df.copy()
                
        # derive a string containing the date range of the tranche
        date_range_string = str(tranche_df['Date'].min()) + '-' + str(tranche_df['Date'].max())

        # drop the date column
        tranche_df.drop('Date', axis=1, inplace=True) 
        
        # insert the date range string as a new date column
        tranche_df['Date'] = date_range_string

        # define columns to group by
        group_by_cols = ['Date', 'LSOA11CD', 'tranche_desc', 'travel_cluster']

        # compute the mean over each week in the tranche
        tranche_df = tranche_df.groupby(group_by_cols).mean().reset_index()

        # sort by LSOA code
        tranche_df = tranche_df.sort_values(by='LSOA11CD').reset_index(drop=True)

        # append the df to the list of aggregated dataframes
        tranche_dfs_agg.append(tranche_df)

    # stack each tranche into one dataframe
    all_tranches_df = pd.concat(tranche_dfs_agg).reset_index(drop=True)

    # convert date to string
    all_tranches_df['Date'] = all_tranches_df['Date'].astype(str)
    
    return all_tranches_df


def derive_tranche_order(all_tranches_df, tranche_description = cf.tranche_description):
    '''
    Create a nested dictionary of tranch dates, descriptions and order by tranche number.
    Use the dictionary to create a new column 'tranche_order 'showing the tranche number 
    that each record corresponds to.
    
    :param all_tranches_df: A dataframe containing data for all tranches
    :type all_tranches_df: Pandas DataFrame
    
    :param tranche_description: A list containing a short description of the conditions in each time tranche
    :type tranche_description: [str]
    
    :return: A dataframe with a 'tranche_order' column added
    :rtype: Pandas DataFrame
    '''

    # find unique date range and tranche description combinations
    df_key = all_tranches_df[['Date','tranche_desc']].drop_duplicates().reset_index(drop=True)

    # put them into a dictionary
    event_dict = dict(zip(df_key['Date'].values, df_key['tranche_desc'].values))

    # list of integers from 1 to n_tranches inclusive 
    tranche_order = list(range(1, cf.n_tranches + 1))

    # zip the tranche descriptions and tranche numbers
    event_order_dict = dict(zip(tranche_description, tranche_order))

    # create new column for tranche number
    all_tranches_df['tranche_order'] = all_tranches_df['tranche_desc'].map(event_order_dict)
    
    return all_tranches_df


def convert_units(df, colname, factor, new_colname=None):
    '''
    Multiply a dataframe column by a factor and replace the original value
    with the multiplied value. Rename the column to reflect the new units.
    
    :param df: A dataframe containing a column whose units should be multipled by
    the user specified factor
    :type df: Pandas DataFrame
    
    :param colname: The name of the column to be multiplied by the factor
    :type colname: string
    
    :param factor: The factor that the contents of column 'colname' will be multiplied by
    :type factor: float
    
    :param new_colname: The name with which to replace 'colname' to reflect the changed units
    :type new_colname: string
    
    :return: A dataframe with the the transformation and column name change applied
    :rtype: Pandas DataFrame
    '''
    
    if new_colname:
        
        df[new_colname] = df[colname] * factor
        
        df.drop(colname, axis=1, inplace=True)
    
    else:
        
        df[colname] = df[colname] * factor
        
    return df
