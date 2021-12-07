# Preprocessing
# Sections: Read in static data; read in dynamic data; apply time lag to dynamic datasets for modelling

# Import Packages
import os
import sys
import string
from datetime import date, datetime, timedelta
import time
import random
from random import randint
import math
from functools import reduce

import pandas as pd
import numpy as np
from numpy.random import seed, randn
import pandas_gbq
import geopandas as gpd

from google.cloud import bigquery

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
import dash
from plotly.offline import iplot, init_notebook_mode

import swifter
import pgeocode

from scipy import stats
from scipy.stats import pearsonr
from scipy.signal import argrelextrema

from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LogisticRegression
                                 ,Lasso
                                 ,ElasticNet
                                 ,LinearRegression)
from sklearn.metrics import median_absolute_error, r2_score, mean_squared_error

from sklego.meta import ZeroInflatedRegressor, EstimatorTransformer

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Import from local data files
current_path = os.path.abspath('.')
sys.path.append(os.path.dirname(current_path))

from data_access.data_factory import DataFactory as factory
from utils import data as dt
from utils import config as cf

#############################
## 
# funcs
# probably move these elsewhere once done

def read_data(table_type, table_dict = cf.data_tables, join_col = 'LSOA11CD'):
    '''
    Read in and join a list of data tables on a common column.
    
    :param table_type: Type of tables to read in, must be one of 'static' or 'dynamic'. This is fed into the table_dict parameter and will determine the list of tables to read in. 
    :type table_type: string
    
    :param table_dict: Dictionary of tables to read in where the values are lists of strings of table names for this function to read in. These strings feed into the DataFactory.get() function within src/data_access/data_factory.py. Defaults to the data_tables variable within src/data_access/config.py
    :type table_dict: dictionary
    
    :param join_col: The common column on which to join ALL of the tables. Any common columns not specified here will be dropped from the right table in each join. Defaults to 'LSOA11CD'
    :type join_col: string or list of strings
    
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
            df_final = df_final.merge(df, on=join_col, how='left', suffixes=['', '_drop'])
            
    drop_cols = [col for col in df_final.columns if col.endswith('_drop')]
    df_final.drop(columns=drop_cols, inplace=True)
            
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
    gdf = gpd.GeoDataFrame(df, crs="EPSG:27700", geometry=static_df[geo_col])
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

def ffill_cum(df, sort_col='Date', col_substring = '_norm_lag', group_col = 'LSOA11CD'):
    
    '''
    Perform forward fill on cumulative sum columns. This is needed because cumulative sums have been performed on the source data which does not have entries for every day.
    
    :param df: Input dataset, typically resulting from calls to read_data and other subsequent preprocessing functions.
    :type df: Pandas DataFrame
    
    :param sort_col: Column(s) by which to sort the dataframes prior to forward filling, defaults to 'Date'. 
    :type sort_col: string, or list of strings
    
    :param col_substring: Substring to search for in the columns of the df, to provide the list of columns to forward fill. Defaults to '_norm_lag'.
    :type col_substring: string
    
    :param group_col: Column(s) to group by prior to forward filling. Defaults to 'LSOA11CD'.
    :type group_col: string, or list of strings
    
    :return: DataFrame with fixed cumulative sums and 0s in place of any starting NaNs should there be any.
    :rtype: Pandas DataFrame
    
    '''
    
    df = df.sort_values(by=sort_col)
    df.replace(0, np.nan, inplace=True)
    
    cols = [col for col in df.columns if col_substring in col]
    
    df[cols] = df.groupby(group_col)[cols].ffill().fillna(0).astype(float)
    
    df.fillna(0, inplace=True)
    
    return df

###

# read in static data
static_df = read_data('static')
static_df = geo_merge(static_df)

# normalise static data
static_df = normalise_data(static_df, 'static')
    
# remove ethnicity subgroups
ethnicity_list = dt.get_ethnicities_list(static_df,subgroups=True)
static_df = static_df.drop(columns=ethnicity_list) 

static_df=static_df[static_df.LSOA11CD.str.startswith('E')] #only keeping England

static_df=static_df.fillna(0)  #fill 0s(nans exist in occupations where nobody works in them)

# save to wip file
static_df.to_gbq(cf.static_data_file, project_id=cf.project_name,if_exists='replace')

# read in dynamic data
# shift up to be with static etc?
    
dynamic_df = read_data('dynamic', join_col=['LSOA11CD', 'Date'])

# join on subset of static data for geographic variables
col_list = cf.static_subset
static_subset_df = static_df[col_list]   
dynamic_df = dynamic_df.merge(static_subset_df,on=['LSOA11CD'],how='right')
    
# Filter to England only
dynamic_df = dynamic_df[dynamic_df.LSOA11CD.str.startswith('E')] 
dynamic_df = dynamic_df.fillna(0)

# Normalise population by a common geography so lag values in following code can be calculated correctly
# need to rethink these names
lag_granularity = cf.chosen_granularity_for_lag
dynamic_df_norm = dynamic_df.copy()

df_travel_clusters = dynamic_df_norm.drop_duplicates(subset='LSOA11CD',keep='first')[[lag_granularity,'Area','ALL_PEOPLE']].groupby(lag_granularity).sum().reset_index()\
.rename(columns={'Area':'Area_chosen_geo','ALL_PEOPLE':'Population_chosen_geo'})

dynamic_df_norm = dynamic_df_norm.merge(df_travel_clusters, how='left', on=lag_granularity)

# convert back to raw so we can divide by travel cluster area
for i in [i for i in dynamic_df_norm.columns.tolist() if (('footfall' in i)|('inflow' in i))]:
    dynamic_df_norm[i] = dynamic_df_norm[i]*dynamic_df_norm['Area']  

# normalise dynamic data
dynamic_df_norm = normalise_data(dynamic_df_norm, 'dynamic_norm')

dynamic_df_norm = ffill_cum(dynamic_df_norm)

# normalise the original dynamic df
dynamic_df = normalise_data(dynamic_df, 'dynamic')

dynamic_df = ffill_cum(dynamic_df, col_substring='_norm2')
                       
# TODO: rename columns (or change subsequent code to use the new ones...)
# also think of better suffix for these columns

dynamic_df.to_gbq(cf.dynamic_data_file, project_id=cf.project_name, if_exists='replace')
dynamic_df_norm.to_gbq(cf.dynamic_data_file_normalised, project_id=cf.project_name, if_exists='replace')

########################
# lag section

dynamic_df['Date'] = pd.to_datetime(dynamic_df['Date'])
dynamic_df_norm['Date'] = pd.to_datetime(dynamic_df_norm['Date'])

# stationarity check - have as optional output printed to file?
# ignoring for now

# Computing and storing weeks lag for each variable

dynamic_df['Country'] = 'England'


    
    
    
    
    
    
    