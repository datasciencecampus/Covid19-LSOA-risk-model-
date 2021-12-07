## LB - temp notes:
# create_static_data
# create_dynamic_data
# apply_timelag_dynamic

## to do:
# remove exploratory/plotting bits from code (currently this is just copy pasted)
# look into doing code in for loops where there are the concats etc 
# arrange code into functions to allow for easy testing

#############################

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

# SECTION ONE OF THREE
# Read in and normalise the static datasets
# i.e., those which do not change week by week
# Demographics, travel clusters, flow to work, high risk industries, LSOA geography information

# Read in data
static_df = factory.get('static_vars').create_dataframe()

binned = factory.get('mid_year_lsoa').create_dataframe()
travel_clusters = factory.get('mobility_clusters_processed').create_dataframe()
df_flow_to_work = factory.get('flow_to_work').create_dataframe()
LSOA_2011_map = factory.get('LSOA_2011').create_dataframe()

# merge data into a single Dataframe
static_df = static_df.merge(binned, on='LSOA11CD',how='left')
static_df = static_df.merge(travel_clusters,on='LSOA11CD')
static_df = static_df.merge(df_flow_to_work, on="LSOA11CD")
static_df = static_df.merge(LSOA_2011_map.drop(columns='LSOA11NM'), on='LSOA11CD')

# merge geo data to get areas of each LSOA
gdf = gpd.GeoDataFrame(static_df, crs="EPSG:27700", geometry=static_df['geometry'])
gdf.crs
gdf['Area']=gdf['geometry'].area/10**6

# combine and rename travel clusters 
static_df=dt.combining_and_remap_travel_cluster(gdf)

### Normalise variables

# convert ethnicity counts to proportions by LSOA
static_df = dt.normalise(static_df, cf.features_dict['umbrella_ethnicity'])

# remove ethnicity subgroups
ethnicity_list = dt.get_ethnicities_list(static_df,subgroups=True)
static_df = static_df.drop(columns=ethnicity_list) 

# normalise communal variables by area
#LB: this can be replaced by dt.normalise
static_df = dt.divide_comm_vars_by_area(static_df)  
    
# normalise family variables by total count of families
static_df = dt.normalise(static_df, cf.features_dict['fam_ftrs'], by=static_df['FAMILIES_WITH_DEPENDENT_CHILDREN_ALL_FAMILIES'])

# proportion of paid and unpaid care
# use features_dict['paid_ftrs'] 
static_df = dt.normalise(static_df, ['UNPAID_CARE_1_HOUR_PLUS', 'NO_UNPAID_CARE'])

# Get proportion of household sizes
static_df = dt.normalise(static_df, cf.features_dict['hh_sizes'])
    
# Normalise travel to work by people
static_df = dt.normalise(static_df, cf.features_dict['trvl_ftrs'])
    
#Normalise ages by total count of people
static_df = dt.normalise(static_df, cf.features_dict['age_features'], by=static_df['ALL_PEOPLE'])


static_df=static_df[static_df.LSOA11CD.str.startswith('E')] #only keeping England

static_df=static_df.fillna(0)  #fill 0s(nans exist in occupations where nobody works in them)

# save to wip file
static_df.to_gbq(cf.static_data_file, project_id=cf.project_name,if_exists='replace')

#############################

# SECTION TWO OF THREE
# Read in and normalise the dynamic datasets
# i.e., those which do change week by week
# Cases, vaccinations, and travel clusters

# read in dynamic data
cases_df = factory.get('aggregated_tests_lsoa').create_dataframe()
vaccination_df = factory.get('lsoa_vaccinations').create_dataframe()
deimos_trip_df = factory.get('Deimos_trip_end_count').create_dataframe()
deimos_footfall_df = factory.get('lsoa_daily_footfall').create_dataframe()

# read in static data for geographic variables
col_list = cf.static_subset
static_subset_df = static_df[col_list]

# Merge into a single dataframe 
dynamic_df = deimos_footfall_df.merge(cases_df,how='left', on=['LSOA11CD','Date']).fillna(0)
dynamic_df = dynamic_df.merge(deimos_trip_df,how='left', on=['LSOA11CD','Date']).fillna(0)
dynamic_df = dynamic_df.merge(vaccination_df,how='left', on=['LSOA11CD','Date']).fillna(0)

dynamic_df = dynamic_df.merge(static_subset_df,on=['LSOA11CD'],how='right')

# Filter to England only
dynamic_df = dynamic_df[dynamic_df.LSOA11CD.str.startswith('E')] 
 
# Normalise population by a common geography so lag values in following code can be calculated correctly
lag_granularity = cf.chosen_granularity_for_lag
dynamic_df_norm = dynamic_df.copy()

# LB: maybe rename this? already have a travel_clusters variable
df_travel_clusters = dynamic_df_norm.drop_duplicates(subset='LSOA11CD',keep='first')[[lag_granularity,'Area','ALL_PEOPLE']].groupby(lag_granularity).sum().reset_index()\
.rename(columns={'Area':'Area_chosen_geo','ALL_PEOPLE':'Population_chosen_geo'})

dynamic_df_norm = dynamic_df_norm.merge(df_travel_clusters, how='left', on=lag_granularity)

# LB: do this in a for loop
# can group some of these columns up like so:
# (use features_dict['dynamic_pop/area_norm'])
###########
df_raw2=dt.normalise(df_raw2, ['total_vaccinated_first_dose','total_vaccinated_second_dose', "full_vacc_cumsum", 'COVID_Cases', "cases_cumsum"],by=df_raw['Population_chosen_geo'], suffix='_norm_lag_pop')

df_raw2=dt.normalise(df_raw2, ["COVID_Cases", "cases_cumsum"],by=df_raw2['Area_chosen_geo'], suffix='_norm_lag_area')
#########

dynamic_df_norm = dt.normalise(dynamic_df_norm, ['total_vaccinated_first_dose','total_vaccinated_second_dose'],by=dynamic_df_norm['Population_chosen_geo'], suffix='_norm_lag_pop')

dynamic_df_norm = dt.normalise(dynamic_df_norm, ["full_vacc_cumsum"],by=dynamic_df_norm['Population_chosen_geo'], suffix='_norm_lag_pop')

dynamic_df_norm = dt.normalise(dynamic_df_norm, ["COVID_Cases"],by=dynamic_df_norm['Population_chosen_geo'], suffix='_norm_lag_pop')
dynamic_df_norm = dt.normalise(dynamic_df_norm, ["COVID_Cases"],by=dynamic_df_norm['Area_chosen_geo'], suffix='_norm_lag_area')

dynamic_df_norm = dt.normalise(dynamic_df_norm, ["cases_cumsum"],by=dynamic_df_norm['Population_chosen_geo'], suffix='_norm_lag_pop')
dynamic_df_norm =dt.normalise(dynamic_df_norm, ["cases_cumsum"],by=dynamic_df_norm['Area_chosen_geo'], suffix='_norm_lag_area')

# LB: how to deal with this bit with the normalisation function??
# take the normalise out of the for loop duh
# consider renaming the columns after this for clarity (will still have the _sqkm suffix)
# will need to put the *area bit before the normalise - it's in the features_dict['dynamic_area'] dict now

# convert back to raw so we can divide by travel cluster area
for i in [i for i in dynamic_df_norm.columns.tolist() if (('footfall' in i)|('inflow' in i))]:
    dynamic_df_norm[i]=dynamic_df_norm[i]*dynamic_df_norm['Area']          
    dynamic_df_norm=dt.normalise(dynamic_df_norm, [i],by=dynamic_df_norm['Area_chosen_geo'], suffix='_norm_lag_area')
    
# Forward fill cumulative sums for rows where there was no data

dynamic_df_norm = dynamic_df_norm.sort_values(by='Date')
#switch 0s to np.nans for forward fills
dynamic_df_norm.replace(0, np.nan, inplace=True)  
#fill in empty cumulative sum values with a forward sum
dynamic_df_norm.cases_cumsum_norm_lag_pop = dynamic_df_norm.groupby(['LSOA11CD']).cases_cumsum_norm_lag_pop.ffill().fillna(0).astype(float)  
dynamic_df_norm.full_vacc_cumsum_norm_lag_pop = dynamic_df_norm.groupby(['LSOA11CD']).full_vacc_cumsum_norm_lag_pop.ffill().fillna(0).astype(float)
dynamic_df_norm.cases_cumsum_norm_lag_area=dynamic_df_norm.groupby(['LSOA11CD']).cases_cumsum_norm_lag_area.ffill().fillna(0).astype(float)
dynamic_df_norm=dynamic_df_norm.fillna(0)  #fill nans with 0s

#LB: use dt.normalise function
# and features_dict['dynamic_pop/area']
# might need to change the flag there

# Normalise variables
dynamic_df.loc[:,['total_vaccinated_first_dose','total_vaccinated_second_dose']] = \
dynamic_df[['total_vaccinated_first_dose','total_vaccinated_second_dose']].div(dynamic_df['ALL_PEOPLE'],axis=0)

dynamic_df['pct_of_people_full_vaccinated'] = dynamic_df["full_vacc_cumsum"].div(dynamic_df['ALL_PEOPLE'],axis=0)
dynamic_df['cases_per_person'] = dynamic_df[['COVID_Cases']].div(dynamic_df['ALL_PEOPLE'],axis=0)
dynamic_df['COVID_Cases'] = dynamic_df[['COVID_Cases']].div(dynamic_df['Area'],axis=0)
dynamic_df['pct_infected_all_time'] = dynamic_df[['cases_cumsum']].div(dynamic_df['ALL_PEOPLE'],axis=0)
dynamic_df['cumsum_divided_area'] = dynamic_df[['cases_cumsum']].div(dynamic_df['Area'],axis=0)

# LB rename columns after this bit

# Final data cleaning
dynamic_df=dynamic_df.sort_values(by='Date')
dynamic_df.replace(0, np.nan, inplace=True)  #switch 0s to np.nans for forward fills
dynamic_df.pct_infected_all_time = dynamic_df.groupby(['LSOA11CD']).pct_infected_all_time.ffill().fillna(0).astype(float)  #fill in empty cumulative sum values with a forward sum
dynamic_df.pct_of_people_full_vaccinated = dynamic_df.groupby(['LSOA11CD']).pct_of_people_full_vaccinated.ffill().fillna(0).astype(float)
dynamic_df.cumsum_divided_area = dynamic_df.groupby(['LSOA11CD']).cumsum_divided_area.ffill().fillna(0).astype(float)
dynamic_df=dynamic_df.fillna(0)  #fill nans with 0s

dynamic_df.to_gbq(cf.dynamic_data_file, project_id=cf.project_name, if_exists='replace')
dynamic_df_norm.to_gbq(cf.dynamic_data_file_normalised, project_id=cf.project_name, if_exists='replace')

#############################

# SECTION THREE OF THREE
# Lag values are calculated and applied to the dynamic dataset produced in section two above
# Variables are renamed for modelling

dynamic_df['Date'] = pd.to_datetime(dynamic_df['Date']).dt.tz_convert(None)
dynamic_df_norm['Date'] = pd.to_datetime(dynamic_df_norm['Date']).dt.tz_convert(None)

# Optional stationarity check for each variable
run_stationarity_check = cf.explore_stationarity_check  

if run_stationarity_check:
    for col in cf.features_dict['deimos_cols'] +cf.features_dict['non_deimos']:
        print("------------------",col,"-----------------")
        dyn.check_for_stationarity(dynamic_df_norm,col)
        dyn.plot_var_by_travel_cluster(dynamic_df_norm,col)

# Computing and storing weeks lag for each variable

dynamic_df['Country'] = 'England'

# LB: was this supposed to be dynamic_df_norm or dynamic_df? Original name suggests dynamic_df...
dynamic_df_norm_split = [pd.DataFrame(y) for x, y in \
                                      dynamic_df_norm.groupby(lag_granularity, as_index=False)] #splitting data by chosen region

tl=dyn.TimeLag() 

# calculate and store lag values for mobility
lag_values_mobility={}

for c in cf.mobility_cols_to_lag:
    lag_values_mobility[f'{c}']=tl.get_time_lag_value(dynamic_df_norm_split, 'COVID_Cases_norm_lag_area','total_vaccinated_first_dose_norm_lag_pop',
                                                   c, lag_granularity,0,'2020-01-01',n_lag=12, plt_flg=True, moblty_flag=True)
    
# calculate and store lag values for vaccination
lag_values_vacc={}
for c in cf.vacc_cols_to_lag:
    lag_values_vacc[f'{c}']=tl.get_time_lag_value(dynamic_df_norm_split, 'COVID_Cases_norm_lag_area',c,
                                                   'worker_visitor_footfall_sqkm', lag_granularity,30, '2020-01-01',n_lag=12, plt_flg=True,moblty_flag=False)
    
# Make variables stationary if required
# split the data for each LSOA
dynamic_df_lsoa = [pd.DataFrame(y) for x, y in dynamic_df.groupby('LSOA11CD', as_index=False)]

# flag for whether to perform differencing on mobility and vaccination data
flg_stnrty_both=False  

mobility_vars=[s.replace('_norm_lag_area','') for s in  list(lag_values_mobility.keys())]  #get name of mobility columns which were lagged

vacc_vars=[s.replace('_norm_lag_pop','') for s in  list(lag_values_vacc.keys())]  #get name of vacc columns which were lagged

if flg_stnrty_both:
    # Perform first order differencing to achieve stationarity on mobility and vaccination data only
    dynamic_df_diff = [x.sort_values(by='Date')[['Date','LSOA11CD']+mobility_vars+vacc_vars].\
                                     set_index(cf.cols_not_to_lag).\
                                     diff().dropna() for x in dynamic_df_lsoa]
    
else:
    # Non-stationary versions of mobility and vaccination data
    dynamic_df_diff = [x.sort_values(by='Date')[['Date','LSOA11CD']+mobility_vars+vacc_vars].set_index(cf.cols_not_to_lag)\
                                       for x in dynamic_df_lsoa]
    
dynamic_df_diff = pd.concat(dynamic_df_diff,axis=0).reset_index()

# Concat cases data for all LSOAs
demo_cases_lsoa = pd.concat(dynamic_df_lsoa,axis=0)[['Date','LSOA11CD','COVID_Cases','cases_per_person', 'pct_infected_all_time',
                                                                       'cumsum_divided_area']].reset_index(drop=True)

# Join case, vaccination, and mobility data for all LSOAs
dynamic_df_diff = dynamic_df_diff.merge(demo_cases_lsoa,on=['Date','LSOA11CD'],how='inner')

# Apply time lag to variables

# Deimos - Use caclulated time lag values to apply lag and then concatenate into a dataframe
resident_lsoa_lagged_values = pd.concat(tl.split_df_apply_time_lag(dynamic_df_diff,[x for x in mobility_vars if 'resident' in x],
                                                       lag_values_mobility['resident_footfall_sqkm_norm_lag_area'],apply_lag=True),axis=0).reset_index() 

worker_visitor_lsoa_lagged_values = pd.concat(tl.split_df_apply_time_lag(dynamic_df_diff,[x for x in mobility_vars if 'worker' in x],
                                                       lag_values_mobility['worker_visitor_footfall_sqkm_norm_lag_area'],apply_lag=True),axis=0).reset_index()


#deimos trips

commute_lsoa_lagged_values = pd.concat(tl.split_df_apply_time_lag(dynamic_df_diff,[x for x in mobility_vars if 'commute' in x],
                                                       lag_values_mobility['commute_inflow_sqkm_norm_lag_area'],apply_lag=True),axis=0).reset_index()

other_lsoa_lagged_values = pd.concat(tl.split_df_apply_time_lag(dynamic_df_diff,[x for x in mobility_vars if 'other' in x],
                                                       lag_values_mobility['other_inflow_sqkm_norm_lag_area'],apply_lag=True),axis=0).reset_index()

# STORE ZERO LAGGED CASES DATA FOR ALL THE LSOA - No lag is happening here

cases_lsoa_lagged_all = pd.concat(tl.split_df_apply_time_lag(dynamic_df_diff,['COVID_Cases','cases_per_person', 'pct_infected_all_time','cumsum_divided_area'],
                                                           apply_lag=False),axis=0).reset_index()

dfs_lagged = [resident_lsoa_lagged_values,worker_visitor_lsoa_lagged_values,commute_lsoa_lagged_values,other_lsoa_lagged_values]  #list of dataframes where columns were lagged

mobility_vaccine_deimos_lag_merged=reduce(lambda left,right: pd.merge(left,right,on=['LSOA11CD','Date'],how='inner'), dfs_lagged)  #combine these lagged dataframes



mobility_vaccine_deimos_cases_lag_merged=mobility_vaccine_deimos_lag_merged.merge(cases_lsoa_lagged_all,\
                                                           on=['LSOA11CD','Date'],how='inner')  #merge cases with lagged values as cases was not lagged
 
mobility_vaccine_deimos_cases_lag_merged=mobility_vaccine_deimos_cases_lag_merged.merge(travel_clusters,on=['LSOA11CD'],how='inner')  #add travel cluster information

mobility_vaccine_deimos_cases_lag_merged.rename(columns={'COVID_Cases':'COVID_Cases_per_unit_area',
                "cases_per_person":"COVID_Cases_prop_population",
                "cumsum_divided_area":"COVID_Cases_per_unit_area_cumsum",
                "pct_infected_all_time":"COVID_Cases_prop_population_cumsum"}, inplace=True)  #change column names as required 

if flg_stnrty_both:
    mobility_vaccine_deimos_cases_lag_merged.to_gbq(cf.lagged_dynamic_stationary,\
                           project_id = project_name,if_exists='replace')
else:
    mobility_vaccine_deimos_cases_lag_merged.to_gbq(cf.lagged_dynamic_non_stationary,\
                                             project_id=project_name,if_exists='replace')
    
