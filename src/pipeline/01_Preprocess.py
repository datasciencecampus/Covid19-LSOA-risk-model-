# Preprocessing
# Sections: Read in static data; read in dynamic data; apply time lag to dynamic datasets for modelling

# Import Packages
import os
import sys

import pandas as pd
import pandas_gbq

# Import from local data files
current_path = os.path.abspath('.')
sys.path.append(os.path.dirname(current_path))

from data_access.data_factory import DataFactory as factory
from data_access import prep_pipeline as pp
from utils import data as dt
from utils import config as cf

#############################

# read in static data
static_df = pp.read_data('static')
static_df = pp.geo_merge(static_df)

# normalise static data
static_df = pp.normalise_data(static_df, 'static')
    
# remove ethnicity subgroups
ethnicity_list = dt.get_ethnicities_list(static_df,subgroups=True)
static_df = static_df.drop(columns=ethnicity_list) 

static_df=static_df.fillna(0)  #fill 0s(nans exist in occupations where nobody works in them)

# apply_vif_statistic notebook is no longer being used
# but this still filters the columns in the final output

static_df.drop(columns=cf.static_col_drop, inplace=True)

# save to wip file
static_df.to_gbq(cf.static_data_file, project_id=cf.project_name,if_exists='replace')

# read in dynamic data
# shift up to be with static etc?
    
dynamic_df = pp.read_data('dynamic', join_col=['LSOA11CD', 'Date'])

# join on subset of static data for geographic variables
col_list = cf.static_subset
static_subset_df = static_df[col_list]   
dynamic_df = dynamic_df.merge(static_subset_df,on=['LSOA11CD'],how='right')

# date filter due to join being changed to outer resulting in extraneous rows prior to the pandemic
dynamic_df = dynamic_df[dynamic_df['Date'] >= '2020-10-04']
    
dynamic_df = dynamic_df.fillna(0)

dynamic_df['Country'] = 'England'

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
dynamic_df_norm = pp.normalise_data(dynamic_df_norm, 'dynamic_norm')

dynamic_df_norm = pp.ffill_cumsum(dynamic_df_norm, cf.ffill_cols['dynamic_norm'])

# normalise the original dynamic df
dynamic_df = pp.normalise_data(dynamic_df, 'dynamic')

dynamic_df = pp.ffill_cumsum(dynamic_df, cf.ffill_cols['dynamic'])
                       
# TODO: rename columns (or change subsequent code to use the new ones...)
# also think of better suffix for these columns

# rename columns to be in line with old code 
# may want to remove this later
dynamic_df.drop(columns=cf.dynamic_col_drop, inplace=True)
dynamic_df.rename(columns=cf.dynamic_rename, inplace=True)


dynamic_df.to_gbq(cf.dynamic_data_file, project_id=cf.project_name, if_exists='replace')
dynamic_df_norm.to_gbq(cf.dynamic_data_file_normalised, project_id=cf.project_name, if_exists='replace')

########################
# lag section
# writing to gbq is included in the function

df_final = pp.apply_timelag(dynamic_df, dynamic_df_norm)


    
    
    
    
    
    
    