import os
import sys
import numpy as np
import pandas as pd
import pandas_gbq

# import from local data files
current_path = os.path.abspath('.')
sys.path.append(os.path.dirname(current_path))

sys.path.append(current_path + '/src')

from data_access.data_factory import DataFactory as factory
from data_access import prep_pipeline as pp
from utils import data as dt
from utils import config as cf

print("Loading static data from BigQuery...")

# read in static data
static_df_raw = pp.read_data('static')

print("Processing static data...")

# merge geography polygon data to find LSOA areas
static_df = pp.geo_merge_precalc(static_df_raw)

# normalise static data
static_df = pp.normalise_data(static_df, 'static')
    
# remove ethnicity subgroups
ethnicity_list = dt.get_ethnicities_list(static_df, subgroups=True)
static_df = static_df.drop(columns=ethnicity_list) 

#fill 0s (NaNs exist for industries in which zero residents of a given LSOA work)
static_df = static_df.fillna(0)  

static_df.drop(columns=cf.static_col_drop, inplace=True)

# combine the flow to work columns from factor analysis
static_df = pp.sum_features(static_df)

# pre-processing for the two way fixed effects model
if cf.model_type == "two_way_fixed_effects":
    
    print("Loading dynamic data for two-way fixed effects model...")
    
    dynamic_df = pp.read_data('dynamic', join_col=['LSOA11CD', 'Date'])

    # join on subset of static data for geographic variables
    col_list = cf.static_subset
    static_subset_df = static_df[col_list]   
    dynamic_df = dynamic_df.merge(static_subset_df, on=['LSOA11CD'], how='right')

    # date filter due to join being changed to outer resulting in extraneous rows prior to the pandemic
    dynamic_df = dynamic_df[dynamic_df['Date'] >= '2020-10-04']
    
    # Filter to England only
    dynamic_df = dynamic_df[dynamic_df.LSOA11CD.str.startswith('E')] 
    dynamic_df = dynamic_df.fillna(0)

    dynamic_df['Country'] = 'England'

    # Normalise population by a common geography so lag values in following code can be calculated correctly
    lag_granularity = cf.chosen_granularity_for_lag
    dynamic_df_norm = dynamic_df.copy()

    df_travel_clusters = dynamic_df_norm.drop_duplicates(subset='LSOA11CD', keep='first')[[lag_granularity,'Area','ALL_PEOPLE']].groupby(lag_granularity).sum().reset_index()\
    .rename(columns={'Area':'Area_chosen_geo', 'ALL_PEOPLE':'Population_chosen_geo'})

    dynamic_df_norm = dynamic_df_norm.merge(df_travel_clusters, how='left', on=lag_granularity)

    # convert back to raw so we can divide by travel cluster area
    for i in [i for i in dynamic_df_norm.columns.tolist() if (('footfall' in i) | ('inflow' in i))]:
        dynamic_df_norm[i] = dynamic_df_norm[i] * dynamic_df_norm['Area']  

    # normalise dynamic data
    dynamic_df_norm = pp.normalise_data(dynamic_df_norm, 'dynamic_norm')

    dynamic_df_norm = pp.ffill_cumsum(dynamic_df_norm, cf.ffill_cols['dynamic_norm'])

    # normalise the original dynamic df
    dynamic_df = pp.normalise_data(dynamic_df, 'dynamic')
    dynamic_df = pp.ffill_cumsum(dynamic_df, cf.ffill_cols['dynamic'])

    dynamic_df.drop(columns=cf.dynamic_col_drop, inplace=True)
    dynamic_df.rename(columns=cf.dynamic_rename, inplace=True)

    # write results to BigQuery
    dynamic_df.to_gbq(cf.dynamic_data_file, project_id=cf.project_name, if_exists='replace')
    dynamic_df_norm.to_gbq(cf.dynamic_data_file_normalised, project_id=cf.project_name, if_exists='replace')

    # apply the calculated time lag
    df_final = pp.apply_timelag(dynamic_df, dynamic_df_norm)
    
# pre-processing for time tranches model
elif cf.model_type == "time_tranche":
    
    # the time tranches model doesn't need any further normalisation by population
    static_df_dropped = static_df.drop('ALL_PEOPLE', axis=1)
    
    print("Joining cases data...")
        
    # join cases to the static data
    cases_all_weeks_df = pp.join_cases_to_static_data(static_df_dropped)
    
    # generate a 'week number' column
    cases_all_weeks_df = pp.derive_week_number(cases_all_weeks_df)
    
    print("Joining vaccination data...")
    
    df_vax = factory.get('lsoa_vaccinations').create_dataframe()
    
    vax_processed_df, cases_all_weeks_df = pp.join_vax_data(cases_all_weeks_df, df_vax)
    
    print("Loading mobility data from BigQuery...")
    
    # load the mobility data from BigQuery
    deimos_footfall_df = factory.get('lsoa_daily_footfall').create_dataframe()
    
    # load and process mobility data
    cases_mobility_all_weeks_df = pp.join_tranches_mobility_data(cases_all_weeks_df, deimos_footfall_df)
    
    # convert mobility metric units from square kilometres to square metres
    cases_mobility_all_weeks_df = pp.convert_units(df = cases_mobility_all_weeks_df, 
                                                   colname = 'worker_visitor_footfall_sqkm',
                                                   factor = 0.000001,
                                                   new_colname = 'worker_visitor_footfall_sqm')
    
    # import list of IDBR features from the config file
    idbr_features = cf.tranche_model_idbr_features
    
    # loop over IDBR features converting from square kilometres to hectares
    for feature in idbr_features:
        
        cases_mobility_all_weeks_df = pp.convert_units(df = cases_mobility_all_weeks_df, 
                                                       colname = feature, 
                                                       factor = 0.01)
    
    # define dataframe for normalisation by population
    pop_df = static_df_raw[['LSOA11CD','ALL_PEOPLE']]
    
    cases_mobility_all_weeks_df = cases_mobility_all_weeks_df.merge(pop_df, how='inner', on=['LSOA11CD'])
    
    # express vaccinations as proportion of LSOA population
    cases_mobility_all_weeks_df['total_vaccinated_first_dose'] = cases_mobility_all_weeks_df['total_vaccinated_first_dose'].div(cases_mobility_all_weeks_df['ALL_PEOPLE'])
    cases_mobility_all_weeks_df['total_vaccinated_second_dose'] = cases_mobility_all_weeks_df['total_vaccinated_second_dose'].div(cases_mobility_all_weeks_df['ALL_PEOPLE'])
    
    cases_mobility_all_weeks_df.drop('ALL_PEOPLE', axis=1, inplace=True)
    
    # generate the test data set - weeks for which we have mobility and vaccination data but no cases data
    test_df = pp.create_test_data(all_weeks_df = cases_mobility_all_weeks_df, 
                                  static_df = static_df, 
                                  deimos_footfall_df = deimos_footfall_df, 
                                  vax_processed_df = vax_processed_df)
    
    print("Organising the data into time tranches...")
    
    # split the data into tranches
    tranches_df = pp.create_time_tranches(cases_mobility_all_weeks_df)
    
    # create a 'tranche_order' column for plotting later
    tranches_df = pp.derive_tranche_order(tranches_df)
    
    # create excess vaccinations feature
    tranches_df['vax_2_minus_1'] = tranches_df['total_vaccinated_second_dose'] - tranches_df['total_vaccinated_first_dose']   
    tranches_df.drop(['total_vaccinated_first_dose', 'total_vaccinated_second_dose'], axis=1, inplace=True)
    
    print("Writing tranche model training and test data to BigQuery...")
    
    # write the results to BigQuery 
    tranches_df.to_gbq(cf.tranches_model_input_processed, project_id=cf.project_name, if_exists='replace')
    test_df.to_gbq(cf.tranches_model_test_data, project_id=cf.project_name, if_exists='replace')
        
else:
    raise ValueError('The ''model_type'' provided in the config file is invalid. See config file for options.')
    

# drop columns for the version of the file that will be picked up by 'AllTranches' in the TWFE modelling phase
static_df.drop(['Area','ALL_PEOPLE'], axis=1, inplace=True)

# write to BigQuery
static_df.to_gbq(cf.static_data_file, project_id=cf.project_name, if_exists='replace')
