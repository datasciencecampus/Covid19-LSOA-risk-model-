import numpy as np

####################
## Section A - Model input dataset locations
###################

# Define the location of LSOA 2011 GeoJSON file
geography_bucket = 'hotspot-prod-geodata'
lsoa_geojson_filename = 'LSOA_2011_EW_BSC.geojson'
msoa_geojson_filename = 'msoa.geojson'

# create a empty dictionary to store the file locations of input data sets
data_location_big_query = {}

# define the location of the input data sets on BigQuery
data_location_big_query['static'] = "ons-hotspot-prod.ingest_risk_model.risk_model_lsoa"
data_location_big_query['lsoa_midyear_population_2019'] = "ons-hotspot-prod.ingest_risk_model.mid_year_pop19_lsoa"
data_location_big_query['mobility_clusters_processed'] = "ons-hotspot-prod.ingest_geography.lsoa_mobility_cluster_ew_lu"
data_location_big_query['flow_to_work'] = "ons-hotspot-prod.review_ons.idbr_census_flowtowork_lsoa_highindustry"
data_location_big_query['cases'] = "ons-hotspot-prod.ingest_track_and_trace.aggregated_positive_tests_lsoa"
data_location_big_query['vaccination'] = "ons-hotspot-prod.ingest_vaccination.lsoa_vaccinations_new"
data_location_big_query['mobility_DEIMOS'] = "ons-hotspot-prod.review_ons.people_counts_df_lsoa_daily_latest"
data_location_big_query['deimos_aggregated'] = "ons-hotspot-prod.ingest_deimos_2021.uk_footfall_people_counts_ag"
data_location_big_query['lsoa_2001_to_2011_lookup'] = "ons-hotspot-prod.ingest_geography. lsoa_2001_to_2011_look_up"
data_location_big_query['lsoa_area'] = "ons-hotspot-prod.ingest_geography.lsoa_2011_area_centroid"

######################
## Section B - Location to write intermediate data sets created during the modelling process
######################

# Project ID for writing to BigQuery. This value is passed to the project_id argument of the to_gbq function
# from the pandas_gbq package
project_name = 'ons-hotspot-prod'

# Processed static data set
static_data_file = 'review_ons.risk_model_static_variables'

# Processed dynamic data sets 
dynamic_data_file = 'review_ons.dynamic_lsoa_variables'
dynamic_data_file_normalised = 'review_ons.dynamic_lsoa_variables_raw_norm_chsn_lag'

# Lagged data sets
lagged_dynamic_stationary = 'review_ons.time_lagged_dynamic_data_deimos_cumsum_stationary_main'
lagged_dynamic_non_stationary = 'review_ons.time_lagged_dynamic_data_deimos_cumsum_non_stationary_main'

# Tranches model inputs
tranches_model_input_processed = 'review_ons.tranches_model_input_processed'
tranches_model_test_data = 'review_ons.tranches_model_test_data'

# Tranche model coefs
tranche_coefs_regularisation = 'review_ons.tranches_coefs_regularisation'
tranche_coefs_standardised = 'review_ons.tranche_coefs_standardised'
tranche_coefs_non_standardised = 'review_ons.tranche_coefs_non_standardised'

# Tranche model predictions and residuals
tranche_preds_all_tranches = 'review_ons.tranche_preds'
tranche_preds_latest = 'review_ons.tranche_preds_latest'

# Processed outputs to be picked up by Google Data Studio dashboard
dashboard_tranche_coefs_regularisation = 'review_ons.dashboard_tranche_reg_coefs'
dashboard_tranche_coefs_standardised = 'review_ons.dashboard_tranche_non_reg_std_coefs'
dashboard_tranche_coefs_non_standardised = 'review_ons.dashboard_tranche_non_reg_non_std_coefs'
dashboard_feature_spatial_dist = 'review_ons.dashboard_tranche_model_features'
dashboard_tranche_residuals = 'review_ons.dashboard_tranche_residuals'
dashboard_tranche_latest_preds = 'review_ons.dashboard_tranche_latest_preds'

###############
## Section C - Global Parameters
##
## Options in this section affect the time tranches and the two way fixed effects model
###############

# select the model type to run
# valid options are 'time_tranche' or 'two_way_fixed_effects'
# any other value will throw an error
model_type = "time_tranche"

# start date from which the number of positive tests and vaccinations should be loaded
data_start_date = "'2020-04-01'"

# the total number of LSOAs in England
# assert statements in the modelling code check that every LSOA is present
n_lsoa = 32844

# Data sets for the 'read_data' function from prep_pipeline.py to iterate over and import
# from the Data Factory
# the values in each list are passed to the DataFactory.get() method in data_factory.py
data_tables = {
    
    # static features are shared by both models
     'static': ['static_vars', 'mid_year_lsoa', 'mobility_clusters_processed', 'flow_to_work', 'LSOA_2011']
    
    # two-way fixed effects model uses all of the dynamic features
    ,'dynamic': ['aggregated_tests_lsoa', 'lsoa_vaccinations', 'lsoa_daily_footfall']

}

# feature engineering
# the keys of this dictionary represent a new column to be created in the static data
# the new column is populated with the sum of the columns listed in the value of this dictionary
static_cols_to_sum = {'ready_meals_textiles': ['ready_meals', 'textiles'] 
                     ,'care_homes_warehousing': ['care', 'warehousing']
                     }

# unused columns to drop from the static data
static_col_drop = ['BAME_PROP',
 'STUDENT_LIVING_IN_A_COMMUNAL_ESTABLISHMENT_TOTAL',
 'COMMUNAL_ESTABLISHMENT_MEDICAL_AND_CARE_TOTAL',
 'CENSUS_2011_BLACK_AFRICAN_CARIBBEAN_BLACK_BRITISH',
 'HEALTH_AGE_50_to_64_BAD_HEALTH',
 'HEALTH_AGE_75_PLUS_BAD_HEALTH',
 'HEALTH_AGE_50_to_64_GOOD_FAIR_HEALTH',
 'METHOD_OF_TRAVEL_TO_WORK_PRIVATE_TRANSPORT',
 'IMD_SCORE',
 'HEALTH_AGE_UNDER_50_GOOD_FAIR_HEALTH',
 'HEALTH_AGE_65_to_74_BAD_HEALTH',
 'LSOA11NMW',
 'NO_UNPAID_CARE',
 'HEALTH_AGE_75_PLUS_GOOD_FAIR_HEALTH',
 'HOUSEHOLD_SIZE_1_PERSON_IN_HOUSEHOLD',
 'IMD_EMPLOYMENT_SCORE',
 'age_18_to_29',
 'HEALTH_AGE_65_to_74_GOOD_FAIR_HEALTH',
 'HEALTH_AGE_UNDER_50_BAD_HEALTH',
 'geometry',
 'FAMILIES_WITH_DEPENDENT_CHILDREN_ALL_FAMILIES',
 'HOUSEHOLD_SIZE_3_PLUS_PEOPLE_IN_HOUSEHOLD',
 'COMMUNAL_ESTABLISHMENT_MEDICAL_AND_CARE_CARE_HOMES',
 'COMMUNAL_ESTABLISHMENT_OTHER_PRISON_AND_OTHER_DETENTION',
 'UNPAID_CARE_1_HOUR_PLUS',
 'CENSUS_2011_WHITE', 'CENSUS_2011_MIXED_MULTIPLE_ETHINIC_GROUPS',
 'CENSUS_2011_OTHER_ETHNIC_GROUP',
 'HOUSEHOLD_SIZE_2_PEOPLE_IN_HOUSEHOLD',
 'HOUSEHOLD_SIZE_3_PEOPLE_IN_HOUSEHOLD',
 'SHARED_DWELLINGS_NUMBER_OF_PEOPLE',
 'COMMUNAL_ESTABLISHMENT_OTHER_EDUCATION',
 'COMMUNAL_ESTABLISHMENT_OTHER_HOSTEL_OR_TEMPORARY_SHELTER_FOR_THE_HOMELESS',
 'IMD_INCOME_SCORE',
 'STUDENT_LIVING_WITH_PARENTS', 
 'age_0_to_12', 
 'age_13_to_17',
 'age_30_to_39', 
 'age_40_to_49', 
 'age_50_to_54', 
 'age_55_to_59',
 'age_60_to_64', 
 'age_65_to_69', 
 'age_70_to_74', 
 'age_75_to_79',
 'age_80_to_90_PLUS',
 'warehousing_manc_def']

## Model parameters
## Number of different combinations of grid search hyperparameters
## Default is 500, use a lower value, >=1 to speed-up the
## evaluations at the cost of reduced search of the optimal
## parameters
param_search_space = 500

# Create a list of alphas for regularisation
alphas_val = np.logspace(-3, 3, 101)

###############
## Section D - Time Tranche Model Parameters
##
## Options in this section relate to the time tranches model only
#################

# select the number of tranches to model
n_tranches = 8

# define the dates on which to split into time tranches
tranche_dates = ['2020-04-26','2020-08-31','2020-11-14','2020-12-31','2021-02-14','2021-04-29','2021-07-15','2021-08-31']

# description of each tranche
# eg. period between '2020-04-26'to '2020-08-31' is of low prevalence and majority of schools closed in that period
tranche_description = ['low_prev_no_school',
                       'high_prev_school_opn',
                       'high_prev_school_opn_alph',
                       'high_prev_no_school_alph_vaccn',
                       'low_prev_school_opn_vaccn_dbl',
                       'high_prev_school_opn_dlta_vaccn_dbl',
                       'lifting_lockdown',
                       'high_prev_school_open_delta_vaccn']

# flag indicating whether to apply regularisation to the cost function for prediction
use_regularisation = True

# define a list of the features derived from the IDBR data set
# these features are engineered by summing other columns in the static data
tranche_model_idbr_features = ['care_homes_warehousing', 'ready_meals_textiles', 'meat_and_fish_processing']

###################
## Section E - Two Way Fixed Effect Model Parameters
##
## Options in this section affect the two way fixed effects model only
###################

# start date for the two way fixed effects model
# a more recent start date reduces model runtime
model_start_date = "2021-01-01"

# map to define which column to use for calculating the dynamic feature time lag
geography_dict = {'region':'RGN19NM',
              'travel_cluster':'travel_cluster',
              'utla':'UTLA20NM',
              'msoa':'MSOA11NM',
              'country':'Country'} 

# select the lag you want to use for caclulating dynamic time lag 
chosen_granularity_for_lag = geography_dict['travel_cluster']

# select 'travel_cluster' to model a specific travel cluster
granularity_for_modelling = geography_dict['country'] 

# select True to run a stationarity check on all dynamic features before calculating the
# optimal time lag
explore_stationarity_check = False

# columns of static data subset for geography
static_subset = ['LSOA11CD', 'ALL_PEOPLE', 'Area', 'travel_cluster', 'RGN19NM', 'UTLA20NM', 'MSOA11NM']

# listing which columns need to be forward filled for dynamic data processing
# these are cumulative sums which have been done over incomplete data
# therefore at the end of processing will have NaNs where the original data had no entry
# the forward fill deals with this issue
ffill_cols = {}

ffill_cols['dynamic_norm'] = ['cases_cumsum_norm_lag_pop', 'dbl_vacc_cumsum_norm_lag_pop', 'cases_cumsum_norm_lag_area']

ffill_cols['dynamic'] = ['cases_cumsum_pop_norm', 'dbl_vacc_cumsum_pop_norm', 'cases_cumsum_area_norm']

# drop columns which were replaced in the original dynamic preprocessing script
dynamic_col_drop = ['COVID_Cases', 'total_vaccinated_first_dose','total_vaccinated_second_dose']

# rename new columns in dynamic data preprocessing to match original column names
dynamic_rename = {
 'COVID_Cases_area_norm': 'COVID_Cases'
 ,'COVID_Cases_pop_norm': 'cases_per_person'
 ,'cases_cumsum_area_norm': 'cumsum_divided_area'
 ,'cases_cumsum_pop_norm': 'pct_infected_all_time'
 ,'dbl_vacc_cumsum_pop_norm': 'pct_of_people_full_vaccinated'
 ,'total_vaccinated_first_dose_pop_norm': 'total_vaccinated_first_dose'
 ,'total_vaccinated_second_dose_pop_norm': 'total_vaccinated_second_dose'}

# flag to select whether to use a zero-inflated model 
zero_infltd_modl = False

# Lag configuration
cols_not_to_lag = ['Date','LSOA11CD']  
mobility_cols_to_lag = ['worker_visitor_footfall_sqkm_norm_lag_area','resident_footfall_sqkm_norm_lag_area']
vacc_cols_to_lag = [] 


#modelling - These vars will be included in the dataset that undergoes modelling (they do not have to be included in the modelling itself if not wanted)
dynamic_vacc = []
dynamic_mobility = ['worker_visitor_footfall_sqkm','resident_footfall_sqkm']

# GCP dataset prefix for static and dynamic risk model outputs
# suffix will change for static/dynamic and whether zero inflation is applied
risk_coef = 'review_ons.multi_grp_coef'
risk_coef_ci = 'review_ons.multi_grp_coef_ci'
risk_pred = 'review_ons.multi_grp_pred'

model_suffixes = {
     'static_main': '_zir_only_static_main'
    ,'dynamic': '_zir_only_dynamic'
    ,'static_dynamic': '_zir_static_dynamic_sep_main'
}

for model in model_suffixes.keys():
    if not zero_infltd_modl:
        model_suffixes[model] = '_no' + model_suffixes[model]

#########################
## Section F - Feature normalisation config
##
## Options in this section define how the input features are normalised
#######################

# Features dictionary
# This is a nested dictionary of dictionaries to allow easy looping within functions
# when normalising datasets

features_dict = {}

features_dict['umbrella_ethnicity']={
     'flag': 'static'
    ,'by': None
    ,'suffix': ''
    ,'columns': ['CENSUS_2011_WHITE', 
              'CENSUS_2011_MIXED_MULTIPLE_ETHINIC_GROUPS',
              'CENSUS_2011_ASIAN_ASIAN_BRITISH', 
              'CENSUS_2011_BLACK_AFRICAN_CARIBBEAN_BLACK_BRITISH',
              'CENSUS_2011_OTHER_ETHNIC_GROUP']
}

features_dict['commnl_ftrs']={
     'flag': 'static'
    ,'by': 'Area'
    ,'suffix': ''
    ,'columns': ['COMMUNAL_ESTABLISHMENT_MEDICAL_AND_CARE_CARE_HOMES',
             'COMMUNAL_ESTABLISHMENT_OTHER_PRISON_AND_OTHER_DETENTION',
             'COMMUNAL_ESTABLISHMENT_OTHER_EDUCATION',
             'COMMUNAL_ESTABLISHMENT_OTHER_HOSTEL_OR_TEMPORARY_SHELTER_FOR_THE_HOMELESS',
              'STUDENT_LIVING_WITH_PARENTS'] #only keep granular ftrs
}

features_dict['shared_dwellings']={
     'flag': 'static'
    ,'by': 'Area'
    ,'suffix': ''
    ,'columns':
    ['SHARED_DWELLINGS_NUMBER_OF_PEOPLE'] 
}

features_dict['hh_ftrs_occupancy_rating_room']=['HH_OCCUPANCY_RATING_FOR_ROOMS_UNDERCROWEDED',
         'HH_OCCUPANCY_RATING_FOR_ROOMS_JUST_RIGHT',
         'HH_OCCUPANCY_RATING_FOR_ROOMS_OF_OVERCROWDED']

features_dict['hh_ftrs_occupancy_rating_bedroom']=['HH_OCCUPANCY_RATING_FOR_BEDROOMS_UNDERCROWDED',
         'HH_OCCUPANCY_RATING_FOR_BEDROOMS_JUST_RIGHT',
         'HH_OCCUPANCY_RATING_FOR_BEDROOMS_OVERCROWDED']

features_dict['fam_ftrs']={
     'flag': 'static'
    ,'by': 'FAMILIES_WITH_DEPENDENT_CHILDREN_ALL_FAMILIES'
    ,'suffix': ''
    ,'columns': ['FAMILIES_WITH_DEPENDENT_CHILDREN_NO_DEPENDENT_CHILDREN']
}

features_dict['good_health_ftrs']=['HEALTH_AGE_UNDER_50_GOOD_FAIR_HEALTH','HEALTH_AGE_50_to_64_GOOD_FAIR_HEALTH','HEALTH_AGE_65_to_74_GOOD_FAIR_HEALTH',
                     'HEALTH_AGE_75_PLUS_GOOD_FAIR_HEALTH']

features_dict['bad_health_ftrs']=['HEALTH_AGE_UNDER_50_BAD_HEALTH','HEALTH_AGE_50_to_64_BAD_HEALTH', 'HEALTH_AGE_65_to_74_BAD_HEALTH','HEALTH_AGE_75_PLUS_BAD_HEALTH']

features_dict['hh_sizes']=['HOUSEHOLD_SIZE_1_PERSON_IN_HOUSEHOLD',
         'HOUSEHOLD_SIZE_2_PEOPLE_IN_HOUSEHOLD',
         'HOUSEHOLD_SIZE_3_PEOPLE_IN_HOUSEHOLD',
         'HOUSEHOLD_SIZE_3_PLUS_PEOPLE_IN_HOUSEHOLD']

features_dict['occpn_ftrs']=['OCCUPATION_MANAGERS_DIRECTORS_AND_SENIOR_OFFICIALS',
            'OCCUPATION_SCEINCE_RESEARCH_ENGINEERING_AND_TECHNOLOGY_PROFESSIONALS',
            'OCCUPATION_HEALTH_PROFESSIONALS',
            'OCCUPATION_BUSINESS_MEDIA_AND_PUBLIC_PROFESSIONALS',
            'OCCUPATION_SCIENCE_ENGINEERING_AND_TECHNOLOGY_ASSOCIATE_PROFESSIONALS',
            'OCCUPATION_HEALTH_AND_SOCIAL_CARE_ASSOCIATE_PROFESSIONALS',
            'OCCUPATION_HEALTH_ASSOCIATE_PROFESSIONALS',
            'OCCUPATION_WELFARE_AND_HOUSING_ASSOCIATE_PROFESSIONALS',
            'OCCUPATION_PROTECTIVE_SERVICE_OCCUPATIONS',
            'OCCUPATION_CULTURE_MEDIA_AND_SPORTS_OCCUPATIONS',
            'OCCUPATION_BUSINESS_AND_PUBLIC_SERVICE_ASSOCIATE_PROFESSIONALS',
            'OCCUPATION_ADMINISTRATIVE_AND_SECRETARIAL_OCCUPATIONS',
            'OCCUPATION_SKILLED_AGRICULTURAL_AND_RELATED_TRADES',
            'OCCUPATION_SKILLED_METAL_ELECTRICAL_AND_ELECTRICAL_TRADES',
            'OCCUPATION_SKILLED_CONSTRUCTION_AND_BUILDING_TRADES',
            'OCCUPATION_PRINTING_TRADES',
            'OCCUPATION_FOOD_PREPARATION_AND_HOSPITALITY_TRADES',
            'OCCUPATION_CARING_PERSONAL_SERVICE_OCCUPATIONS',
            'OCCUPATION_LEISURE_TRAVEL_AND_RELATED_PERSONAL_SERVICE_OCCUPATIONS',
            'OCCUPATION_SALES_OCCUPATIONS',
            'OCCUPATION_CUSTOMER_SERVICE_OCCUPATIONS',
            'OCCUPATION_PROCESS_PLANT_AND_MACHINE_OPERATIVES',
            'OCCUPATION_TRANSPORT_AND_MOBILE_MACHINE_DRIVERS_AND_OPERATIVES']

features_dict['occpn_groups']=['1_prof_other','1_prof_healthcare',"1_Group_skilled","1_Group_trade"]

features_dict['age_features']={
     'flag': 'static'
    ,'by': 'ALL_PEOPLE'
    ,'suffix': ''    
    ,'columns': ['age_0_to_12','age_13_to_17','age_18_to_29','age_30_to_39','age_40_to_49',
                               'age_50_to_54','age_55_to_59','age_60_to_64','age_65_to_69','age_70_to_74',
                               'age_75_to_79','age_80_to_90_PLUS']
}


features_dict['trvl_ftrs']={
     'flag': 'static'
    ,'by': None
    ,'suffix': '' 
    ,'columns': ['METHOD_OF_TRAVEL_TO_WORK_WORK_MAINLY_FROM_HOME', 'METHOD_OF_TRAVEL_TO_WORK_Public_TRANSPORT',
               'METHOD_OF_TRAVEL_TO_WORK_NON_MOTORISED', 'METHOD_OF_TRAVEL_TO_WORK_PRIVATE_TRANSPORT']
}

features_dict['ftw_cols']={
     'flag': 'static'
    ,'by': 'Area'
    ,'suffix': ''
    ,'columns':['care','meat_and_fish_processing','ready_meals','textiles','warehousing']
}

features_dict['hh_sizes']={
     'flag': 'static'
    ,'by': None
    ,'suffix': ''
    ,'columns': ['HOUSEHOLD_SIZE_1_PERSON_IN_HOUSEHOLD', 'HOUSEHOLD_SIZE_2_PEOPLE_IN_HOUSEHOLD',
         'HOUSEHOLD_SIZE_3_PEOPLE_IN_HOUSEHOLD', 'HOUSEHOLD_SIZE_3_PLUS_PEOPLE_IN_HOUSEHOLD']
}

features_dict['reference_categories']=["CENSUS_2011_BLACK_AFRICAN_CARIBBEAN_BLACK_BRITISH",'METHOD_OF_TRAVEL_TO_WORK_WORK_MAINLY_FROM_HOME','GOOD_FAIR_HEALTH','HOUSEHOLD_SIZE_1_PERSON_IN_HOUSEHOLD']

features_dict['reference_category_ethnicity']=["CENSUS_2011_BLACK_AFRICAN_CARIBBEAN_BLACK_BRITISH"]
features_dict['reference_category_travel_work']=['METHOD_OF_TRAVEL_TO_WORK_PRIVATE_TRANSPORT']
features_dict['reference_category_hh_size']=['HOUSEHOLD_SIZE_1_PERSON_IN_HOUSEHOLD']


features_dict['etnc_ftrs']=['CENSUS_2011_WHITE',
           'CENSUS_2011_MIXED_MULTIPLE_ETHINIC_GROUPS',
           'CENSUS_2011_ASIAN_ASIAN_BRITISH',
           'CENSUS_2011_BLACK_AFRICAN_CARIBBEAN_BLACK_BRITISH',# REFERENCE CATEGORY 
           'CENSUS_2011_OTHER_ETHNIC_GROUP']

features_dict['trvl_ftrs']={
     'flag': 'static'
    ,'by': None
    ,'suffix': ''    
    ,'columns': ['METHOD_OF_TRAVEL_TO_WORK_NON_MOTORISED',
           'METHOD_OF_TRAVEL_TO_WORK_PRIVATE_TRANSPORT',
           'METHOD_OF_TRAVEL_TO_WORK_Public_TRANSPORT',
           'METHOD_OF_TRAVEL_TO_WORK_WORK_MAINLY_FROM_HOME']
}

features_dict['imd_ftrs']=['IMD_INCOME_SCORE']

features_dict['paid_ftrs']={
     'flag': 'static'
    ,'by': None
    ,'suffix': ''
    ,'columns': ['NO_UNPAID_CARE','UNPAID_CARE_1_HOUR_PLUS']
}

features_dict['health_ftrs']=['GOOD_FAIR_HEALTH','BAD_HEALTH']


features_dict['deimos_cols']=['worker_footfall_sqkm',"visitor_footfall_sqkm","resident_footfall_sqkm","total_footfall_sqkm"]
features_dict['non_deimos']=['COVID_Cases','total_vaccinated_first_dose']

# dynamic features

features_dict['dynamic_pop_norm'] = {
     'flag': 'dynamic_norm'
    ,'by': 'Population_chosen_geo'
    ,'suffix': '_norm_lag_pop'
    ,'columns': ['total_vaccinated_first_dose','total_vaccinated_second_dose', 'dbl_vacc_cumsum',
                'COVID_Cases', 'cases_cumsum']
}

features_dict['dynamic_area_norm'] = {
     'flag': 'dynamic_norm'
    ,'by': 'Area_chosen_geo'
    ,'suffix': '_norm_lag_area'
    ,'columns': ['COVID_Cases', 'cases_cumsum', 
                'worker_footfall_sqkm',
                'visitor_footfall_sqkm',
                'resident_footfall_sqkm',
                'total_footfall_sqkm',
                'worker_visitor_footfall_sqkm']
}

features_dict['dynamic_pop'] = {
     'flag': 'dynamic'
    ,'by': 'ALL_PEOPLE'
    ,'suffix': '_pop_norm'
    ,'columns': ['total_vaccinated_first_dose','total_vaccinated_second_dose', 'dbl_vacc_cumsum',
                'COVID_Cases', 'cases_cumsum']
}

features_dict['dynamic_area'] = {
     'flag': 'dynamic'
    ,'by': 'Area'
    ,'suffix': '_area_norm'
    ,'columns': ['COVID_Cases', 'cases_cumsum']
}


###################
## Section G - Define column renames for the Google Data Studio dashboard
##
## Options in this section tell the pipeline how to rename features from
## the model outputs so that they appear in the Google Data Studio dashboard
##################


## Pretty feature names for dashboard
feature_pretty_names = {
                       # processed features (factor scores and normalised values) 
                       'IMD_INCOME_SCORE':'IMD Income Score',
                       'imd_income_score':'IMD Income Score',
                       'METHOD_OF_TRAVEL_TO_WORK_Public_TRANSPORT':'Commute Method - Public Transport',
                       'method_of_travel_to_work_public_transport':'Commute Method - Public Transport',
                       'METHOD_OF_TRAVEL_TO_WORK_WORK_MAINLY_FROM_HOME':'Commute Method - Work From Home',
                       'method_of_travel_to_work_work_mainly_from_home':'Commute Method - Work From Home',
                       'STUDENT_LIVING_WITH_PARENTS':'Student Living With Parents',
                       'student_living_with_parents':'Student Living With Parents',
                       'care_homes_and_workers':'Care Home and Workers',
                       'high_risk_industry_workers':'High Risk Industry Workers',
                       'middle_age_groups_medium_hh':'Middle Age Groups Medium HH Size',
                       'non_white_pop_larger_hh':'Non White Population, Larger HH Size',
                       'older_pop_smaller_hh':'Older Population, Smaller HH Size',
                       'smaller_hh_no_children':'Smaller HH Size, No Children',
                    
                        # simpler model features
                       'census_2011_asian_asian_british':'Asian/Asian British',
                       'CENSUS_2011_ASIAN_ASIAN_BRITISH':'Asian/Asian British',
                       'method_of_travel_to_work_non_motorised':'Commute Method - Non Motorised',
                       'METHOD_OF_TRAVEL_TO_WORK_NON_MOTORISED':'Commute Method - Non Motorised',
                       'families_with_dependent_children_no_dependent_children':'No Dependent Children',
                       'FAMILIES_WITH_DEPENDENT_CHILDREN_NO_DEPENDENT_CHILDREN':'No Dependent Children',
                       'care_homes_warehousing':'Care, Warehousing Workers',
                       'meat_and_fish_processing':'Meat & Fish Processing Workers',
                       'ready_meals_textiles':'Ready Meals & Textile Workers',
                       'worker_visitor_footfall_sqm':'Worker Visitor Footfall',
    
                        # quintiles
                       'IMD_INCOME_SCORE_quint':'IMD Income Score Quintile',
                       'METHOD_OF_TRAVEL_TO_WORK_Public_TRANSPORT_quint':'Commute Method - Public Transport Quintile',
                       'METHOD_OF_TRAVEL_TO_WORK_WORK_MAINLY_FROM_HOME_quint':'Commute Method - Work From Home Quintile',
                       'STUDENT_LIVING_WITH_PARENTS_quint':'Student Living With Parents Quintile',
                       'care_homes_and_workers_quint':'Care Home and Workers Quintile',
                       'high_risk_industry_workers_quint':'High Risk Industry Workers Quintile',
                       'middle_age_groups_medium_hh_quint':'Middle Age Groups Medium HH Size Quintile',
                       'non_white_pop_larger_hh_quint':'Non White Population, Larger HH Size Quintile',
                       'older_pop_smaller_hh_quint':'Older Population, Smaller HH Size Quintile',
                       'smaller_hh_no_children_quint':'Smaller HH Size, No Children Quintile',
                       'CENSUS_2011_ASIAN_ASIAN_BRITISH_quint':'Asian/Asian British Quintile',
                       'FAMILIES_WITH_DEPENDENT_CHILDREN_NO_DEPENDENT_CHILDREN_quint':'No Dependent Children Quintile',
                       'METHOD_OF_TRAVEL_TO_WORK_NON_MOTORISED_quint':'Commute Method - Non Motorised Quintile',
                       'meat_and_fish_processing_quint':'Meat & Fish Processing Workers Quintile',
                       'care_homes_warehousing_quint':'Care, Warehousing Workers Quintile',
                       'worker_visitor_footfall_sqm_quint':'Worker Visitor Footfall Quintile',
                       'ready_meals_textiles_quint':'Ready Meals & Textile Workers Quintile',
    
                        # ready meals rank
                       'ready_meals_rank':'Ready Meals Workers Rank'
                         }

# Travel cluster pretty names for dashboard
tc_pretty_names = {'L1. >70% metropolitan core dwellers':' >70% Metropilitan Core Dwellers',
                  'L2. >70% outer metropolitan dwellers':'>70% Outer Metropolitan Core Dwellers',
                  'L3. >70% suburban dwellers':'>70% Suburban Dwellers',
                  'L4. >70% exurban dwellers':'>70% Exurban Dwellers',
                  'L5. >70% rural dwellers':'>70% Rural Dwellers'}

# Travel cluster short names for dashboard
tc_short_names = {'L1. >70% metropolitan core dwellers':'Metro Core',
                  'L2. >70% outer metropolitan dwellers':'Outer Metro',
                  'L3. >70% suburban dwellers':'Suburban',
                  'L4. >70% exurban dwellers':'Exurban',
                  'L5. >70% rural dwellers':'Rural'}