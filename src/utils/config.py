import numpy as np


chsen_datum = "'2020-04-01'"  #date to select data from

model_start_date = "2021-01-01" #date to begin modelling

geography_dict = {'region':'RGN19NM',
              'travel_cluster':'travel_cluster',
              'utla':'UTLA20NM',
              'msoa':'MSOA11NM',
              'country':'Country'} #config for the geogaphy to calculate time lag.

chosen_granularity_for_lag = geography_dict['travel_cluster']#pick which lag you want to use for caclulating dynamic time lag 
granularity_for_modelling = geography_dict['country']
explore_stationarity_check = False

# columns of static data subset for geography
static_subset = ['LSOA11CD', 'ALL_PEOPLE', 'Area', 'travel_cluster', 'RGN19NM', 'UTLA20NM', 'MSOA11NM']

# dictionary of datasets to read in 
# these link to the DataFactory.get() method in data_factory.py
data_tables = {
     'static': ['static_vars', 'mid_year_lsoa', 'mobility_clusters_processed', 'flow_to_work', 'LSOA_2011']
    ,'dynamic': ['aggregated_tests_lsoa', 'lsoa_vaccinations', 'Deimos_trip_end_count', 'lsoa_daily_footfall']
}

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

# 'OCCUPATION_TEXTILES_AND_GARMENTS_TRADES',

features_dict['occpn_groups']=['1_prof_other','1_prof_healthcare',"1_Group_skilled","1_Group_trade"]

# features_dict['age_features']=['age_0_to_3','age_4_to_11','age_12_to_18','age_19_to_30','age_31_to_50','age_51_to_70','age_71_to_90_PLUS']


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
    ,'columns': ['total_vaccinated_first_dose','total_vaccinated_second_dose', 'full_vacc_cumsum',
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
                'worker_visitor_footfall_sqkm',
                'commute_inflow_sqkm',
                'other_inflow_sqkm']
}

features_dict['dynamic_pop'] = {
     'flag': 'dynamic'
    ,'by': 'ALL_PEOPLE'
    ,'suffix': '_pop_norm'
    ,'columns': ['total_vaccinated_first_dose','total_vaccinated_second_dose', 'full_vacc_cumsum',
                'COVID_Cases', 'cases_cumsum']
}

features_dict['dynamic_area'] = {
     'flag': 'dynamic'
    ,'by': 'Area'
    ,'suffix': '_area_norm'
    ,'columns': ['COVID_Cases', 'cases_cumsum']
}

# listing which columns need to be forward filled for dynamic data processing
# these are cumulative sums which have been done over incomplete data
# therefore at the end of processing will have NaNs where the original data had no entry
# the forward fill deals with this issue
ffill_cols = {}

ffill_cols['dynamic_norm'] = ['cumsum_norm_lag_pop', 'full_vacc_cumsum_norm_lag_pop', 'cases_cumsum_norm_lag_area']

ffill_cols['dynamic'] = ['pct_infected_all_time', 'pct_of_people_full_vaccinated', 'cumsum_divided_area']



# User inputs the location of data they wish to use 

data_location_big_query = {}

data_location_big_query['cases'] = "ons-hotspot-prod.ingest_track_and_trace.aggregated_positive_tests_lsoa"

data_location_big_query['vaccination'] = "ons-hotspot-prod.ingest_vaccination.lsoa_vaccinations_8_july_2021"

data_location_big_query['mobility_MARS'] = "wip.mars_daily_trips_to_and_from_home_2"

data_location_big_query['mobility_DEIMOS'] = "ons-hotspot-prod.wip.people_counts_df_lsoa_daily_latest"

# file names
project_name = 'ons-hotspot-prod'

static_data_file = 'wip.static_lsoa_variables'
dynamic_data_file = 'wip.dynamic_lsoa_variables'
dynamic_data_file_normalised = 'wip.dynamic_lsoa_variables_raw_norm_chsn_lag'

lagged_dynamic_stationary = 'wip.time_lagged_dynamic_data_deimos_cumsum_stationary_main'
lagged_dynamic_non_stationary = 'wip.time_lagged_dynamic_data_deimos_cumsum_non_stationary_main'


# zero-inflated model training
zero_infltd_modl=False

## Model parameters
## Number of different combinations of grid search hyperparameters
## Default is 500, use a lower value, >=1 to speed-up the
## evaluations at the cost of reduced search of the optimal
## parameters
parm_spce_grid_srch=500

# Create a list of alphas for regularisation
alphas_val = np.logspace(-3, 3, 101)


#Lag_configurations
cols_not_to_lag=['Date','LSOA11CD']  
mobility_cols_to_lag=['worker_visitor_footfall_sqkm_norm_lag_area','resident_footfall_sqkm_norm_lag_area','commute_inflow_sqkm_norm_lag_area','other_inflow_sqkm_norm_lag_area']
vacc_cols_to_lag=[] 


#modelling - These vars will be included in the dataset that undergoes modelling (they do not have to be included in the modelling itself if not wanted)
dynamic_vacc=[]
dynamic_mobility=['worker_visitor_footfall_sqkm','resident_footfall_sqkm','commute_inflow_sqkm','other_inflow_sqkm']

# GCP dataset prefix for static and dynamic risk model outputs
# suffix will change for static/dynamic and whether zero inflation is applied
risk_coef = 'wip.multi_grp_coef'
risk_coef_ci = 'wip.multi_grp_coef_ci'
risk_pred = 'wip.multi_grp_pred'

model_suffixes = {
     'static_main': '_zir_only_static_main'
    ,'dynamic': '_zir_only_dynamic'
    ,'static_dynamic': '_zir_static_dynamic_sep_main'
}

for model in model_suffixes.keys():
    if not zero_infltd_modl:
        model_suffixes[model] = '_no' + model_suffixes[model]