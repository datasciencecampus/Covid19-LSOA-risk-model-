read_data_table_dict =  {
     'static_test': ['unit_test_static_vars', 'unit_test_mid_year_lsoa', 'unit_test_mobility_clusters', 'unit_test_flow_to_work', 'unit_test_lsoa_2011']
    ,'dynamic_test': ['unit_test_cases', 'unit_test_deimos']
}

static_columns = ['LSOA11CD', 'LSOA11NM', 'MSOA11CD', 'MSOA11NM', 'LTLA20CD', 'LTLA20NM',
       'UTLA20CD', 'UTLA20NM', 'RGN19CD', 'RGN19NM',
       'HOUSEHOLD_SIZE_3_PLUS_PEOPLE_IN_HOUSEHOLD',
       'COMMUNAL_ESTABLISHMENT_MEDICAL_AND_CARE_CARE_HOMES',
       'COMMUNAL_ESTABLISHMENT_OTHER_PRISON_AND_OTHER_DETENTION',
       'METHOD_OF_TRAVEL_TO_WORK_Public_TRANSPORT',
       'METHOD_OF_TRAVEL_TO_WORK_NON_MOTORISED',
       'METHOD_OF_TRAVEL_TO_WORK_PRIVATE_TRANSPORT', 'NO_UNPAID_CARE',
       'UNPAID_CARE_1_HOUR_PLUS', 'HEALTH_AGE_UNDER_50_GOOD_FAIR_HEALTH',
       'HEALTH_AGE_UNDER_50_BAD_HEALTH', 'HEALTH_AGE_50_to_64_BAD_HEALTH',
       'HEALTH_AGE_50_to_64_GOOD_FAIR_HEALTH',
       'HEALTH_AGE_65_to_74_GOOD_FAIR_HEALTH',
       'HEALTH_AGE_65_to_74_BAD_HEALTH', 'HEALTH_AGE_75_PLUS_GOOD_FAIR_HEALTH',
       'HEALTH_AGE_75_PLUS_BAD_HEALTH', 'CENSUS_2011_WHITE',
       'CENSUS_2011_WHITE_ENGLISH_WELSH_SCOTTISH_NORTHERN_IRISH_BRITISH',
       'CENSUS_2011_WHITE_IRISH', 'CENSUS_2011_WHITE_GYSPY_OR_IRISH_TRAVELLER',
       'CENSUS_2011_WHITE_OTHER_WHITE',
       'CENSUS_2011_MIXED_MULTIPLE_ETHINIC_GROUPS',
       'CENSUS_2011_MIXED_MULTIPLE_ETHINIC_GROUPS_WHITE_AND_BLACK_CARIBBEAN',
       'CENSUS_2011_MIXED_MULTIPLE_ETHINIC_GROUPS_WHITE_AND_BLACK_AFRICAN',
       'CENSUS_2011_MIXED_MULTIPLE_ETHINIC_GROUPS_WHITE_AND_ASIAN',
       'CENSUS_2011_MIXED_MULTIPLE_ETHINIC_GROUPS_OTHER_MIXED',
       'CENSUS_2011_ASIAN_ASIAN_BRITISH',
       'CENSUS_2011_ASIAN_ASIAN_BRITISH_INDIAN',
       'CENSUS_2011_ASIAN_ASIAN_BRITISH_PAKISTANI',
       'CENSUS_2011_ASIAN_ASIAN_BRITISH_BANGLADESHI',
       'CENSUS_2011_ASIAN_ASIAN_BRITISH_CHINESE',
       'CENSUS_2011_ASIAN_ASIAN_BRITISH_OTHER_ASIAN',
       'CENSUS_2011_BLACK_AFRICAN_CARIBBEAN_BLACK_BRITISH',
       'CENSUS_2011_BLACK_AFRICAN_CARIBBEAN_BLACK_BRITISH_AFRICAN',
       'CENSUS_2011_BLACK_AFRICAN_CARIBBEAN_BLACK_BRITISH_CARIBBEAN',
       'CENSUS_2011_BLACK_AFRICAN_CARIBBEAN_BLACK_BRITISH_OTHER_BLACK',
       'CENSUS_2011_OTHER_ETHNIC_GROUP', 'CENSUS_2011_OTHER_ETHNIC_GROUP_ARAB',
       'CENSUS_2011_OTHER_ETHNIC_GROUP_ANY_OTHER_ETHNIC_GROUP',
       'HOUSEHOLD_SIZE_1_PERSON_IN_HOUSEHOLD',
       'HOUSEHOLD_SIZE_2_PEOPLE_IN_HOUSEHOLD',
       'HOUSEHOLD_SIZE_3_PEOPLE_IN_HOUSEHOLD', 'BAME_PROP',
       'SHARED_DWELLINGS_NUMBER_OF_PEOPLE',
       'COMMUNAL_ESTABLISHMENT_MEDICAL_AND_CARE_TOTAL',
       'COMMUNAL_ESTABLISHMENT_OTHER_EDUCATION',
       'COMMUNAL_ESTABLISHMENT_OTHER_HOSTEL_OR_TEMPORARY_SHELTER_FOR_THE_HOMELESS',
       'FAMILIES_WITH_DEPENDENT_CHILDREN_NO_DEPENDENT_CHILDREN',
       'FAMILIES_WITH_DEPENDENT_CHILDREN_ALL_FAMILIES',
       'METHOD_OF_TRAVEL_TO_WORK_WORK_MAINLY_FROM_HOME', 'IMD_SCORE',
       'IMD_INCOME_SCORE', 'IMD_EMPLOYMENT_SCORE',
       'STUDENT_LIVING_WITH_PARENTS',
       'STUDENT_LIVING_IN_A_COMMUNAL_ESTABLISHMENT_TOTAL', 'ALL_PEOPLE',
       'age_0_to_12', 'age_13_to_17', 'age_18_to_29', 'age_30_to_39',
       'age_40_to_49', 'age_50_to_54', 'age_55_to_59', 'age_60_to_64',
       'age_65_to_69', 'age_70_to_74', 'age_75_to_79', 'age_80_to_90_PLUS',
       'travel_cluster', 'care', 'meat_and_fish_processing', 'ready_meals',
       'textiles', 'warehousing', 'LSOA11NMW', 'geometry']

# dictionary for testing the normalise_data function 
# key1 and key2 should be ignored as they are not dictionaries
# and key4 should be ignored as the 'flag' key value is not 'test'
normalise_dic = {
     'key1' : ['col1', 'col2']
    ,'key2': 'test'
    ,'key3': {
         'flag': 'test'
        ,'by': ''
        ,'suffix': ''
        ,'columns':['col1', 'col2']
    }
    # this should be ignored
    ,'key4': {
         'flag': 'test2'
        ,'by': ''
        ,'suffix': ''
        ,'columns':['col2', 'col3']
    }
    # col3 and 4 should remain untouched, with new _test columns created
    ,'key5': {
         'flag': 'test'
        ,'by': 'std'
        ,'suffix': '_test'
        ,'columns':['col3', 'col4']
    }
}

# dictionary for testing the sum_features column

sum_dic = {'newcol':['col1', 'col2']
        ,'newcol2':['col3', 'col4', 'col5']
       }

tranche_model_idbr_features = ['care_homes_warehousing', 'ready_meals_textiles', 'meat_and_fish_processing']

tranche_dates = ['2020-04-26','2020-08-31','2020-11-14','2020-12-31','2021-02-14','2021-04-29','2021-07-15','2021-08-31']

tranche_description = ['low_prev_no_school',
                       'high_prev_school_opn',
                       'high_prev_school_opn_alph',
                       'high_prev_no_school_alph_vaccn',
                       'low_prev_school_opn_vaccn_dbl',
                       'high_prev_school_opn_dlta_vaccn_dbl',
                       'lifting_lockdown',
                       'high_prev_school_open_delta_vaccn']