from data_access.idata import IData
import pandas as pd
from utils import config as cf


class StaticVars(IData):
    "A Concrete Class that implements the IProduct interface"
    def __init__(self):
        self.name = "Static Vars"
    def create_dataframe(self):
        
        static_data_location = cf.data_location_big_query['static']
        
        query_job = super().client.query(
            f"""
            select * from `{static_data_location}`
            """
        )

        demography_df = query_job.to_dataframe()
        
        
        df2=pd.DataFrame(demography_df.loc[:,'LSOA11CD':'RGN19NM'])  #make a new df to bring in aggregated variables into
        df3 = demography_df[['HOUSEHOLD_SIZE_1_PERSON_IN_HOUSEHOLD','HOUSEHOLD_SIZE_2_PEOPLE_IN_HOUSEHOLD', 'HOUSEHOLD_SIZE_3_PEOPLE_IN_HOUSEHOLD',
                                     'BAME_PROP', 'SHARED_DWELLINGS_NUMBER_OF_PEOPLE', 'COMMUNAL_ESTABLISHMENT_MEDICAL_AND_CARE_TOTAL',
                                     'COMMUNAL_ESTABLISHMENT_OTHER_EDUCATION', 'COMMUNAL_ESTABLISHMENT_OTHER_HOSTEL_OR_TEMPORARY_SHELTER_FOR_THE_HOMELESS',
                                     'FAMILIES_WITH_DEPENDENT_CHILDREN_NO_DEPENDENT_CHILDREN', 'FAMILIES_WITH_DEPENDENT_CHILDREN_ALL_FAMILIES',
                                     'METHOD_OF_TRAVEL_TO_WORK_WORK_MAINLY_FROM_HOME', 'IMD_SCORE', 'IMD_INCOME_SCORE', 'IMD_EMPLOYMENT_SCORE',
                                     'STUDENT_LIVING_WITH_PARENTS', 'STUDENT_LIVING_IN_A_COMMUNAL_ESTABLISHMENT_TOTAL', 
                                     ]]


        df2['HOUSEHOLD_SIZE_3_PLUS_PEOPLE_IN_HOUSEHOLD']=demography_df.loc[:, 'HOUSEHOLD_SIZE_4_PEOPLE_IN_HOUSEHOLD': 
                                                                     'HOUSEHOLD_SIZE_8_PEOPLE_IN_HOUSEHOLD'].sum(axis=1)




        df2['COMMUNAL_ESTABLISHMENT_MEDICAL_AND_CARE_CARE_HOMES']= demography_df.loc[:, 'COMMUNAL_ESTABLISHMENT_MEDICAL_AND_CARE_OTHER_CARE_HOME_WITH_NURSING': 
                                                                     'COMMUNAL_ESTABLISHMENT_MEDICAL_AND_CARE_OTHER_CARE_HOME_WITHOUT_NURSING'].sum(axis=1)

        df2['COMMUNAL_ESTABLISHMENT_OTHER_PRISON_AND_OTHER_DETENTION']= demography_df.loc[:, 'COMMUNAL_ESTABLISHMENT_OTHER_PRISON': 
                                                                     'COMMUNAL_ESTABLISHMENT_OTHER_DETENTION_CENTRES_AND_OTHER_DETENTION'].sum(axis=1)



        df2['METHOD_OF_TRAVEL_TO_WORK_Public_TRANSPORT']= demography_df.loc[:, 'METHOD_OF_TRAVEL_TO_WORK_UNDERGROUND_METRO_LIGHTRAIL_TRAM': 
                                                                     'METHOD_OF_TRAVEL_TO_WORK_TAXI'].sum(axis=1) 

        df2['METHOD_OF_TRAVEL_TO_WORK_NON_MOTORISED']= demography_df['METHOD_OF_TRAVEL_TO_WORK_BICYCLE']+ demography_df['METHOD_OF_TRAVEL_TO_WORK_ON_FOOT']

        df2['METHOD_OF_TRAVEL_TO_WORK_PRIVATE_TRANSPORT']= demography_df.loc[:, 'METHOD_OF_TRAVEL_TO_WORK_MOTORCYCLE_SCOOTER_OR_MOPED': 
                                                                     'METHOD_OF_TRAVEL_TO_WORK_PASSANGERS_IN_A_CAR_OR_VAN'].sum(axis=1) \
                                                                                    + demography_df.loc[:,'METHOD_OF_TRAVEL_TO_WORK_OTHER']


        ethnicity=demography_df.loc[:, 'CENSUS_2011_WHITE': 'CENSUS_2011_OTHER_ETHNIC_GROUP_ANY_OTHER_ETHNIC_GROUP']


        df2['NO_UNPAID_CARE'] = \
        demography_df['PROVISION_OF_UNPAID_CARE_PROVIDES_NO_UNPAID_CARE'] 

        df2['UNPAID_CARE_1_HOUR_PLUS'] = \
        demography_df['PROVISION_OF_UNPAID_CARE_PROVIDES_1_TO_19_HOURS_UNPAID_CARE_A_WEEK'] \
        + demography_df['PROVISION_OF_UNPAID_CARE_PROVIDES_20_TO_49_HOURS_UNPAID_CARE_A_WEEK']  \
        + demography_df['PROVISION_OF_UNPAID_CARE_PROVIDES_50_OR_MORE_UNPAID_CARE_A_WEEK'] 

        df2['HEALTH_AGE_UNDER_50_GOOD_FAIR_HEALTH'] = demography_df['HEALTH_AGE_UNDER_50_VERY_GOOD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']\
        +demography_df['HEALTH_AGE_UNDER_50_FAIR_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']  #Healthy        


        df2['HEALTH_AGE_UNDER_50_BAD_HEALTH']=demography_df['HEALTH_AGE_UNDER_50_BAD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']  #unhealthy
        
        df2['HEALTH_AGE_50_to_64_BAD_HEALTH']=demography_df['HEALTH_AGE_50_TO_54_BAD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']\
        +demography_df['HEALTH_AGE_55_TO_59_BAD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY'] \
        +demography_df['HEALTH_AGE_60_TO_64_BAD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY'] #unhealthy

        df2['HEALTH_AGE_50_to_64_GOOD_FAIR_HEALTH'] = demography_df['HEALTH_AGE_50_TO_54_VERY_GOOD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']\
        +demography_df['HEALTH_AGE_50_TO_54_FAIR_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY'] \
        +demography_df['HEALTH_AGE_55_TO_59_VERY_GOOD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY'] \
        +demography_df['HEALTH_AGE_55_TO_59_FAIR_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']\
        +demography_df['HEALTH_AGE_60_TO_64_VERY_GOOD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']\
        +demography_df['HEALTH_AGE_60_TO_64_FAIR_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']   #healthy

        df2['HEALTH_AGE_65_to_74_GOOD_FAIR_HEALTH'] = demography_df['HEALTH_AGE_65_TO_69_VERY_GOOD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']\
        +demography_df['HEALTH_AGE_65_TO_69_FAIR_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY'] \
        +demography_df['HEALTH_AGE_70_TO_74_VERY_GOOD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY'] \
        +demography_df['HEALTH_AGE_70_TO_74_FAIR_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY'] #healthy

        df2['HEALTH_AGE_65_to_74_BAD_HEALTH']=demography_df['HEALTH_AGE_65_TO_69_BAD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']\
        +demography_df['HEALTH_AGE_70_TO_74_BAD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']  #unhealthy


        df2['HEALTH_AGE_75_PLUS_GOOD_FAIR_HEALTH']=demography_df['HEALTH_AGE_75_TO_79_VERY_GOOD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']\
        +demography_df['HEALTH_AGE_75_TO_79_FAIR_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']\
        +demography_df['HEALTH_AGE_80_OR_OVER_VERY_GOOD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']\
        +demography_df['HEALTH_AGE_80_OR_OVER_FAIR_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY'] #healthy


        df2['HEALTH_AGE_75_PLUS_BAD_HEALTH']=demography_df['HEALTH_AGE_75_TO_79_BAD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']\
        +demography_df['HEALTH_AGE_80_OR_OVER_BAD_HEALTH_TOTAL_LONG_TERM_HEALTH_PROBLEM_OR_DISABILITY']  #unhealthy

        #Full_time_occupations=demography_df.loc[:, 'OCCUPATION_MANAGERS_DIRECTORS_AND_SENIOR_OFFICIALS': 'OCCUPATION_TRANSPORT_AND_MOBILE_MACHINE_DRIVERS_AND_OPERATIVES']
        df2= pd.concat([df2, ethnicity, df3], axis=1)# Full_time_occupations

        
        return df2


class StaticVars_rgn(IData):
    "A Concrete Class that implements the IProduct interface"
    def __init__(self):
        self.name = "StaticVars_rgn"
    
    def create_dataframe(self):
        
        query_job = super().client.query(
            """
            select * from `ons-hotspot-prod.ingest_risk_model.risk_model_lsoa`
            """
        )

        demography_df = query_job.to_dataframe()
        
        
        df=pd.DataFrame(demography_df.loc[:,'LSOA11CD':'RGN19NM']).drop_duplicates().reset_index(drop=True)
        
        
        return df
