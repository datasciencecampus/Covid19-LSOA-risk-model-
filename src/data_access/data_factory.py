import itertools
import numpy as np

import pandas as pd
from google.cloud import storage
from google.cloud import bigquery

import google.cloud.bigquery.magics
google.cloud.bigquery.magics.context.use_bqstorage_api = True

import geopandas as gpd

from utils import data as dt
from utils import config as conf
from data_access.idata import IData
from data_access.prep_static import StaticVars, StaticVars_rgn
import data_access.data_objects as do


"""
Data Factory for convenient data access
"""

class NoProcessing(IData):
    "A Concrete Class that implements the IProduct interface"
    def __init__(self, query):
        self.name = "Query no processing"
        self.__query = query
    def create_dataframe(self):
        query_job = super().client.query(self.__query)
        return query_job.to_dataframe()

class DataFactory:
    "The Factory Class"
    @staticmethod
    def get(data_name) -> IData:
        "A static method to get a concrete product"
        try:
            if data_name == 'static_vars':
                return StaticVars()
            if data_name =='static_vars_rgns':
                return StaticVars_rgn()
            if data_name == 'LSOA_2011':
                return do.LSOA2011() #only utilised within data_objects.py
            if data_name == 'mid_year_lsoa':
                return do.LSOA_MidYear() # only utilised within data_objects.py
            if data_name == 'aggregated_tests_lsoa':
                return do.AggregatedTestsLSOA() #used in create_dynamic
            if data_name == 'mobility_clusters_processed':
                return do.MobilityClustersProcessed()#used in apply_time_lag, create_static
            if data_name == 'lsoa_daily_footfall':
                return do.LSOADailyFootfall()  #used in create_dynamic
            if data_name == 'lsoa_vaccinations':
                return do.LSOAVaccinations() #used in create_dynamic,
            # if data_name == 'flows_mars_data':
            #     return do.FlowsMarsData() # not currently used
            # if data_name == 'static_subset_for_norm': #used in create_dynamic for normalising variables
            #     return do.StaticSubset()
            # if data_name == 'static_normalised':    
            #     return do.StaticNormalised()  #created in create_static, used in apply_vif_static
            if data_name == 'all_tranches_dynamic_static':
                return do.AllTranches() #uses outputs from apply_vif_Static and apply_time_lag, used in modelling static and dynamic and used within do.DynamicChangesWeekly.
            if data_name == 'dynamic_changes_weekly':
                return do.DynamicChangesWeekly()#created in risk_weekly_dynamic
            # if data_name == 'static_changes_weekly':
            #     return do.StaticChangesWeekly() #created in risk_weekly_static
            # if data_name == 'static_changes_weekly_ci':
            #     return do.StaticChangesWeekly_ci()#created in risk_weekly static
            if data_name=='Deimos_aggregated': 
                return do.DeimosAggregated() #used as input into LSOADailyFootfall().
            if data_name=='Deimos_trip_end_count':
                return do.DeimosEndTrip() #used in create_dynamic
            # if data_name == 'lsoa_industry':
            #     query="SELECT * FROM `ons-hotspot-prod.wip.idbr_lsoa_industry_wide`" #used in create static
            #     return NoProcessing(query)
            # if data_name == 'travel_clusters':
            #     query="SELECT * FROM `ons-hotspot-prod.ingest_geography.lsoa_mobility_cluster_ew_lu`" #used in apply_time_lag
            #     return NoProcessing(query)
            if data_name == 'flow_to_work':
                table = conf.data_location_big_query['flow_to_work']
                query = f"SELECT * FROM `{table}`"
                return NoProcessing(query)
            # if data_name == 'lsoa_dynamic':
            #     table = conf.dynamic_data_file
            #     query = f"SELECT * FROM `{table}`"  #created in create_dynamic, used in apply_time_lag
            #     return NoProcessing(query)
            # if data_name == 'static_vars_for_modelling':
            #     query="SELECT * FROM `wip.risk_model_static_variables_main`" #created in apply_vif_static, not currently used in any notebooks
            #     return NoProcessing(query)
            if data_name == 'dynamic_time_lagged':
                table = conf.lagged_dynamic_non_stationary
                query = f"SELECT * FROM `{table}`" #created in apply_time_lag, used in data object AllTranches()
                return NoProcessing(query)
#             if data_name == 'dynamic_raw_norm_chosen_geo':
#                 table = conf.dynamic_data_file_normalised
#                 query = f"SELECT * FROM `{table}`" #created in create_dynamic, used in apply_timelag
#                 return NoProcessing(query)     
            
            # processed time tranches model inputs
            # these data sets are read in by the time tranches modelling functions
            if data_name == 'tranche_model_input':
                table = conf.tranches_model_input_processed
                query = f"SELECT * FROM {table}"
                return NoProcessing(query)
            
            if data_name == 'tranche_model_test_data':
                table = conf.tranches_model_test_data
                query = f"SELECT * FROM {table}"
                return NoProcessing(query)
            
            # processed static data used in both models
            if data_name == 'static_features':
                table = conf.static_data_file
                query = f"SELECT * FROM {table}"
                return NoProcessing(query)
            
            # outputs of time tranches model
            # these data sets are read into the 03_Outputs.py
            if data_name == 'tranche_regularised_coefs':
                table = conf.tranche_coefs_regularisation
                query=f"SELECT * FROM `{table}`"
                return NoProcessing(query)
            if data_name == 'tranche_non_reg_std_coefs':
                table = conf.tranche_coefs_standardised
                query = f"SELECT * FROM `{table}`"
                return NoProcessing(query)
            if data_name == 'tranche_non_reg_non_std_coefs':
                table = conf.tranche_coefs_non_standardised
                query = f"SELECT * FROM `{table}`"
                return NoProcessing(query)  
            if data_name == 'tranche_preds_all_tranches':
                table = conf.tranche_preds_all_tranches
                query = f"SELECT * FROM `{table}`"
                return NoProcessing(query)  
            if data_name == 'tranche_preds_latest':
                table = conf.tranche_preds_latest
                query = f"SELECT * FROM `{table}`"
                return NoProcessing(query)      
            if data_name == 'tranche_model_features':
                table = conf.tranche_model_features_gbq_loc
                query = f"SELECT * FROM `{table}`"
                return NoProcessing(query)  
            
            # data used for unit tests
            if data_name.startswith('unit_test'):
                query = f"SELECT * FROM {'wip.' + data_name}"
                return NoProcessing(query)
            raise Exception('Data Class Not Found')
        except Exception as _e:
            print(_e)
        return None