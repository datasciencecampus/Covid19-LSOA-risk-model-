import os
import sys
import pandas as pd
from google.cloud import storage
import geopandas as gpd
from tqdm import tqdm

# Import from local data files
current_path = os.path.abspath('.')
sys.path.append(os.path.dirname(current_path))

from data_access.idata import IData
import utils.config as cf
import utils.dynamic as dyn
import utils.model as md
import utils.data as dt

from functools import reduce


class LSOA2011(IData):
    """A Class that reads an LSOA GeoJSON file and returns the information
    as a Pandas DataFrame"""
    def __init__(self):
        self.name = "data description"
        
    def create_dataframe(self):
                
        client = storage.Client()
        
        # set the bucket
        bucket = client.get_bucket(cf.geography_bucket)
        
        # Get the LSOA boundaries and download to store locally (on notebook)
        blob = bucket.get_blob(cf.lsoa_geojson_filename)
        blob.download_to_filename('LSOA_2011')
        LSOA_2011_map = gpd.read_file('LSOA_2011')
        return LSOA_2011_map



class AggregatedTestsLSOA(IData):
    "A Class for creating a dataframe containing the number of weekly positive tests in an LSOA from a specified date"
    def __init__(self):
        self.name = "data description"
    def create_dataframe(self):
        
        query = f""" 
            SELECT Specimen_Date, Lower_Super_Output_Area_Code, SUM(positive_test)
            FROM `{cf.data_location_big_query['cases']}`
            WHERE Specimen_Date >= {cf.data_start_date}
            GROUP BY Lower_Super_Output_Area_Code, Specimen_Date
        """

        
        query_job = super().client.query(
            query
        )

        cases_df = query_job.to_dataframe()
        
        # convert date format
        cases_df['Specimen_Date'] = pd.to_datetime(cases_df['Specimen_Date'])
        
        # rename columns
        cases_df.rename(columns={'f0_':'COVID_Cases'},inplace=True)       
        cases_df.rename(columns={'Specimen_Date':'Date'},inplace=True)
        cases_df.rename(columns={'Lower_Super_Output_Area_Code':'LSOA11CD'},inplace=True)
        
        # calculate cumulative sum of positive tests
        cases_df_cumsum = cases_df.copy()
        cases_df_cumsum = cases_df_cumsum.sort_values(by=['LSOA11CD','Date'])
        cases_df_cumsum = cases_df_cumsum.groupby(["LSOA11CD",'Date']).sum().groupby(level=0).cumsum().reset_index()
        cases_df_cumsum = cases_df_cumsum.rename(columns={'COVID_Cases':'cases_cumsum'})[['LSOA11CD','Date','cases_cumsum']]
        
        # merge with the daily counts
        cases_df = cases_df.merge(cases_df_cumsum, how='left', on=['LSOA11CD', 'Date'])
        
        # snap dates to the Sunday of the week and aggregate
        cases_df['Date'] = cases_df['Date'].apply(lambda x: dyn.end_of_week(x))
        cases_df = cases_df.groupby(['Date','LSOA11CD']).agg(({'COVID_Cases':'sum', 'cases_cumsum':'max'})).reset_index()
        cases_df['Date'] = pd.to_datetime(cases_df['Date'])

        return cases_df


class LSOA_MidYear(IData):
    "A Concrete Class that implements the IProduct interface"
    def __init__(self):
        self.name = "LSOA_MidYear"
        
    def __bin_ages(self,dfin, dfout, bins):
        for bin in bins:
            dfout['age_'+bin[0] + '_to_' + bin[1]] = dfin.loc[:, bin[0] : bin[1]].sum(axis=1)
        return dfout
    
    def create_dataframe(self):
        
        lsoa_midyear_2019_data_location = cf.data_location_big_query['lsoa_midyear_population_2019']
        
        query_job = super().client.query(
            f"""
            SELECT *
            FROM `{lsoa_midyear_2019_data_location}` 
            """
        )

        age_df = query_job.to_dataframe()
        
        
        age_df_subset = age_df.loc[:,'LSOA11CD':'MF_AGE_90_PLUS']
        age_df_subset.columns = age_df_subset.columns.str.replace('MF_AGE_',"")

        #bin ages
        binned = age_df_subset.loc[:,'LSOA11CD':'ALL_PEOPLE']
        
        
        age_bins = [['0','12'],['13','17'],['18','29'],['30','39'],['40','49'],
                   ['50','54'],['55','59'],['60','64'],['65','69'],['70','74'],
                   ['75','79'],['80','90_PLUS']]

      
        binned = self.__bin_ages(age_df_subset, binned, age_bins)

        binned.drop(columns=['LSOA11NM','MSOA11CD','MSOA11NM','LTLA20CD','LTLA20NM','UTLA20CD','UTLA20NM','RGN19CD','RGN19NM'],inplace=True)

        return binned


class MobilityClustersProcessed(IData):
    def __init__(self):
        self.name = "data"
    def create_dataframe(self):
        
        mobility_clusters_processed_location = cf.data_location_big_query['mobility_clusters_processed']
        
        query = f"SELECT * FROM `{mobility_clusters_processed_location}`"
        
        query_job = super().client.query(
            query
        )

        travel_clusters = query_job.to_dataframe()
        travel_clusters.drop(columns=['lsoa11nm', 'index_right'], inplace=True)
        travel_clusters.rename(columns={'ct_2':'travel_cluster', 'lsoa11cd':'LSOA11CD'}, inplace=True)

        travel_clusters = dt.combining_and_remap_travel_cluster(travel_clusters)
        
        return travel_clusters

    
class LSOADailyFootfall(IData):
    def __init__(self):
        self.name = "data"
    def create_dataframe(self):
        
        df = DeimosAggregated().create_dataframe()

        df = df.rename(columns={'date_dt':'Date', 
                                'lsoa11cd':'LSOA11CD',
                                'lsoa_people_perhactares':'lsoa_people_perHactares'})
    
        df['Date'] = pd.to_datetime(df['Date'])
        
        # transform each date to the date of the Sunday for each given week
        df['Date'] = df['Date'].apply(lambda x: dyn.end_of_week(x))
        
        # define new columns
        df['worker_footfall_sqkm'] = 0
        df['visitor_footfall_sqkm'] = 0
        df['resident_footfall_sqkm'] = 0
        
        # change units from per hectare to per square kilometre
        df.loc[df['purpose'] == 'Worker', 'worker_footfall_sqkm'] = df[df['purpose'] == 'Worker'].lsoa_people_perHactares.div(0.01)
        df.loc[df['purpose'] == 'Visitor', 'visitor_footfall_sqkm'] = df[df['purpose'] == 'Visitor'].lsoa_people_perHactares.div(0.01)
        df.loc[df['purpose'] == 'Resident', 'resident_footfall_sqkm'] = df[df['purpose'] == 'Resident'].lsoa_people_perHactares.div(0.01)
        
        # group by the transformed date to return total weekly footfall per square kilometre for each purpose
        df = df.groupby(["Date","LSOA11CD"]).sum().reset_index().drop(columns=["lsoa_people", "lsoa_people_perHactares"])
        
        # create new features with combinations of the purposes
        df['total_footfall_sqkm'] = df["worker_footfall_sqkm"] + df["visitor_footfall_sqkm"] + df["resident_footfall_sqkm"]
        df['worker_visitor_footfall_sqkm'] = df["worker_footfall_sqkm"] + df["visitor_footfall_sqkm"]
        df['Date'] = pd.to_datetime(df['Date'])
        
        return df

class LSOAVaccinations(IData):
    def __init__(self):
        self.name = "data"
    def create_dataframe(self):
        
        # define query
        query = f"""SELECT LSOA_OF_RESIDENCE,
                          vcc_date,
                          GENDER,
                          dose_first, 
                          dose_second, 
                          booster 
                    
                    FROM `{cf.data_location_big_query['vaccination']}` 
                    WHERE vcc_date >= {cf.data_start_date}"""
        
        query_job = super().client.query(
            query
        )

        vaccination_df = query_job.to_dataframe()
        
        # import 2001 to 2011 LSOA codes lookup
        lsoa_query = f"SELECT LSOA01CD, LSOA11CD FROM `{cf.data_location_big_query['lsoa_2001_to_2011_lookup']}`"
        
        lu_query_job = super().client.query(lsoa_query)
        
        lsoa_lu = lu_query_job.to_dataframe()

        # join on old LSOA codes to see which LSOA are using the 2001 LSOA code
        vaccination_df = vaccination_df.merge(lsoa_lu, left_on='LSOA_OF_RESIDENCE', right_on='LSOA01CD', how='outer') 

        # fill missing values with 2011 codes
        vaccination_df['LSOA11CD'] = vaccination_df['LSOA11CD'].fillna(vaccination_df['LSOA_OF_RESIDENCE'])  
    
        # drop rows where LSOA code is missing
        vaccination_df = vaccination_df[~vaccination_df['LSOA_OF_RESIDENCE'].isna()]

        # filter for LSOAs in England
        vaccination_df = vaccination_df[vaccination_df['LSOA11CD'].str.startswith('E')]

        # check that all LSOAs are present in the data
        assert vaccination_df['LSOA11CD'].nunique() == cf.n_lsoa, "Invalid LSOAs,the number of unique LSOAs does not equal the value for n_lsoa in the config file"

        # drop features that are not needed
        vaccination_df = vaccination_df.drop(columns=['LSOA_OF_RESIDENCE','LSOA01CD'])
        
        # drop records where GENDER == 'Unknown' and drop the GENDER column
        vaccination_df = vaccination_df[vaccination_df['GENDER'] != 'Unknown']
        vaccination_df.drop('GENDER', axis=1, inplace=True)

        # aggregate vaccinations records at LSOA level
        vaccination_df.rename(columns={'vcc_date':'Date'},inplace=True)

        cols_to_aggregate = ['dose_first',
                             'dose_second',
                             'booster'] 
        
        # group by LSOA 2011 code to remove any reference to 2001 LSOA codes
        vaccination_df = vaccination_df.groupby(['Date','LSOA11CD'])[cols_to_aggregate].sum().reset_index()
        
        # proportion of vaccinationed populaition (cumulative sum)
        vaccination_df_cumsum = vaccination_df.copy()
        vaccination_df_cumsum = vaccination_df_cumsum.sort_values(by=['LSOA11CD', 'Date'])
        vaccination_df_cumsum = vaccination_df_cumsum.groupby(['LSOA11CD', 'Date']).sum().groupby(level=0).cumsum().reset_index()
        vaccination_df_cumsum = vaccination_df_cumsum.rename(columns={'dose_second':'dbl_vacc_cumsum',
                                                                      'booster':'trpl_vacc_cumsum'})[['LSOA11CD', 'Date', 'dbl_vacc_cumsum', 'trpl_vacc_cumsum']]
        # merge vaccinated counts and cumulative sums
        vaccination_df = vaccination_df.merge(vaccination_df_cumsum, on=['LSOA11CD', 'Date'], how='left')
        
        # make the column names more informative
        vaccination_df.rename(columns={'dose_first':'total_vaccinated_first_dose', 
                                       'dose_second':'total_vaccinated_second_dose',
                                       'booster':'total_vaccinated_booster'}, inplace=True)
        
        # define columns to keep
        few_cols_vacct = ['Date', 
                         'LSOA11CD', 
                         'total_vaccinated_first_dose',
                         'total_vaccinated_second_dose',
                         'total_vaccinated_booster',
                         'dbl_vacc_cumsum',
                         'trpl_vacc_cumsum']
        
        # subset for the columns listed
        vaccination_df = vaccination_df[few_cols_vacct]
        
        # convert to datetime
        vaccination_df['Date'] = pd.to_datetime(vaccination_df['Date'])

        # transform the date to the Sunday of the same week
        vaccination_df['Date'] = vaccination_df['Date'].apply(lambda x: dyn.end_of_week(x))

        # aggregate to weekly 
        vaccination_df = vaccination_df.groupby(['Date', 'LSOA11CD']).agg(({'total_vaccinated_first_dose':'sum',
                                                                                 'total_vaccinated_second_dose':'sum',
                                                                                 'total_vaccinated_booster':'sum',
                                                                                 'dbl_vacc_cumsum':'max',
                                                                                 'trpl_vacc_cumsum':'max'})).reset_index()
        
        vaccination_df['Date'] = pd.to_datetime(vaccination_df['Date'])

        return vaccination_df


class AllTranches(IData):
    def __init__(self):
        self.name = "data"
    def create_dataframe(self):
        
        table = cf.static_data_file
    
        query_static = f"SELECT * FROM `{table}`"
        query_job_static = super().client.query(
            query_static
        ) 
        static_variables_df = query_job_static.to_dataframe()
        
        table = cf.lagged_dynamic_non_stationary
        
        query_dynamic = f"SELECT * FROM `{table}`"
        query_job_dynamic = super().client.query(
            query_dynamic
        )  
        dynamic_lagged_variables_df = query_job_dynamic.to_dataframe()
        
        #common columns between the static and dynamic datasets
        commn_colmns = list(set(dynamic_lagged_variables_df.columns) & set(static_variables_df.columns))


        # Get a joint dataframe
        df_all_tranches_dynamic_static=dynamic_lagged_variables_df.merge(static_variables_df, on=commn_colmns, how='inner')


        # Choose a specific date to start the risk model training from
        cut_off_datum = cf.model_start_date

        df_all_tranches_dynamic_static['Date'] = pd.to_datetime(df_all_tranches_dynamic_static['Date'])
        df_all_tranches_dynamic_static['Date'] = df_all_tranches_dynamic_static['Date'].dt.date

        df_all_tranches_dynamic_static = df_all_tranches_dynamic_static[df_all_tranches_dynamic_static['Date']>=\
                                                                      pd.to_datetime(cut_off_datum).date()].\
        sort_values(by='Date').reset_index(drop=True)

        # Convert dates to dummy weeks
        date_list = sorted(df_all_tranches_dynamic_static['Date'].unique())
        date_list = [str(x) for x in date_list]

        week_list = ['week_' + str(x + 1) for x in range(len(date_list))]
        date_dict = dict(zip(date_list, week_list))
        
        # Convert dates to Months
        date_list = sorted(df_all_tranches_dynamic_static['Date'].unique())

        month_list = [pd.to_datetime(x).month_name() for x in date_list]
        year_list = [pd.to_datetime(x).year for x in date_list]

        month_list = [str(i) +'_'+ str(j) for i, j in zip(month_list, year_list)]
        
        date_list = [str(x) for x in date_list]
        month_dict = dict(zip(date_list, month_list))

        df_all_tranches_dynamic_static['Date'] = df_all_tranches_dynamic_static['Date'].astype(str)
        df_all_tranches_dynamic_static['week'] = df_all_tranches_dynamic_static['Date'].map(date_dict)

        #create a Month variable
        df_all_tranches_dynamic_static['Month'] = df_all_tranches_dynamic_static['Date'].map(month_dict)

        no_cases_week = df_all_tranches_dynamic_static.groupby('week')['COVID_Cases_per_unit_area'].sum().\
        index[df_all_tranches_dynamic_static.groupby('week')['COVID_Cases_per_unit_area'].sum().values==0]

        df_all_tranches_sbset = df_all_tranches_dynamic_static[~df_all_tranches_dynamic_static['week'].isin(no_cases_week)]

        return df_all_tranches_sbset


class DynamicChangesWeekly(IData):
    def __init__(self):
        self.name = "data"
    def create_dataframe(self):
        table = cf.project_name + '.' + cf.risk_pred + cf.model_suffixes['static_main']
        
        query = f"SELECT * FROM `{table}`"
        query_job = super().client.query(
            query
        )
        
        df_pred_tc_all_week_static_zir = query_job.to_dataframe()

        df_all_tranches_sbset = AllTranches().create_dataframe()
        
        # Residuals from static predictors
        # These residuals will be used as a target variable
        # based on change of dynamic predictors
        df_pred_tc_all_week_static_zir['Residual'] = df_pred_tc_all_week_static_zir['Actual_cases']-\
        df_pred_tc_all_week_static_zir['Predicted_cases_train']

        df_pred_tc_all_week_static_zir['week_train']=df_pred_tc_all_week_static_zir['week_train'].str.strip('week_').astype(int)

        df_pred_tc_all_week_static_zir=df_pred_tc_all_week_static_zir[['LSOA11CD','week_train','travel_cluster','Residual']]


        ##### WE ARE MODELLING
        ##### CHANGE IN RESIDUALS
        ##### AS TARGET VARIABLES

        ################################################## 
        ##### split the data for each LSOA
        df_pred_tc_all_week_static_zir_lsoa = [pd.DataFrame(y) for x, y in 
                                               df_pred_tc_all_week_static_zir.groupby('LSOA11CD', as_index=False)]

        ##### CHANGE IN RESIDUAL DATA FOR ALL THE LSOA FROM THE PREVIOUS WEEK
        chnge_pred_tc_all_week_static_zir_lsoa=[x.sort_values(by='week_train')[['LSOA11CD','week_train','travel_cluster','Residual']].\
                                                set_index(['week_train','LSOA11CD','travel_cluster']).diff().dropna() for \
                                                x in df_pred_tc_all_week_static_zir_lsoa]

        ##### CONCAT CHANGE IN RESIDUAL FOR ALL THE LSOA
        df_pred_tc_all_week_static_zir=pd.concat(chnge_pred_tc_all_week_static_zir_lsoa,axis=0).reset_index()

        df_pred_tc_all_week_static_zir['week_train']=str('week_')+df_pred_tc_all_week_static_zir['week_train'].astype(str)
       
        print('Vaccination data preparation for dynamic training....Started')
        #Cumulative vaccine from previous weeks
        dynamic_colmns_vaccine=['LSOA11CD','Date'] + cf.dynamic_vacc

        # CONCAT cumulative vaccine DATA (FROM THE PREVIOUS WEEK) FOR ALL THE LSOA
        cum_vaccine_df=md.cumulative_data(df_all_tranches_sbset, dynamic_colmns_vaccine)
        print('Vaccination data preparation for dynamic training....Finished')
        
        
        print('Mobility data preparation for dynamic training....Started')
        #Change in mobility indicators from previous weeks
        dynamic_colmns_mobility=['LSOA11CD','Date'] + cf.dynamic_mobility

        # CONCAT CHANGE in mobility DATA (FROM THE PREVIOUS WEEK) FOR ALL THE LSOA
        change_mobility_df=md.difference_data(df_all_tranches_sbset, dynamic_colmns_mobility)
        print('Mobility data preparation for dynamic training....Finished')
        
        # MERGE ALL THE DYNAMIC PREDICTORS 

        date_list=sorted(df_all_tranches_sbset['Date'].unique())
        date_list=[str(x) for x in date_list]
        week_list=['week_'+str(x+1) for x in range(len(date_list))]
        date_dict=dict(zip(date_list,week_list))
        
        df_dynamic_changes_weekly = reduce(lambda left,right: pd.merge(left,right,on=['Date','LSOA11CD']), [cum_vaccine_df,change_mobility_df])
        df_dynamic_changes_weekly['week_train']=df_dynamic_changes_weekly['Date'].map(date_dict)
        df_dynamic_changes_weekly_with_trgt=df_pred_tc_all_week_static_zir.merge(df_dynamic_changes_weekly,on=['week_train','LSOA11CD'],how='inner')
        
        return df_dynamic_changes_weekly_with_trgt

    
class MSOA2011(IData):
    """A Class that reads an MSOA GeoJSON file and returns the information
    as a Pandas DataFrame"""
    def __init__(self):
        self.name = "data description"
    def create_dataframe(self):
        
        client = storage.Client()
        
        # set the bucket
        bucket = client.get_bucket(cf.geography_bucket)
        
        # Get the MSOA boundaries and download to store locally (on notebook)
        blob = bucket.get_blob(cf.msoa_geojson_filename)
        blob.download_to_filename('msoa')
        MSOA_2011_map = gpd.read_file('msoa')
        
        
        MSOA_2011_map = MSOA_2011_map.set_crs(27700, allow_override=True) 

        MSOA_2011_map['AREALHECT'] = MSOA_2011_map.geometry.area
        
        return MSOA_2011_map


class DeimosAggregated(IData):
    
    """A class for dis-aggregation of Deimos data from MSOA level to LSOA. Counts of footfall are 
    distributed amongst the LSOAs based on the number of workers within each LSOA"""
    
    def __init__(self):
        self.name = 'Deimos data aggregated from deimos_ag'
        
    def create_dataframe(self):
        
        # read in deimos date seprated by age and gender
        query_people_counts = f"""SELECT date_dt, purpose, msoa , SUM(people) as msoa_people
                                FROM `{cf.data_location_big_query['deimos_aggregated']}`
                                WHERE date_dt>={cf.data_start_date} AND ((msoa LIKE 'E%') OR (msoa LIKE 'W%')) 
                                GROUP BY date_dt, purpose, msoa""" 
        
        query_job_deimos = super().client.query(query_people_counts) 
        people_counts_df_msoa_daily = query_job_deimos.to_dataframe()

        #get msoa area data
        msoa = MSOA2011().create_dataframe() 
        
        people_counts_df_msoa_daily = pd.merge(people_counts_df_msoa_daily,\
                                       msoa, left_on='msoa',\
                                       right_on='MSOA11CD', how='left')
        
        people_counts_df_msoa_daily['msoa_people_perHactares'] = people_counts_df_msoa_daily['msoa_people']/people_counts_df_msoa_daily['AREALHECT']
       
        # final msoa counts
        people_counts_df_msoa_daily = people_counts_df_msoa_daily[['date_dt', 'msoa', 'purpose', \
                                                           'msoa_people', 'AREALHECT', 'msoa_people_perHactares']] 
        
        
        # get worker population to create weights for de-aggregation
        workers_query = "SELECT LSOA11 as LSOA11CD, SUM(N_EMPLOYED) as n_workers FROM `ons-hotspot-prod.ingest_risk_model.idbr_2019_lsoa11` GROUP BY LSOA11CD"  
        query_job_workers = super().client.query(workers_query) 
        worker19_lsoa = query_job_workers.to_dataframe()
        
        # get LSOA to MSOA lookup
        lookup_query = "SELECT lsoa11cd as LSOA11CD,msoa11cd as MSOA11CD FROM `ons-hotspot-prod.ingest_geography.PCD_OA_LSOA_MSOA_LAD_FEB21_UK_LU`" 
        query_job_lookup = super().client.query(lookup_query) 
        lsoa_msoa_lookup = query_job_lookup.to_dataframe()
        
        # remove duplicates since LSOAs appear multiple times per MSOA in this data set
        lsoa_msoa_lookup = lsoa_msoa_lookup.drop_duplicates() 

        worker19_lsoa_msoa = pd.merge(worker19_lsoa, lsoa_msoa_lookup, left_on='LSOA11CD', right_on='LSOA11CD', how='left')
        
        del worker19_lsoa, lsoa_msoa_lookup
        
        # get population of workers of MSOAs from LSOA level
        worker19_msoa = worker19_lsoa_msoa.groupby(['MSOA11CD'], as_index=False).agg(msoa_workers=('n_workers','sum')).reset_index() 
        
        worker19_lsoa_msoa = pd.merge(worker19_lsoa_msoa, worker19_msoa, on='MSOA11CD', how='left') 
        
        # create weights using population
        worker19_lsoa_msoa['weight'] = worker19_lsoa_msoa['n_workers']/worker19_lsoa_msoa['msoa_workers'] 
        
        # merge footfall df with weights
        people_counts_df_lsoa_daily = pd.merge(people_counts_df_msoa_daily, worker19_lsoa_msoa, left_on='msoa', right_on='MSOA11CD', how='left') 
        del people_counts_df_msoa_daily
        
        # apply weights to footfall
        people_counts_df_lsoa_daily['lsoa_people'] = people_counts_df_lsoa_daily['msoa_people']*people_counts_df_lsoa_daily['weight'] 
        people_counts_df_lsoa_daily = people_counts_df_lsoa_daily[['date_dt','LSOA11CD', 'MSOA11CD', 'purpose', 'lsoa_people', 'msoa_people']]
        
        # get LSOA areas to calculate footfall per hectare
        lsoa_area_query = "SELECT LSOA11CD, LSOA11NM, AREALHECT FROM `ons-hotspot-prod.ingest_geography.arealhect_lsoa_dec_2011`"
        query_job_lsoa_area = super().client.query(lsoa_area_query) 
        sam_lsoa_df = query_job_lsoa_area.to_dataframe()
        
        people_counts_df_lsoa_daily = pd.merge(people_counts_df_lsoa_daily,\
                                                sam_lsoa_df, left_on='LSOA11CD',\
                                                right_on='LSOA11CD', how='left')
        people_counts_df_lsoa_daily['lsoa_people_perHactares'] = people_counts_df_lsoa_daily['lsoa_people']/people_counts_df_lsoa_daily['AREALHECT']
        
        people_counts_df_lsoa_daily = people_counts_df_lsoa_daily[['date_dt', 'LSOA11CD', 'MSOA11CD', 'purpose', 'lsoa_people', 'msoa_people', 'lsoa_people_perHactares']]
        
        people_counts_df_lsoa_daily = people_counts_df_lsoa_daily.drop_duplicates()
        
        people_counts_df_lsoa_daily.columns = map(str.lower, people_counts_df_lsoa_daily.columns)  
        
        return people_counts_df_lsoa_daily
    