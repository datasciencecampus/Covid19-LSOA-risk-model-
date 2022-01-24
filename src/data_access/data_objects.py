
import os
import sys
import pandas as pd
from google.cloud import storage
import geopandas as gpd
from tqdm import tqdm
import fnmatch

# Import from local data files
current_path = os.path.abspath('.')
sys.path.append(os.path.dirname(current_path))

from data_access.idata import IData
import utils.config as conf
import utils.dynamic as dyn
import utils.model as md
from functools import reduce


class LSOA2011(IData):
    "A Class creating a static vars dataframe"
    def __init__(self):
        self.name = "data description"
    def create_dataframe(self):
        
        client = storage.Client()
        # set the bucket
        geography_bucket = 'hotspot-prod-geodata'
        bucket = client.get_bucket(geography_bucket)
        # Get the LSOA boundaries and download to store locally (on notebook)
        blob = bucket.get_blob('LSOA_2011_EW_BSC.geojson')
        blob.download_to_filename('LSOA_2011')
        LSOA_2011_map = gpd.read_file('LSOA_2011')
        return LSOA_2011_map



class AggregatedTestsLSOA(IData):
    "A Class for creating a dataframe containing the number of weekly positive tests in an LSOA from a specified date"
    def __init__(self):
        self.name = "data description"
    def create_dataframe(self):
        
        query = """ 
            SELECT Specimen_Date, Lower_Super_Output_Area_Code, SUM(positive_test)
            FROM `{}`
            WHERE Specimen_Date >={}
            GROUP BY Lower_Super_Output_Area_Code, Specimen_Date
        """.format(conf.data_location_big_query['cases'], conf.data_start_date) #TOC: Added in user defined table and date from config 

        
        query_job = super().client.query(
            query
        )

        cases_df = query_job.to_dataframe()
        

        cases_df.rename(columns={'f0_':'COVID_Cases'},inplace=True)
        cases_df['Specimen_Date']=pd.to_datetime(cases_df['Specimen_Date'])
        cases_df.rename(columns={'Specimen_Date':'Date'},inplace=True)
        cases_df.rename(columns={'Lower_Super_Output_Area_Code':'LSOA11CD'},inplace=True)
        cases_df_cumsum=cases_df.copy()
        cases_df_cumsum=cases_df_cumsum.sort_values(by=['LSOA11CD','Date'])  #sort by lsoa and week
        cases_df_cumsum=cases_df_cumsum.groupby(["LSOA11CD",'Date']).sum().groupby(level=0).cumsum().reset_index()
        cases_df_cumsum=cases_df_cumsum.rename(columns={'COVID_Cases':'cases_cumsum'})[['LSOA11CD','Date','cases_cumsum']]
        cases_df=cases_df.merge(cases_df_cumsum, how='left', on=['LSOA11CD', 'Date'])
        cases_df['Date']=cases_df['Date'].apply(lambda x: dyn.end_of_week(x))
        cases_df=cases_df.groupby(['Date','LSOA11CD']).agg(({'COVID_Cases':'sum', 'cases_cumsum':'max'})).reset_index()
        cases_df['Date']=pd.to_datetime(cases_df['Date'])

        return cases_df

    

class FlowsMarsData(IData):
    "A Class creating a static vars dataframe"
    def __init__(self):
        self.name = "Dynamic static variables normalized"
    '''
    This query and view present the data for mobilty flows in and out of LSOAs. 
    The calculations as essentially as follows:
    1. we first get the volume data aggregated for each of the in and out MSOAs. This is done in another view
       for brevity sake. See the query underlying the `wip.mars_daily_trips_from_and_to_home` view. 
    2. then we group by the `ons-hotspot-prod.ingest_risk_model.mid_year_pop19_lsoa` data by MSOA to get the 
       MSOA population, then we divide the LSOA population in the same data by that MSOA population numbers to get 
       a weight value that we will use to disaggregate the volume
    3. we then do a left join from this weight data based on population onto the volme data.
    4. finally we multiply the inflow and outflow volumes at MSOA level by the weights at LSOA level. 

    '''

    def create_dataframe(self):
        query = """
            SELECT
              date,
              msoa,
              lsoa11cd AS LSOA11CD,
              outflow_volume AS msoa_outflow_volume,
              inflow_volume AS msoa_inflow_volume,

              /* multiply volume by the weight here */
              outflow_volume * weight as lsoa_outflow_volume,
              inflow_volume * weight as lsoa_inflow_volume,
              weight as msoa_to_lsoa_weight
            FROM
              `{}`
            JOIN (
              SELECT
                tablea.lsoa11cd,
                tablea.msoa11cd,

                /* divide the all_people (ASSUMPTION: the all_people refers to LSOA level population) by 
                the MSOA population to get the % weight the LSOA represents of the total MSOA*/
                all_people / msoa_population AS weight
              FROM (
                SELECT
                  LSOA11CD,
                  MSOA11CD,
                  ALL_PEOPLE
                FROM
                  `ons-hotspot-prod.ingest_risk_model.mid_year_pop19_lsoa`) tableA
              LEFT JOIN (
                SELECT
                  msoa11cd,
                  SUM(ALL_PEOPLE) AS msoa_population
                FROM
                  `ons-hotspot-prod.ingest_risk_model.mid_year_pop19_lsoa`
                GROUP BY
                  msoa11cd) tableB
              ON
                tablea.msoa11cd = tableb.msoa11cd) tableB
            ON
              msoa = tableB.msoa11cd
            """.format(conf.data_location_big_query['mobility_MARS']) #TOC: Added in user defined table from config 
        query_job = super().client.query(
            query
        )

        df_flows_mars_data = query_job.to_dataframe()
        
        df_flows_mars_data.rename(columns={'date':'Date'},inplace=True)
        df_flows_mars_data['Date']=pd.to_datetime(df_flows_mars_data['Date'])
        df_flows_mars_data=df_flows_mars_data[df_flows_mars_data['Date']>=pd.to_datetime(conf.data_start_date)].reset_index(drop=True)
        
        df_flows_mars_data=df_flows_mars_data.groupby(['Date', 'LSOA11CD'])\
        [['lsoa_inflow_volume']].sum().reset_index()

        # WEEKLY SAMPLING
        df_flows_mars_data['Date']=df_flows_mars_data['Date'].apply(lambda x: dyn.end_of_week(x))

        df_flows_mars_data=df_flows_mars_data.groupby(['Date', 'LSOA11CD'])[['lsoa_inflow_volume']].sum().reset_index()
        
        df_flows_mars_data['Date']=pd.to_datetime(df_flows_mars_data['Date'])
        
        return df_flows_mars_data

class LSOA_MidYear(IData):
    "A Concrete Class that implements the IProduct interface"
    def __init__(self):
        self.name = "LSOA_MidYear"
        
    def __bin_ages(self,dfin, dfout, bins):
        for bin in bins:
            dfout['age_'+bin[0] + '_to_' + bin[1]] = dfin.loc[:, bin[0] : bin[1]].sum(axis=1)
        return dfout
    
    def create_dataframe(self):
        
        query_job = super().client.query(
            """
            SELECT *
            FROM `ons-hotspot-prod.ingest_risk_model.mid_year_pop19_lsoa` 
            """
        )

        age_df = query_job.to_dataframe()
        
        
        age_df_subset=age_df.loc[:,'LSOA11CD':'MF_AGE_90_PLUS']
        age_df_subset.columns=age_df_subset.columns.str.replace('MF_AGE_',"")

    

        #bin ages
        binned=age_df_subset.loc[:,'LSOA11CD':'ALL_PEOPLE']
        #age_bins=[['0','3'],['4','11'],['12','18'],['19','30'],['31','50'],['51','70'],['71','90_PLUS']]
        
        age_bins=[['0','12'],['13','17'],['18','29'],['30','39'],['40','49'],
                  ['50','54'],['55','59'],['60','64'],['65','69'],['70','74'],
                  ['75','79'],['80','90_PLUS']]

      
        binned = self.__bin_ages(age_df_subset, binned, age_bins)

        binned.drop(columns=['LSOA11NM','MSOA11CD','MSOA11NM','LTLA20CD','LTLA20NM','UTLA20CD','UTLA20NM','RGN19CD','RGN19NM'],inplace=True)

        return binned


class MobilityClustersProcessed(IData):
    def __init__(self):
        self.name = "data"
    def create_dataframe(self):
        query = "SELECT * FROM `ons-hotspot-prod.ingest_geography.lsoa_mobility_cluster_ew_lu`"
        
        query_job = super().client.query(
            query
        )

        travel_clusters = query_job.to_dataframe()
        travel_clusters.drop(columns=['lsoa11nm', 'index_right'], inplace=True)
        travel_clusters.rename(columns={'ct_2':'travel_cluster', 'lsoa11cd':'LSOA11CD'}, inplace=True)

        travl_clst_dict={'L3. Transition area from L2 to L4': 'L4. >70% suburban dwellers',
               'L5. Transition area from L4 to L6': 'L4. >70% suburban dwellers',
               'L7. Transition area from L6 to L8':'L6. >70% exurban dwellers'}

        travel_clusters['travel_cluster']=travel_clusters['travel_cluster'].map(travl_clst_dict).fillna(travel_clusters['travel_cluster'])

        travl_renam_dict={'L1. >70% metropolitan core dwellers':'L1. >70% metropolitan core dwellers',
                         'L2. >70% outer metropolitan dwellers':'L2. >70% outer metropolitan dwellers',
                         'L4. >70% suburban dwellers':'L3. >70% suburban dwellers',
                         'L6. >70% exurban dwellers':'L4. >70% exurban dwellers',
                         'L8. >70% rural dwellers':'L5. >70% rural dwellers'}

        travel_clusters['travel_cluster']=travel_clusters['travel_cluster'].\
        map(travl_renam_dict).fillna(travel_clusters['travel_cluster'])
        return travel_clusters

class LSOADailyFootfall(IData):
    def __init__(self):
        self.name = "data"
    def create_dataframe(self):
        df=DeimosAggregated().create_dataframe()

#         query = "SELECT date_dt as Date, lsoa11cd as LSOA11CD, lsoa_people, lsoa_people_perHactares, purpose, FROM `{}`".format(conf.data_location_big_query['mobility_DEIMOS']) #The input to this query is the table produced by the data object DeimosAggregated().

#         query_job = super().client.query(
#             query
#         )  
#         df = query_job.to_dataframe()
        df=df.rename(columns={'date_dt':'Date', 
                              'lsoa11cd':'LSOA11CD',
                             'lsoa_people_perhactares':'lsoa_people_perHactares'})
    
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date']=df['Date'].apply(lambda x: dyn.end_of_week(x)) #TOC: Changed to the swifter mehod as offset shifts days in to different weeks
        #df['Date'] = df['Date'] + pd.offsets.Week(weekday=6) TOC: Changed to the swifter mehod as offset shifts days in to different weeks
        df['worker_footfall_sqkm']=0
        df['visitor_footfall_sqkm']=0
        df['resident_footfall_sqkm']=0
        df.loc[df['purpose']=='Worker', 'worker_footfall_sqkm']=df[df['purpose']=='Worker'].lsoa_people_perHactares.div(0.01)
        df.loc[df['purpose']=='Visitor', 'visitor_footfall_sqkm']=df[df['purpose']=='Visitor'].lsoa_people_perHactares.div(0.01)
        df.loc[df['purpose']=='Resident', 'resident_footfall_sqkm']=df[df['purpose']=='Resident'].lsoa_people_perHactares.div(0.01)
        df=df.groupby(["Date","LSOA11CD"]).sum().reset_index().drop(columns=["lsoa_people","lsoa_people_perHactares"])
        df['total_footfall_sqkm']=df["worker_footfall_sqkm"]+df["visitor_footfall_sqkm"]+df["resident_footfall_sqkm"]
        df['worker_visitor_footfall_sqkm']=df["worker_footfall_sqkm"]+df["visitor_footfall_sqkm"]
        df['Date'] = pd.to_datetime(df['Date'])
        return df

class LSOAVaccinations(IData):
    def __init__(self):
        self.name = "data"
    def create_dataframe(self):
        query = "SELECT * FROM `{}`".format(conf.data_location_big_query['vaccination']) #TOC: Added in user defined table from config #ons-hotspot-prod.ingest_vaccination.lsoa_vaccinations_8_july_2021
        
        query_job = super().client.query(
            query
        )

        vaccination_df = query_job.to_dataframe()
        
        #vaccination data renaming columns
        vaccination_df.drop(columns='int64_field_0',inplace=True)  
        vaccination_df.rename(columns={'LSOA_OF_RESIDENCE':'LSOA11CD'},inplace=True)
        #seperate male and female
        df_vaccination_male=vaccination_df[vaccination_df['GENDER']=='Male']      
        df_vaccination_female=vaccination_df[vaccination_df['GENDER']=='Female']

        #add suffixes and bring them back together
        keep_same = {'LSOA11CD', 'vcc_date','GENDER'}     
        df_vaccination_male.columns = ['{}{}'.format(c, '' if c in keep_same else '_vaccine_male')
                  for c in df_vaccination_male.columns]

        df_vaccination_female.columns = ['{}{}'.format(c, '' if c in keep_same else '_vaccine_female')
                  for c in df_vaccination_female.columns]

        vaccination_df=df_vaccination_male.merge(df_vaccination_female, how='outer',
                                                 on=['LSOA11CD','vcc_date']).drop(columns=['GENDER_x','GENDER_y'])

        vaccination_df.fillna(0.0, inplace=True)

        vaccination_df=vaccination_df[vaccination_df['LSOA11CD'].str.startswith('E')].reset_index(drop=True)


        # aggregate vaccinations records at LSOA level
        vaccination_df.rename(columns={'vcc_date':'Date'},inplace=True)

        cols_to_aggregate=['dose_first_vaccine_male', 'dose_first_vaccine_female','dose_second_vaccine_male','dose_second_vaccine_female'] 

        vaccination_df = vaccination_df.groupby(['Date','LSOA11CD'])[cols_to_aggregate].sum().reset_index()

        #AA-REMOVED A DUPLICATE OF LINE ABOVE

        vaccination_df['Date']=pd.to_datetime(vaccination_df['Date'])
        vaccination_df=vaccination_df[vaccination_df['Date']>=pd.to_datetime(conf.data_start_date)].reset_index(drop=True) 
        
        vaccination_df['total_vaccinated_first_dose']=vaccination_df['dose_first_vaccine_male'] +vaccination_df['dose_first_vaccine_female']
        vaccination_df['total_vaccinated_second_dose']=vaccination_df['dose_second_vaccine_male'] +vaccination_df['dose_second_vaccine_female']
        # Proportion of vaccinationed populaition (cumsum)
        vaccination_df_cumsum=vaccination_df.copy()
        vaccination_df_cumsum=vaccination_df_cumsum.sort_values(by=['LSOA11CD','Date'])  #sort by lsoa and week
        vaccination_df_cumsum=vaccination_df_cumsum.groupby(["LSOA11CD",'Date']).sum().groupby(level=0).cumsum().reset_index()
        vaccination_df_cumsum=vaccination_df_cumsum.rename(columns={'total_vaccinated_second_dose':'full_vacc_cumsum'})[['LSOA11CD','Date','full_vacc_cumsum']]

        #   PROPORTION OF THE VACCINATED POPULATION DURING THAT DATE


        #  VACCINATED PER UNIT AREA (SQ KM)
        # vaccination_df.loc[:,['total_vaccinated_first_dose']]=\
        # vaccination_df[['total_vaccinated_first_dose']].div(vaccination_df['Area'],axis=0)  

        # MERGE CUMSUM WITH VACC and divide by people
        vaccination_df=vaccination_df.merge(vaccination_df_cumsum, on=['LSOA11CD','Date'],how='left')

        few_cols_vacct=['Date', 'LSOA11CD', 'total_vaccinated_first_dose','full_vacc_cumsum','total_vaccinated_second_dose']

        vaccination_df=vaccination_df[few_cols_vacct]

        #  DAILY SAMPLING
        vaccination_df=vaccination_df.groupby(['Date','LSOA11CD'])['total_vaccinated_first_dose','full_vacc_cumsum','total_vaccinated_second_dose'].\
        sum().reset_index()

        #  WEEKLY SAMPLING
        vaccination_df['Date']=vaccination_df['Date'].apply(lambda x: dyn.end_of_week(x))

        vaccination_df=vaccination_df.groupby(['Date','LSOA11CD']).agg(({'total_vaccinated_first_dose':'sum',
                                                                         'total_vaccinated_second_dose':'sum',
                                                                         'full_vacc_cumsum':'max'})).reset_index()
        vaccination_df['Date']=pd.to_datetime(vaccination_df['Date'])

        del vaccination_df_cumsum
        return vaccination_df

class StaticSubset(IData):
    def __init__(self):
        self.name = "data"
    def create_dataframe(self):
        table = conf.static_data_file
        col_list = ', '.join(conf.static_subset)
        query = f"SELECT {col_list}  FROM `{table}`"
        
        query_job = super().client.query(
            query
        )  
        df = query_job.to_dataframe()
        df=df[df['LSOA11CD'].str.startswith('E')].reset_index(drop=True)
        df = df.drop_duplicates(subset=['LSOA11CD'])
        df['Country']='England'
        

        return df

class StaticNormalised(IData):
    def __init__(self):
        self.name = "Static data which has been normalised"
    def create_dataframe(self):
        table = conf.static_data_file
        query = f"SELECT * FROM `{table}`"
        
        query_job = super().client.query(
            query
        )  
        df = query_job.to_dataframe()
        return df


class AllTranches(IData):
    def __init__(self):
        self.name = "data"
    def create_dataframe(self):
        
        table = conf.static_data_file
    
        query_static = f"SELECT * FROM `{table}`"
        query_job_static = super().client.query(
            query_static
        ) 
        static_variables_df = query_job_static.to_dataframe()
        
        table = conf.lagged_dynamic_non_stationary
        
        query_dynamic = f"SELECT * FROM `{table}`"
        query_job_dynamic = super().client.query(
            query_dynamic
        )  
        dynamic_lagged_variables_df = query_job_dynamic.to_dataframe()
        
        #common columns between the static and dynamic datasets
        commn_colmns=list(set(dynamic_lagged_variables_df.columns) & set(static_variables_df.columns))


        # Get a joint dataframe
        df_all_tranches_dynamic_static=dynamic_lagged_variables_df.merge(static_variables_df,on=commn_colmns,how='inner')
 #.merge(static_variables_df,on=commn_colmns,how='inner')

        # Choose a specific date to start the risk model training from

        cut_off_datum=conf.model_start_date

        df_all_tranches_dynamic_static['Date']=pd.to_datetime(df_all_tranches_dynamic_static['Date'])


        df_all_tranches_dynamic_static['Date']=df_all_tranches_dynamic_static['Date'].dt.date

        df_all_tranches_dynamic_static=df_all_tranches_dynamic_static[df_all_tranches_dynamic_static['Date']>=\
                                                                      pd.to_datetime(cut_off_datum).date()].\
        sort_values(by='Date').reset_index(drop=True)

        # Convert dates to dummy weeks

        date_list=sorted(df_all_tranches_dynamic_static['Date'].unique())

        date_list=[str(x) for x in date_list]



        week_list=['week_'+str(x+1) for x in range(len(date_list))]

        date_dict=dict(zip(date_list,week_list))
        # Convert dates to Months

        date_list=sorted(df_all_tranches_dynamic_static['Date'].unique())

        month_list=[pd.to_datetime(x).month_name() for x in date_list]

        year_list=[pd.to_datetime(x).year for x in date_list]

        month_list = [str(i) +'_'+ str(j) for i, j in zip(month_list, year_list)]

        date_list=[str(x) for x in date_list]

        month_dict=dict(zip(date_list,month_list))


        df_all_tranches_dynamic_static['Date']=df_all_tranches_dynamic_static['Date'].astype(str)

        df_all_tranches_dynamic_static['week']=df_all_tranches_dynamic_static['Date'].map(date_dict)


        #create a Month variable
        df_all_tranches_dynamic_static['Month']=df_all_tranches_dynamic_static['Date'].map(month_dict)
        # Define a region dataframe

        df_rgns=static_variables_df[['LSOA11CD','LSOA11NM','MSOA11CD','MSOA11NM','LTLA20CD','LTLA20NM',\
                         'UTLA20CD','UTLA20NM','RGN19CD','RGN19NM','travel_cluster']].drop_duplicates().reset_index(drop=True)
        # remove any weeks for which there are no non-zero target variables
        # TRAINING ON ALL-ZEROS DATA, COULD THIS WORK ?

        no_cases_week=df_all_tranches_dynamic_static.groupby('week')['COVID_Cases_per_unit_area'].sum().\
        index[df_all_tranches_dynamic_static.groupby('week')['COVID_Cases_per_unit_area'].sum().values==0]


        df_all_tranches_sbset=df_all_tranches_dynamic_static[~df_all_tranches_dynamic_static['week'].isin(no_cases_week)]

        return df_all_tranches_sbset

# If zero-inflated regression is used for static training then
# apprpriate changes need to be made in the location of the
# static residual wip tables
# ons-hotspot-prod.wip.multi_grp_pred_no/zir_only_static_main

class DynamicChangesWeekly(IData):
    def __init__(self):
        self.name = "data"
    def create_dataframe(self):
        table = conf.project_name + '.' + conf.risk_pred + conf.model_suffixes['static_main']
        
        query = f"SELECT * FROM `{table}`"
        query_job = super().client.query(
            query
        )
        
        df_pred_tc_all_week_static_zir = query_job.to_dataframe()
        #df_pred_tc_all_week_static_zir.rename(columns={'tranche':'week'},inplace=True)

        df_all_tranches_sbset = AllTranches().create_dataframe()
        
        # Residuals from static predictors
        # These residuals will be used as a target variable
        # based on change of dynamic predictors
        df_pred_tc_all_week_static_zir['Residual']=df_pred_tc_all_week_static_zir['Actual_cases']-\
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
        dynamic_colmns_vaccine=['LSOA11CD','Date']+conf.dynamic_vacc

        # CONCAT cumulative vaccine DATA (FROM THE PREVIOUS WEEK) FOR ALL THE LSOA
        cum_vaccine_df=md.cumulative_data(df_all_tranches_sbset, dynamic_colmns_vaccine)
        print('Vaccination data preparation for dynamic training....Finished')
        
        
        print('Mobility data preparation for dynamic training....Started')
        #Change in mobility indicators from previous weeks
        dynamic_colmns_mobility=['LSOA11CD','Date']+conf.dynamic_mobility

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

# If zero-inflated regression is used for static training then
# apprpriate changes need to be made in the location of the
# static residual wip tables
# ons-hotspot-prod.wip.multi_grp_coef_zir_only_static_main

class StaticChangesWeekly(IData):
    def __init__(self):
        self.name = "data"
    def create_dataframe(self):
        query = "SELECT * FROM `ons-hotspot-prod.wip.multi_grp_coef_no_zir_only_static_main`"
        query_job = super().client.query(
            query
        )
        df_coef_tc_all_week_static_zir = query_job.to_dataframe()
        return df_coef_tc_all_week_static_zir

# If zero-inflated regression is used for static training then
# apprpriate changes need to be made in the location of the
# static residual wip tables
# ons-hotspot-prod.wip.multi_grp_coef_ci_zir_only_static_main

class StaticChangesWeekly_ci(IData):
    def __init__(self):
        self.name = 'data'
    def create_dataframe(self):
        query = "SELECT * FROM `ons-hotspot-prod.wip.multi_grp_coef_ci_no_zir_only_static_main`"
        query_job = super().client.query(
            query
        )
        df_coef_ci_tc_all_week_static_zir = query_job.to_dataframe()
        return df_coef_ci_tc_all_week_static_zir


class MSOA2011(IData):
    "A Class creating a MSOA Geojson file"
    def __init__(self):
        self.name = "data description"
    def create_dataframe(self):
        
        client = storage.Client()
        # set the bucket
        geography_bucket = 'hotspot-prod-geodata'
        bucket = client.get_bucket(geography_bucket)
        # Get the LSOA boundaries and download to store locally (on notebook)
        blob = bucket.get_blob('msoa.geojson')
        blob.download_to_filename('msoa')
        MSOA_2011_map = gpd.read_file('msoa')
        
        
        MSOA_2011_map=MSOA_2011_map.set_crs(27700,allow_override=True) #CRS was read in wrong from the file

        MSOA_2011_map['AREALHECT']=MSOA_2011_map.geometry.area
        
        return MSOA_2011_map


class DeimosAggregated(IData):
    def __init__(self):
        self.name='Deimos data aggregated from deimos_ag'
        
    def create_dataframe(self):
        
        query_people_counts="""SELECT date_dt, purpose, msoa , SUM(people) as msoa_people
                                FROM `ons-hotspot-prod.ingest_deimos_2021.uk_footfall_people_counts_ag`
                                WHERE date_dt>={} AND ((msoa LIKE 'E%') OR (msoa LIKE 'W%')) 
                                GROUP BY date_dt, purpose, msoa""".format(conf.data_start_date)  #read in deimos date seprated by age and gender
        query_job_deimos = super().client.query(query_people_counts) 
        people_counts_df_msoa_daily = query_job_deimos.to_dataframe()

#         people_counts_df_msoa_daily = pd.DataFrame()
#         daily_df = hourly_df.groupby(['date_dt','msoa', 'age','gender','purpose'], \
#                             as_index=False).agg({'people':'sum'})
#         if (people_counts_df_msoa_daily.empty):
#             people_counts_df_msoa_daily = daily_df
#         else:
#             people_counts_df_msoa_daily = people_counts_df_msoa_daily.append(daily_df) 

#         del hourly_df


#         #combine age and gender
#         people_counts_df_msoa_daily = people_counts_df_msoa_daily.groupby(['date_dt', 'msoa', 'purpose'],\
#                                         as_index=False).agg(msoa_people=('people','sum'))

#         #only include England and Wales
#         people_counts_df_msoa_daily = people_counts_df_msoa_daily.loc[people_counts_df_msoa_daily['msoa'].str.startswith('E') \
#                                     |people_counts_df_msoa_daily['msoa'].str.startswith('W') ]

        msoa=MSOA2011().create_dataframe() #get msoa area data
        
        people_counts_df_msoa_daily = pd.merge(people_counts_df_msoa_daily,\
                                       msoa, left_on='msoa',\
                                       right_on='MSOA11CD', how='left')
        
        people_counts_df_msoa_daily['msoa_people_perHactares']= people_counts_df_msoa_daily['msoa_people']/people_counts_df_msoa_daily['AREALHECT']
       
    
        people_counts_df_msoa_daily = people_counts_df_msoa_daily[['date_dt', 'msoa', 'purpose', \
                                                           'msoa_people', 'AREALHECT', 'msoa_people_perHactares']] #final msoa counts
        
        
        population_query="SELECT DISTINCT * FROM `ons-hotspot-prod.ingest_risk_model.mid_year_pop19_lsoa`"  #get population to create weights for de-aggregation
        query_job_pop = super().client.query(population_query ) 
        pop19_lsoa = query_job_pop.to_dataframe()
        
        pop19_lsoa['pop_above18'] = pop19_lsoa.loc[:, 'MF_AGE_19':"MF_AGE_90_PLUS"].sum(axis=1)  #get over 18 ages
        pop19_lsoa = pop19_lsoa[['LSOA11CD', 'ALL_PEOPLE', 'pop_above18']]
        
        lookup_query="SELECT lsoa11cd as LSOA11CD,msoa11cd as MSOA11CD FROM `ons-hotspot-prod.ingest_geography.PCD_OA_LSOA_MSOA_LAD_FEB21_UK_LU`" #get look up for lsoa msoa
        query_job_lookup = super().client.query(lookup_query) 
        lsoa_msoa_lookup = query_job_lookup.to_dataframe()
        
        lsoa_msoa_lookup = lsoa_msoa_lookup.drop_duplicates() #remove duplicates as lsoas come up multiple time per msoa

        pop19_lsoa_msoa = pd.merge(pop19_lsoa, lsoa_msoa_lookup, \
                                   left_on='LSOA11CD', right_on='LSOA11CD', how='left')
        
        del pop19_lsoa, lsoa_msoa_lookup
        
        pop19_msoa = pop19_lsoa_msoa.groupby(['MSOA11CD'], \
                                           as_index=False).agg(msoa_pop=('ALL_PEOPLE','sum'), \
                                                               msoa_pop_above18=('pop_above18','sum')).reset_index() #get population of msoas from lsoa level
        pop19_lsoa_msoa = pd.merge(pop19_lsoa_msoa, pop19_msoa, on='MSOA11CD', how='left') 
        pop19_lsoa_msoa['weight'] = pop19_lsoa_msoa['pop_above18']/pop19_lsoa_msoa['msoa_pop_above18'] #create weights using population
        people_counts_df_lsoa_daily = pd.merge(people_counts_df_msoa_daily, pop19_lsoa_msoa,\
                                      left_on='msoa', right_on='MSOA11CD', how='left') #merge footfall with df with weights
        del people_counts_df_msoa_daily
        
        people_counts_df_lsoa_daily['lsoa_people'] = people_counts_df_lsoa_daily['msoa_people']*people_counts_df_lsoa_daily['weight'] #apply weights to footfall
        people_counts_df_lsoa_daily = people_counts_df_lsoa_daily[['date_dt','LSOA11CD', 'MSOA11CD', 'purpose', 'lsoa_people', 'msoa_people']]
        
        #get lsoa areas to get footfall per hectares
        lsoa_area_query="SELECT LSOA11CD, LSOA11NM, AREALHECT FROM `ons-hotspot-prod.ingest_geography.arealhect_lsoa_dec_2011`"
        query_job_lsoa_area = super().client.query(lsoa_area_query) 
        sam_lsoa_df = query_job_lsoa_area.to_dataframe()
        
        people_counts_df_lsoa_daily = pd.merge(people_counts_df_lsoa_daily,\
                                                sam_lsoa_df, left_on='LSOA11CD',\
                                                right_on='LSOA11CD', how='left')
        people_counts_df_lsoa_daily['lsoa_people_perHactares'] = people_counts_df_lsoa_daily['lsoa_people']/people_counts_df_lsoa_daily['AREALHECT']
        
        people_counts_df_lsoa_daily = people_counts_df_lsoa_daily[['date_dt', 'LSOA11CD', 'MSOA11CD', 'purpose', 'lsoa_people', 'msoa_people', 'lsoa_people_perHactares']]
        
        people_counts_df_lsoa_daily = people_counts_df_lsoa_daily.drop_duplicates()
        
        people_counts_df_lsoa_daily.columns = map(str.lower, people_counts_df_lsoa_daily.columns)  #need lower stringged column names to fit in with rest of the pipeline
        
        

        
        return people_counts_df_lsoa_daily


class DeimosEndTrip(IData):
    def __init__(self):
        self.name='Deimos data aggregated from end trip'
        
    def create_dataframe(self):
        
        trip_end_query="""SELECT DISTINCT date as Date, msoa, journey_purpose, SUM(journeys_starting) AS outflow_volume , SUM(journeys_ending) as inflow_volume 
                          FROM `ons-hotspot-prod.ingest_deimos_2021.uk_trips_trip_end_count`
                          GROUP BY date, msoa, msoa_name, journey_purpose
                          ORDER BY date, msoa"""
        
        query_job_deimos_trip_end = super().client.query(trip_end_query) 
        deimos_trip_end_count_msoa_daily = query_job_deimos_trip_end.to_dataframe()
        
        #only include England and Wales
        deimos_trip_end_count_msoa_daily = deimos_trip_end_count_msoa_daily.loc[deimos_trip_end_count_msoa_daily['msoa'].str.startswith('E') \
                                    |deimos_trip_end_count_msoa_daily['msoa'].str.startswith('W') ]
        
        #get lsoa areas to get footfall per hectares
        lsoa_area_query="SELECT LSOA11CD, LSOA11NM, AREALHECT FROM `ons-hotspot-prod.ingest_geography.arealhect_lsoa_dec_2011`"
        query_job_lsoa_area = super().client.query(lsoa_area_query) 
        arealhect_lsoa_dec_2011 = query_job_lsoa_area.to_dataframe()
        
        lookup_query="SELECT lsoa11cd as LSOA11CD,msoa11cd as MSOA11CD FROM `ons-hotspot-prod.ingest_geography.PCD_OA_LSOA_MSOA_LAD_FEB21_UK_LU`" #get look up for lsoa msoa
        query_job_lookup = super().client.query(lookup_query) 
        lsoa11_msoa11_lookup = query_job_lookup.to_dataframe()
        
        lsoa11_msoa11_lookup=lsoa11_msoa11_lookup.drop_duplicates()
        
        arealhect_msoa = pd.merge(arealhect_lsoa_dec_2011,lsoa11_msoa11_lookup, on='LSOA11CD', how='left').groupby('MSOA11CD', \
                                                as_index=False).agg({'AREALHECT':'sum'})
        
        deimos_trip_end_count_msoa_daily = pd.merge(deimos_trip_end_count_msoa_daily,\
                                       arealhect_msoa, left_on='msoa',\
                                       right_on='MSOA11CD', how='left')
        
        deimos_trip_end_count_msoa_daily['msoa_inflow_volume_perHactares']  = deimos_trip_end_count_msoa_daily['inflow_volume']/deimos_trip_end_count_msoa_daily['AREALHECT']
        deimos_trip_end_count_msoa_daily['msoa_outflow_volume_perHactares']  = deimos_trip_end_count_msoa_daily['outflow_volume']/deimos_trip_end_count_msoa_daily['AREALHECT']


        deimos_trip_end_count_msoa_daily = deimos_trip_end_count_msoa_daily[['Date', 'msoa', 'journey_purpose', \
                                                           'inflow_volume','outflow_volume',\
                                                            'AREALHECT', 'msoa_inflow_volume_perHactares',\
                                                             'msoa_outflow_volume_perHactares']]
        
        population_query="SELECT DISTINCT * FROM `ons-hotspot-prod.ingest_risk_model.mid_year_pop19_lsoa`"  #get population to create weights for de-aggregation
        query_job_pop = super().client.query(population_query ) 
        pop19_lsoa = query_job_pop.to_dataframe()
        
        
        pop19_lsoa['pop_above18'] = pop19_lsoa.loc[:, 'MF_AGE_19':"MF_AGE_90_PLUS"].sum(axis=1)
        

        pop19_lsoa = pop19_lsoa[['LSOA11CD', 'ALL_PEOPLE', 'pop_above18']]
        
        lsoa11_msoa11_lookup.columns=map(str.lower, lsoa11_msoa11_lookup.columns)
        
        pop19_lsoa_msoa = pd.merge(pop19_lsoa, lsoa11_msoa11_lookup, \
                           left_on='LSOA11CD', right_on='lsoa11cd', how='left')
        pop19_msoa = pop19_lsoa_msoa.groupby(['msoa11cd'], \
                                           as_index=False).agg(msoa_pop=('ALL_PEOPLE','sum'), \
                                                               msoa_pop_above18=('pop_above18','sum')).reset_index()
        
        pop19_lsoa_msoa = pd.merge(pop19_lsoa_msoa, pop19_msoa, on='msoa11cd', how='left')
        pop19_lsoa_msoa['weight'] = pop19_lsoa_msoa['pop_above18']/pop19_lsoa_msoa['msoa_pop_above18']
        
        trip_end_count_lsoa_daily = pd.merge(deimos_trip_end_count_msoa_daily, pop19_lsoa_msoa,\
                                      left_on='msoa', right_on='msoa11cd', how='left')
        
        trip_end_count_lsoa_daily['lsoa_inflow_volume'] = trip_end_count_lsoa_daily['inflow_volume']*trip_end_count_lsoa_daily['weight']
        trip_end_count_lsoa_daily['lsoa_outflow_volume'] = trip_end_count_lsoa_daily['outflow_volume']*trip_end_count_lsoa_daily['weight']

        trip_end_count_lsoa_daily = trip_end_count_lsoa_daily[['Date','lsoa11cd', 'msoa11cd', 'journey_purpose',\
                                                               'lsoa_inflow_volume', 'lsoa_outflow_volume']]
        
        trip_end_count_lsoa_daily = pd.merge(trip_end_count_lsoa_daily,\
                                                arealhect_lsoa_dec_2011, left_on='lsoa11cd',\
                                                right_on='LSOA11CD', how='left')
        trip_end_count_lsoa_daily['lsoa_inflow_volume_perHactares'] = trip_end_count_lsoa_daily['lsoa_inflow_volume']/trip_end_count_lsoa_daily['AREALHECT']
        trip_end_count_lsoa_daily['lsoa_outflow_volume_perHactares'] = trip_end_count_lsoa_daily['lsoa_outflow_volume']/trip_end_count_lsoa_daily['AREALHECT']
        
        trip_end_count_lsoa_daily = trip_end_count_lsoa_daily[['Date', 'lsoa11cd', 'msoa11cd', 'journey_purpose',\
                                                               'lsoa_inflow_volume', 'lsoa_outflow_volume',\
                                                               'lsoa_inflow_volume_perHactares', \
                                                               'lsoa_outflow_volume_perHactares']]
        
        
        trip_end_count_lsoa_daily['Date'] = pd.to_datetime(trip_end_count_lsoa_daily['Date'])
        
        trip_end_count_lsoa_daily['Date']=trip_end_count_lsoa_daily['Date'].apply(lambda x: dyn.end_of_week(x)) 
        
        trip_end_count_lsoa_daily.loc[trip_end_count_lsoa_daily['journey_purpose']=='Commute', 'commute_inflow_sqkm']=trip_end_count_lsoa_daily[trip_end_count_lsoa_daily['journey_purpose']=='Commute'].lsoa_inflow_volume_perHactares.div(0.01)
        
        trip_end_count_lsoa_daily.loc[trip_end_count_lsoa_daily['journey_purpose']=='Other', 'other_inflow_sqkm']=trip_end_count_lsoa_daily[trip_end_count_lsoa_daily['journey_purpose']=='Other'].lsoa_inflow_volume_perHactares.div(0.01)
        
        trip_end_count_lsoa_daily=trip_end_count_lsoa_daily.groupby(["Date","lsoa11cd"]).sum().reset_index().drop(columns=["lsoa_inflow_volume","lsoa_outflow_volume",'lsoa_inflow_volume_perHactares','lsoa_outflow_volume_perHactares'])
        
        trip_end_count_lsoa_daily['Date'] = pd.to_datetime(trip_end_count_lsoa_daily['Date'])
        trip_end_count_lsoa_daily=trip_end_count_lsoa_daily.rename(columns={'lsoa11cd':'LSOA11CD'})

        trip_end_count_lsoa_daily['Date'] = pd.to_datetime(trip_end_count_lsoa_daily['Date'])


    


        
        return trip_end_count_lsoa_daily
