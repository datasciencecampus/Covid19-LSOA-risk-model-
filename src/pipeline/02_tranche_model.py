# import modules
import sys
import os
import pandas as pd
import numpy as np


# import from local data files
current_path = os.path.abspath('.')
sys.path.append(os.path.dirname(current_path))

from data_access.data_factory import DataFactory as factory
from utils import config as cf

from functools import reduce

#####################################
#### PARAMETERS FROM CONFIG FILE ####
#####################################


''' Number of different combinations of grid search hyperparameters
Default is 500, use a lower value, >=1 to speed-up the evaluations
Will only be used when regression with regularisation model is used
for making predictions'''
    
# Only required if model uses regularisation
parm_spce_grid_srch=cf.parm_spce_grid_srch

# Create a list of alphas to cross-validate against
alphas_val = cf.alphas_val


#When should the weekly training start from
strt_training_period=cf.chsen_datum

#Flag to set if one chooses Zero inflated regression model

'''By default, the flag is reset as the architecture of
the model is more suited for tranches of low prevalence
when there are excessively large number of LSOAs with zero
reported cases. This architecture is not suitable for longer
time periods where the dataset is less likely to have excessive
number of zero reported cases
'''

zero_inf_flg_st=cf.zero_infltd_modl

'''By default, the flag is reset which means
regression with regularisation model will be
used for making predictions (sklearn package). Do note: irrespective 
of the status of the flag, the significant coefficients 
will be through linear regression model (statsmodel package)
'''
lin_regr_or_regrlsn=cf.linear_rgr_flg

################
### INGESTS ####
################

# static variables
static_df = factory.get('static_vars_for_modelling').create_dataframe()

# LSOA Area information to normalise footfall
area_df = factory.get('static_subset_for_norm').create_dataframe()

# Cases data
cases_df = factory.get('aggregated_tests_lsoa').create_dataframe()

# Mobility data
deimos_footfall_df = factory.get('lsoa_daily_footfall').create_dataframe()

##############################
#### PROCESS STATIC DATA #####
##############################

### DROP THE FEATURES THAT THE CORRELATION ANALYSIS SHOWED THAT WE DON'T NEED
### FILTER FOR NUMERIC COLUMNS ONLY
### AGE COLUMNS ARE DROPPED

static_df_num=static_df.set_index(['LSOA11CD','travel_cluster'])
static_df_num=static_df_num.select_dtypes(include=np.number)
static_df_num=static_df_num[[x for x in static_df_num.columns if x!='Area']]


### SELECT THE REQUIRED COLUMNS ###

sel_colmns_set = ['CENSUS_2011_ASIAN_ASIAN_BRITISH','METHOD_OF_TRAVEL_TO_WORK_NON_MOTORISED','METHOD_OF_TRAVEL_TO_WORK_Public_TRANSPORT',\
                'METHOD_OF_TRAVEL_TO_WORK_WORK_MAINLY_FROM_HOME','FAMILIES_WITH_DEPENDENT_CHILDREN_NO_DEPENDENT_CHILDREN',\
                'care', 'meat_and_fish_processing','ready_meals', 'textiles', 'warehousing']


static_df_num=static_df_num[sel_colmns_set]

assert static_df_num.shape == (cf.lsoa_count, 10)


##############################
#### PROCESS AREA DATA #####
##############################

area_df=area_df[['LSOA11CD','ALL_PEOPLE','Area']]

#### JOIN TO STATIC DATA ####
static_df_num=static_df_num.reset_index()
static_df_num=static_df_num.merge(area_df,on=['LSOA11CD'],how='inner')

### NORMALISE ###

# Normalise the IDBR variables: 
'''expressed as number of residents per unit area
who are working in high risk industries'''
fctr=1
idbr_norm='Area'
static_df_num['care']=(static_df_num['care'].div(static_df_num[idbr_norm]))*(fctr)
static_df_num['meat_and_fish_processing']=(static_df_num['meat_and_fish_processing'].div(static_df_num[idbr_norm]))*(fctr)
static_df_num['textiles']=(static_df_num['textiles'].div(static_df_num[idbr_norm]))*(fctr)
static_df_num['ready_meals']=(static_df_num['ready_meals'].div(static_df_num[idbr_norm]))*(fctr)
static_df_num['warehousing']=(static_df_num['warehousing'].div(static_df_num[idbr_norm]))*(fctr)

static_df_num.drop(columns=['ALL_PEOPLE','Area'],inplace=True)

static_df_num.set_index(['LSOA11CD','travel_cluster'],inplace=True)

assert static_df_num.shape == (cf.lsoa_count, 10)

######### IMPLEMENT THE RESULTS OF THE EXPLORATORY FACTOR ANALYSIS ############


'''Based on Factors obtained earlier 
we combine features as follows'''

risk_ftrs=['meat_and_fish_processing', 'textiles', 'ready_meals','warehousing','care']

# features except high risk industry features
sep_ftrs=[x for x in static_df_num.columns if x not in risk_ftrs]
 

df_sep=static_df_num[sep_ftrs].reset_index()

df_risk=static_df_num[risk_ftrs].reset_index()


'''df_risk_sum captures 'interactions'
between IDBR features
'''

df_risk_sum=df_risk.select_dtypes(include=object)

### CREATE THE NEW COLUMNS

df_risk_sum.loc[:,'care_homes_warehousing_textiles']=df_risk[['LSOA11CD', 'travel_cluster','textiles', 'warehousing', 'care']].sum(axis=1)

df_risk_sum.loc[:,'meat_and_fish_processing']=df_risk[['LSOA11CD', 'travel_cluster','meat_and_fish_processing']].sum(axis=1)

df_risk_sum.loc[:,'ready_meals']=df_risk[['LSOA11CD', 'travel_cluster','ready_meals']].sum(axis=1)

### MERGE THE SEPARATE FEATURES WITH THE NEW RISK FEATURES

list_dfs=[df_sep,df_risk_sum]

static_df_new_variables = reduce(lambda left,right: pd.merge(left,right,on=['LSOA11CD','travel_cluster']), list_dfs)

# This is the final static dataframe - still neeed to add cases and dynamic data
static_df_new_variables=static_df_new_variables.merge(area_df,how='inner',on=['LSOA11CD']).reset_index(drop=True)

assert static_df_new_variables.shape == (cf.lsoa_count, 12)


#################################
#### PROCESS THE CASES DATA #####
#################################

# sort values by date
cases_df_datum=cases_df[['Date','LSOA11CD','COVID_Cases']].sort_values(by='Date').reset_index(drop=True)
    
# create a list of dataframes of cases, one df for each week 
cases_df_datum=[pd.DataFrame(y) for x, y in cases_df_datum.groupby('Date', as_index=False)]
    
cases_df_datum_mrgd=[]

# The cases dataframe is split into different dates.
# This splitting allows for each dataframe to be left joined to the static data
# Therefore there will be a cases value for every LSOA for every week
# If no cases data is present for a given week in a given LSOA, the 'Date'
# field is filled with the 'Date' value from that DataFrame


# for each df
for splt_df in cases_df_datum:
    
    # store the date for the given DataFrame
    datm=splt_df['Date'].unique()[0]
    
    # left-join cases onto the static data
    df=static_df_new_variables.merge(splt_df,how='left',on=['LSOA11CD'])
    
    # fill any gaps in the cases data with the correct date
    df['Date']=df['Date'].fillna(datm)
    
    # any dates that needed to be filled had zero cases for that week
    df['COVID_Cases']=df['COVID_Cases'].fillna(0)
    
    # apply normalisation
    df['COVID_Cases']=df['COVID_Cases'].div(df['Area'])
    
    cases_df_datum_mrgd.append(df)
        
# stack the dataframes        
df_all_tranches_sbset=pd.concat(cases_df_datum_mrgd).reset_index(drop=True)

# drop the area column
df_all_tranches_sbset.drop('Area', axis=1, inplace=True)

# rename to reflect normalisation
df_all_tranches_sbset.rename(columns={'COVID_Cases':'COVID_Cases_per_unit_area'},inplace=True)

assert df_all_tranches_sbset.shape == (2627520, 13)



##########################################
#### CHOOSE THE MODEL TRAINING PERIOD ####
##########################################


# Change this filter if different time period 
# is required to infer respective risk predictors

df_all_tranches_sbset=df_all_tranches_sbset[df_all_tranches_sbset['Date']>=strt_training_period]

#### GENERATE A 'WEEK' COLUMN ###

date_list=sorted(df_all_tranches_sbset['Date'].dt.date.unique())
    
week_list=["week_"+str(x+1) for x in range(len(date_list))]
    
date_dict=dict(zip(date_list,week_list))
    
df_all_tranches_sbset['week']=df_all_tranches_sbset['Date'].map(date_dict)

df_all_tranches_sbset['Date']=df_all_tranches_sbset['Date'].astype(str)

df_all_tranches_sbset=df_all_tranches_sbset.reset_index(drop=True)

# This is to visually check we have same number of LSOAs in consecutive weeks of training data
print('Unique LSOAs in various training weeks {}'.format(df_all_tranches_sbset.groupby('Date')['LSOA11CD'].count().unique()))

assert df_all_tranches_sbset.groupby('Date')['LSOA11CD'].count().unique() == cf.lsoa_count


##############################
### PROCESS MOBILITY DATA ####
##############################

deimos_footfall_df['Date']=deimos_footfall_df['Date'].astype(str)

# Issue: December-2021: CJ
# Since the delay in cases data available to us is longer than the 
# delay in regular ingestion of mobility data- we use the 'excess'
# mobility data alongside the static predictors to predict the number
# of cases--This is one way to validate the results

# Dataset containing both the static and dynamic predictors alongside the target variable
df_all_tranches_sbset=df_all_tranches_sbset.merge(deimos_footfall_df,how='inner',on=['LSOA11CD','Date'])

# This is to visually check we have same number of LSOAs in consecutive weeks of training data (including both static and dynamic predictors)
# We should not lose any LSOAS: this value should be 32844
assert df_all_tranches_sbset.groupby('Date')['LSOA11CD'].count().unique() == cf.lsoa_count


#### CREATE TEST SET #####

# These are the dates for which we have mobility data but we don't have cases data

# Test (unseen data) for predicting future cases: we only capture the predictors in this dataset
df_all_tranches_sbset_tst_data=deimos_footfall_df.merge(static_df_new_variables,how='inner',on=['LSOA11CD'])

# Test dataset should contain timestamps not present in the training data
df_all_tranches_sbset_tst_data=df_all_tranches_sbset_tst_data[df_all_tranches_sbset_tst_data['Date']>df_all_tranches_sbset['Date'].max()].reset_index(drop=True)

assert df_all_tranches_sbset_tst_data.groupby('Date')['LSOA11CD'].count().unique() == cf.lsoa_count

#Test data date range
tst_dat_rng=df_all_tranches_sbset_tst_data['Date'].min()+'-'+df_all_tranches_sbset_tst_data['Date'].max()


########################################
#### PROCESS THE DATA INTO TRANCHES ####
########################################

tranches_uk = cf.tranches_uk
events = cf.events

splt_df_tranches=[]

for tim_slice in range(len(tranches_uk)):
    print(tim_slice)
    
    # if tim_slice is the final element of the list
    if tim_slice == len(tranches_uk)-1:
        
        # t1 is the selected date
        t1 = tranches_uk[tim_slice]
        
        # subset for all dates after t1
        df_tim = df_all_tranches_sbset[df_all_tranches_sbset['Date']>t1]
        
        # if dataframe is not empty
        if df_tim.shape[0] != 0:
            
            # add a column for the event description
            df_tim['tranche_desc']=events[tim_slice]
        
        splt_df_tranches.append(df_tim)
    
    # if tim_slice is not the final element of the list
    else:
        
        # t1 is the selected date
        t1 = tranches_uk[tim_slice]
        
        # t2 is the next date in the list
        t2 = tranches_uk[tim_slice+1]
        
        # return the data between t1 and t2
        df_tim=md.split_time_slice(df_all_tranches_sbset,t1,t2)
        
        # if dataframe is not empty
        if df_tim.shape[0] != 0:
            
            # add a column for the event description
            df_tim['tranche_desc'] = events[tim_slice]
        
        splt_df_tranches.append(df_tim)
        
        

# remove sliced df for which there is no data available 
# (mobilty data is available from tranche 2 onwards, so the first sliced df will be empty)
splt_df_tranches=[x for x in splt_df_tranches if len(x)!=0]


# Perform aggregation of predictors and target variable for each tranche
# Each tranche contains multiple weeks, aggregation results in mean of each of the numerical features
# In practice, the static features are the same for each week, so we are averaging footfall over the tranche
# Each sliced df will have one unique record for each LSOA (because of averaging)
splt_df_tranches_agg=[]

for df_x in splt_df_tranches:
    
    # convert date column to string showing date range of the tranche
    df_x['Date']=str(df_x['Date'].min())+'-'+str(df_x['Date'].max())
    
    # define columns to group by
    grpp_colms=['Date','LSOA11CD','tranche_desc','travel_cluster']
    
    # compute the mean over each week in the tranche
    df_x=df_x.groupby(grpp_colms)[[x for x in df_x.columns if x not in grpp_colms]].mean().reset_index()
    
    # sort by LSOA code
    df_x=df_x.sort_values(by='LSOA11CD').reset_index(drop=True)
    
    splt_df_tranches_agg.append(df_x)
    

# stack each tranche into one dataframe
splt_df_tranches_conct=pd.concat(splt_df_tranches_agg).reset_index(drop=True)

# convert date to string
splt_df_tranches_conct['Date']=splt_df_tranches_conct['Date'].astype(str)

# find unique date range and tranche description combinations
df_key=splt_df_tranches_conct[['Date','tranche_desc']].drop_duplicates().reset_index(drop=True)

# put them into a dictionary
event_dict=dict(zip(df_key['Date'].values,df_key['tranche_desc'].values))

# list of integers from 1 to n_tranches
tranche_order=list(range(1, cf.n_tranches))

# zip the tranche descriptions and tranche numbers
event_order_dict=dict(zip(events,tranche_order))

# dict of tranche number: tranche description
rvse_event_dict={v: k for k, v in event_order_dict.items()}

# dict of tranche description: tranche date range
rvse_date_dict={v: k for k, v in event_dict.items()}

# the dataframe with all tranches
df_all_tranches_sbset=splt_df_tranches_conct

# create new column for tranche number
df_all_tranches_sbset['tranche_order']=df_all_tranches_sbset['tranche_desc'].map(event_order_dict)

# test that each LSOA appears one for each tranche
# there is no mobility data for the first tranch, therefore the shape should be (n_tranches - 1) * n_lsoa
assert df_all_tranches_sbset.shape == (((cf.n_tranches - 1) * cf.lsoa_count), 21)

df_all_tranches_sbset.to_gbq('wip.test_alltranches',project_id='ons-hotspot-prod',if_exists='replace')