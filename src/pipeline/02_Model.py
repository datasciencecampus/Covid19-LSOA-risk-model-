## LB notes (temp):
# risk_weekly_static
# risk_weekly_dynamic

## to do:
# organise and tidy import list
# remove exploratory/plotting bits from code (currently this is just copy pasted)
# remove code where data already in variables is read in from GCP
# arrange code into functions to allow for easy testing

###########

# Modelling
# Sections: Model COVID cases based on static variables, and then model the residuals using the dynamic variables

# SECTION ONE OF TWO
# Model COVID cases using static variables (LSOA attributes which do not change)

# Import Packages
import os
import sys
from datetime import date
from datetime import datetime
import random
from random import randint
import math
from functools import reduce

from google.cloud import bigquery

import pandas as pd
import numpy as np
import pandas_gbq
from numpy.random import seed
from numpy.random import randn

import matplotlib.pyplot as plt, figure
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import plotly.offline as py
py.init_notebook_mode()
import plotly.graph_objs as go
import plotly.express as px
import pgeocode
import dash
from plotly.offline import iplot, init_notebook_mode

from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression

from sklego.meta import ZeroInflatedRegressor
from sklego.meta import EstimatorTransformer

from sklearn.metrics import median_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import qqplot


# Import from local data files
current_path = os.path.abspath('.')
sys.path.append(os.path.dirname(current_path))

from data_access.data_factory import DataFactory as factory
from utils import model as md
from utils import config as cf

# SECTION ONE OF TWO
# Model static variables

## Model parameters
## Number of different combinations of grid search hyperparameters
## Default is 500, use a lower value, >=1 to speed-up the evaluations
parm_spce_grid_srch=cf.parm_spce_grid_srch

# Create a list of alphas to cross-validate against
alphas_val = cf.alphas_val

# combine static and dynamic features for modelling

df_all_tranches_sbset = factory.get('all_tranches_dynamic_static').create_dataframe()

df_week_datum=df_all_tranches_sbset[['week','Date']].drop_duplicates().reset_index(drop=True)

date_dict=df_week_datum.set_index('week').to_dict()['Date']

# Visualisation of dynamic features in the dataset, aggregated by travel cluster

which_col='COVID_Cases_per_unit_area'

md.plot_var_by_travel_cluster(df_all_tranches_sbset,which_col,'Date','sum')

# This is to check we have same number of LSOAs in consecutive weeks
# TA: move this to a test!

df_all_tranches_sbset.groupby('Date')['LSOA11CD'].count()

# **FINDING STATIC RISK FACTOR/PREDICTORS FOR EACH TRAVEL CLUSTER**

# Finding the most important features for each travel cluster which 
# are able to predict the confirmed positive cases (target variable).

# **MODEL FIT FOR EACH TRAVEL CLUSTER**

# WE TRY TO FIND MOST IMPORTANT PREDICTORS WHILE TRAINING THE MODEL ON THE FULL TIME PERIOD

# TO ACHIEVE THIS, WE DO THE FOLLOWING:

# (a) TRAIN TRAGET VARIABLE ON STATIC VARIABLES ON A WEEKLY BASIS AND FIND RISK ESTIMATES FOR THE SIGNIFICANT PREDICTORS

# (b) WE THEN MODEL **THE RESIDUALS (AND NOT THE CHANGE IN RESIDUALS ?)** OBTAINED FROM (a) ON THE  CHANGE OF DYNAMIC FEATURES (FROM PREVIOUS WEEK) AND CUMULATIVE CASES (PER UNIT AREA) AND CUMULATIVE VACCINATED LSOA POPULATION FROM THE PREVIOUS WEEK

# weekly training and predictions

# Number of past weeks used in training the model
grp_var=1
zero_inf_flg_st=cf.zero_infltd_modl
dynamic_lagged_variables_df = factory.get('dynamic_time_lagged').create_dataframe()
dynamic_features=list(dynamic_lagged_variables_df.select_dtypes(include='number').columns)

list_of_tc=sorted(df_all_tranches_sbset['travel_cluster'].unique())
# Separate regression model is fit
# for each travel cluster
# This is done to further minimise
# any spatial correlation in 
# the data
str_pred_tc_static=[]
str_coef_tc_static=[]
for sbset_tc in list_of_tc:
    df_chsen=df_all_tranches_sbset[df_all_tranches_sbset['travel_cluster']==sbset_tc].reset_index(drop=True)
    df_chsen['week_number']=df_chsen['week'].str.strip('week_').astype(int)
    df_chsen=df_chsen.sort_values(by=['week_number','LSOA11CD']).reset_index(drop=True)
    df_chsen=df_chsen[[x for x in df_chsen.columns if x not in ['Date','week_number','Month']]]
    #print(df_chsen)
    pred_tc,coef_tc=md.fit_model_one_week_static(df_chsen,grp_var,zero_inf_flg_st,dynamic_features,alphas_val,parm_spce_grid_srch)
    str_pred_tc_static.append(pred_tc)
    str_coef_tc_static.append(coef_tc)

# Store most important features used
# for making predictions for each travel cluster
# and for each training period
str_coef_tc_static=pd.concat(str_coef_tc_static).reset_index()
str_coef_tc_static.rename(columns={'index':'Features'},inplace=True)

# Store the predictions of trained model
str_pred_tc_static=pd.concat(str_pred_tc_static).reset_index(drop=True)

str_pred_tc_static['Date']=str_pred_tc_static['week_train'].map(date_dict)

# Confidence interval calculation

str_coef_tc_static_ci = str_coef_tc_static.groupby(['Features','travel_cluster'])['Coefficients'].agg(['mean', 'count', 'std'])
ci95 = []

for i in str_coef_tc_static_ci.index:
    m, c, s = str_coef_tc_static_ci.loc[i]
    ci95.append(1.96*s/math.sqrt(c))

str_coef_tc_static_ci['ci95'] = ci95

    
str_coef_tc_static_ci=str_coef_tc_static_ci.reset_index().sort_values(by='mean',ascending=False).reset_index(drop=True)
str_coef_tc_static_ci['Features']=str_coef_tc_static_ci['Features'].str.lower()
str_coef_tc_static_ci.rename(columns={'mean':'Coefficients'},inplace=True)

# plot static risk predictors for each travel cluster

str_coef_tc_static=str_coef_tc_static.sort_values(by='Coefficients',ascending=False).reset_index(drop=True)

sns.pointplot('Coefficients','Features', hue='travel_cluster',
    data=str_coef_tc_static, dodge=True, join=False,hue_order=sorted(str_coef_tc_static.travel_cluster.unique()))
sns.set(rc={'figure.figsize':(20,10)})


str_coef_tc_static.groupby('Features')['Coefficients'].mean().reset_index().sort_values(by='Coefficients').set_index('Features').plot(kind='barh')

dataset_suffix = cf.model_suffixes['static_main']
    
str_coef_tc_static.to_gbq(cf.risk_coef + dataset_suffix, project_id=project_name, if_exists='replace')
str_coef_tc_static_ci.to_gbq(cf.risk_coef_ci + dataset_suffix, project_id=project_name, if_exists='replace')
str_pred_tc_static.to_gbq(cf.risk_pred + dataset_suffix, project_id=project_name, if_exists='replace')

# Model's performance
md.plot_var_by_travel_cluster(str_pred_tc_static,'Best_cv_score_train','Date','mean')

# SECTION TWO OF TWO
# Model residuals of previous model with dynamic variables

# find most important dynamic features for each travel cluster which are able to predict the residuals generated by the static variables

df_dynamic_changes_weekly_with_trgt = factory.get('dynamic_changes_weekly').create_dataframe()

static_rgn_df = factory.get('static_vars_rgns').create_dataframe()

df_dynamic_changes_weekly_with_trgt=df_dynamic_changes_weekly_with_trgt.merge(static_rgn_df[['LSOA11CD','RGN19NM']],on=['LSOA11CD'],how='inner').reset_index(drop=True)

df_dynamic_changes_weekly_with_trgt['Country']='England'

df_dynamic_changes_weekly_with_trgt.rename(columns={'week_train':'week'},inplace=True)

# weekly training and predictions



# This can also be chosen as 'travel_cluster','country'
which_clustrng=cf.granularity_for_modelling

list_of_tc=sorted(df_dynamic_changes_weekly_with_trgt[which_clustrng].unique())
# Separate regression model is fit
# for each travel cluster
# This is done to further minimise
# any spatial correlation in 
# the data
str_pred_tc_dynamic=[]
str_coef_tc_dynamic=[]
str_se_coef_tc_dynamic=[]
for sbset_tc in list_of_tc:
    print(sbset_tc)
    print('+'*100)
    df_chsen=df_dynamic_changes_weekly_with_trgt[df_dynamic_changes_weekly_with_trgt[which_clustrng]==sbset_tc].reset_index(drop=True)
    df_chsen['week_number']=df_chsen['week'].str.strip('week_').astype(int)
    df_chsen=df_chsen.sort_values(by=['week_number','LSOA11CD']).reset_index(drop=True)
    # Dynamic features can be dropped from this list 
    # if dynamic training is required on ceratin features only
    # below, we have dropped 'commute_inflow_sqkm', 'other_inflow_sqkm'
    # 'visitor_footfall_sqkm'from dynamic training
    df_chsen=df_chsen[[x for x in df_chsen.columns if x not in ['Date','week_number','lsoa_inflow_volume',
                                                                'total_vaccinated_second_dose_prop_population_cumsum',
                                                                'COVID_Cases_per_unit_area_cumsum','commute_inflow_sqkm', 'other_inflow_sqkm']]]
    pred_tc,coef_tc,se_coef_tc=md.fit_model_one_week_dynamic(df_chsen,grp_var,which_clustrng,alphas_val,parm_spce_grid_srch)
    str_pred_tc_dynamic.append(pred_tc)
    str_coef_tc_dynamic.append(coef_tc)
    str_se_coef_tc_dynamic.append(se_coef_tc)

# Store most important features used
# for making predictions for each travel cluster
# and for each training period
str_coef_tc_dynamic=pd.concat(str_coef_tc_dynamic).reset_index()
str_coef_tc_dynamic.rename(columns={'index':'Features'},inplace=True)

# Store the predictions of trained model
str_pred_tc_dynamic=pd.concat(str_pred_tc_dynamic).reset_index(drop=True)

# Store standard error for coefficients of trained model
str_se_coef_tc_dynamic=pd.concat(str_se_coef_tc_dynamic).reset_index(drop=True)

#Merge dynamic risk predictors with CI (based on SE for each week)
str_coef_tc_dynamic=str_coef_tc_dynamic.merge(str_se_coef_tc_dynamic,on=list(str_coef_tc_dynamic.columns & str_se_coef_tc_dynamic.columns),how='inner')

str_coef_tc_dynamic['ci_95']=1.96*str_coef_tc_dynamic['Standard_error']

# DYNAMIC RISK PREDICTORS CONFIDENCE INTERVAL CALCULATION FOR EACH TRAVEL CLUSTER WHERE THE CONFIDENCE INTERVAL IS CALCULATED BASED ON THE WEEKS OF TRAINING

str_coef_tc_dynamic_ci = str_coef_tc_dynamic.groupby(['Features',which_clustrng])['Coefficients'].agg(['mean', 'count', 'std'])
ci95 = []

for i in str_coef_tc_dynamic_ci.index:
    m, c, s = str_coef_tc_dynamic_ci.loc[i]
    ci95.append(1.96*s/math.sqrt(c))

str_coef_tc_dynamic_ci['ci95'] = ci95

    
str_coef_tc_dynamic_ci=str_coef_tc_dynamic_ci.reset_index().sort_values(by='mean',ascending=False).reset_index(drop=True)
str_coef_tc_dynamic_ci['Features']=str_coef_tc_dynamic_ci['Features'].str.lower()
str_coef_tc_dynamic_ci.rename(columns={'mean':'Coefficients'},inplace=True)

str_coef_tc_dynamic['week_number']=str_coef_tc_dynamic['week'].str.split('week_').apply(lambda x: int(x[1]))

# plot dynamic risk predictors

str_coef_tc_dynamic=str_coef_tc_dynamic.sort_values(by='Coefficients',ascending=False).reset_index(drop=True)

sns.pointplot('Coefficients','Features', hue=which_clustrng,
    data=str_coef_tc_dynamic, dodge=True, join=False,ci='sd',hue_order =sorted(str_coef_tc_dynamic[which_clustrng].unique()))
sns.set(rc={'figure.figsize':(20,10)})
#plt.xlim(-1,1)

str_coef_tc_dynamic.groupby('Features')['Coefficients'].mean().reset_index().sort_values(by='Coefficients').set_index('Features').\
plot(kind='barh')
   
dataset_suffix = cf.model_suffixes['dynamic']

str_coef_tc_dynamic.to_gbq(cf.risk_coef + dataset_suffix, project_id=project_name, if_exists='replace')
str_coef_tc_dynamic_ci.to_gbq(cf.risk_coef_ci+ dataset_suffix, project_id=project_name,if_exists='replace')
str_pred_tc_dynamic.to_gbq(cf.risk_pred+ dataset_suffix, project_id=project_name,if_exists='replace')


# model outputs - plot static and dynamic risk predictors for each travel cluster

#retrieve static risk estimators
str_coef_tc_static = factory.get('static_changes_weekly').create_dataframe()

#retrieve static risk estimators ci
str_coef_tc_static_ci = factory.get('static_changes_weekly_ci').create_dataframe()
    
    
    
risk_predictors_df=pd.concat([str_coef_tc_static,str_coef_tc_dynamic])

risk_predictors_df=risk_predictors_df.sort_values(by='Coefficients',ascending=False).reset_index(drop=True)

sns.pointplot('Coefficients','Features', hue='travel_cluster', data=risk_predictors_df, dodge=True, join=False)
sns.set(rc={'figure.figsize':(20,10)})

risk_predictors_df_ci=pd.concat([str_coef_tc_static_ci,str_coef_tc_dynamic_ci]).reset_index(drop=True)

risk_predictors_df_ci=risk_predictors_df_ci[['Features','travel_cluster','Coefficients','std','ci95']]

dataset_suffix = cf.model_suffixes['static_dynamic']    

risk_predictors_df.to_gbq(cf.risk_coef+dataset_suffix, project_id=project_name, if_exists='replace')

risk_predictors_df_ci.to_gbq(cf.risk_coef_ci+dataset_suffix, project_id=project_name, if_exists='replace')   




