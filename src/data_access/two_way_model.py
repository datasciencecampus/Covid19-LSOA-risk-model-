## LB notes (temp):
# risk_weekly_static
# risk_weekly_dynamic

## to do:
# import list cull

###########

# Modelling
# Sections: Model COVID cases based on static variables, and then model the residuals using the dynamic variables

# SECTION ONE OF TWO
# Model COVID cases using static variables (LSOA attributes which do not change)

# Import Packages
import os
import sys
import math

from google.cloud import bigquery

import pandas as pd
import numpy as np
import pandas_gbq

# Import from local data files
current_path = os.path.abspath('.')
sys.path.append(os.path.dirname(current_path))

from data_access.data_factory import DataFactory as factory
from utils import model as md
from utils import config as cf


def static_model(alphas_val = cf.alphas_val, 
                 parm_spce_grid_srch = cf.parm_spce_grid_srch,
                 grp_var = 1,
                 zero_inf_flg_st = cf.zero_infltd_modl,
                 save_results = True):
    
    """
    First step of the two way fixed effects model uses static variables only to predict cases. Function writes output to GCP tables at the end, and outputs dataframes to be picked up by the dynamic_model function.
    
    :param alphas_val: List of alphas for cross-validation. Default is np.logspace(-3, 3, 101), set in config file.
    :type alphas_val: Numpy array
    
    :param parm_spce_grid_srch: Number of different combinations to use in grid search for hyperparameters. Default is 500, set in config file.
    :type parm_spce_grid_srch: int
    
    :param grp_var: Number of prior weeks to train the model. Defaults to 1.
    :type grp_var: int
    
    :param zero_inf_flg_st: Whether to use zero-inflated regressor model. Default to False, set in config.
    :type zero_inf_flg_st: bool
    
    :param save_results: Flag for whether to output results to tables or not. Default True.
    :type: bool
    
    :return: Two dataframes, one each of the coefficients and confidence intervals of models generated.
    :rtype: Pandas DataFrames
    """

    # combine static and dynamic features for modelling

    df_all_tranches_sbset = factory.get('all_tranches_dynamic_static').create_dataframe()

    df_week_datum=df_all_tranches_sbset[['week','Date']].drop_duplicates().reset_index(drop=True)

    date_dict=df_week_datum.set_index('week').to_dict()['Date']

    # to do: add test to check there are the same number of LSOAs in consecutive weeks

    # **FINDING STATIC RISK FACTOR/PREDICTORS FOR EACH TRAVEL CLUSTER**

    # Finding the most important features for each travel cluster which 
    # are able to predict the confirmed positive cases (target variable).

    # **MODEL FIT FOR EACH TRAVEL CLUSTER**

    # WE TRY TO FIND MOST IMPORTANT PREDICTORS WHILE TRAINING THE MODEL ON THE FULL TIME PERIOD

    # TO ACHIEVE THIS, WE DO THE FOLLOWING:

    # (a) TRAIN TRAGET VARIABLE ON STATIC VARIABLES ON A WEEKLY BASIS AND FIND RISK ESTIMATES FOR THE SIGNIFICANT PREDICTORS

    # (b) WE THEN MODEL **THE RESIDUALS (AND NOT THE CHANGE IN RESIDUALS ?)** OBTAINED FROM (a) ON THE  CHANGE OF DYNAMIC FEATURES (FROM PREVIOUS WEEK) AND CUMULATIVE CASES (PER UNIT AREA) AND CUMULATIVE VACCINATED LSOA POPULATION FROM THE PREVIOUS WEEK

    # weekly training and predictions

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

    dataset_suffix = cf.model_suffixes['static_main']
    
    if save_results:

        str_coef_tc_static.to_gbq(cf.risk_coef + dataset_suffix, project_id=cf.project_name, if_exists='replace')
        str_coef_tc_static_ci.to_gbq(cf.risk_coef_ci + dataset_suffix, project_id=cf.project_name, if_exists='replace')
        str_pred_tc_static.to_gbq(cf.risk_pred + dataset_suffix, project_id=cf.project_name, if_exists='replace')

    return str_coef_tc_static, str_coef_tc_static_ci

# SECTION TWO OF TWO
# Model residuals of previous model with dynamic variables

# find most important dynamic features for each travel cluster which are able to predict the residuals generated by the static variables

def dynamic_model(str_coef_tc_static,
                 str_coef_tc_static_ci,
                 alphas_val = cf.alphas_val, 
                 parm_spce_grid_srch = cf.parm_spce_grid_srch,
                 grp_var = 1,
                 which_clustrng = cf.granularity_for_modelling,
                 save_results = True):
    
    """
    Second step of the two way fixed effects model uses dynamic variables to predict on the residuals output by the first step of the model. The results are written to a GCP table at the end of the function.
    
    :param str_coef_tc_static: Output of static_model function - coefficients.
    :type str_coef_tc_static: Pandas DataFrame
    
    :param str_coef_tc_static_ci: Output of static_model function - 95% confidence intervals. 
    :type str_coef_tc_static_ci: Pandas DataFrame
    
    :param alphas_val: List of alphas for cross-validation. Default is np.logspace(-3, 3, 101), set in config file.
    :type alphas_val: Numpy array
    
    :param parm_spce_grid_srch: Number of different combinations to use in grid search for hyperparameters. Default is 500, set in config file.
    :type parm_spce_grid_srch: int
    
    :param grp_var: Number of prior weeks to train the model. Defaults to 1.
    :type grp_var: int
    
    :param which_clustrng: Chosen geography granularity for modelling. Must be a column in the dataset. Defaults to 'Country', set in config file.
    :type which_clustrng: string
    
    :param save_results: Flag for whether to output results to tables or not. Default True.
    :type: bool

    :return: Two dataframes, one each of the coefficients and confidence intervals of models generated.
    :rtype: Pandas DataFrames
    """
    
    # This function reads in the results of static_model and performs some pre-processing 
    df_dynamic_changes_weekly_with_trgt = factory.get('dynamic_changes_weekly').create_dataframe()

    static_rgn_df = factory.get('static_vars_rgns').create_dataframe()

    df_dynamic_changes_weekly_with_trgt=df_dynamic_changes_weekly_with_trgt.merge(static_rgn_df[['LSOA11CD','RGN19NM']],on=['LSOA11CD'],how='inner').reset_index(drop=True)

    df_dynamic_changes_weekly_with_trgt['Country']='England'

    df_dynamic_changes_weekly_with_trgt.rename(columns={'week_train':'week'},inplace=True)

    # weekly training and predictions

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

    str_coef_tc_dynamic=str_coef_tc_dynamic.sort_values(by='Coefficients',ascending=False).reset_index(drop=True)

    str_coef_tc_dynamic.groupby('Features')['Coefficients'].mean().reset_index().sort_values(by='Coefficients').set_index('Features').\
    plot(kind='barh')

    dataset_suffix = cf.model_suffixes['dynamic']
    
    if save_results:

        str_coef_tc_dynamic.to_gbq(cf.risk_coef + dataset_suffix, project_id=cf.project_name, if_exists='replace')
        str_coef_tc_dynamic_ci.to_gbq(cf.risk_coef_ci+ dataset_suffix, project_id=cf.project_name,if_exists='replace')
        str_pred_tc_dynamic.to_gbq(cf.risk_pred+ dataset_suffix, project_id=cf.project_name,if_exists='replace')

    risk_predictors_df=pd.concat([str_coef_tc_static,str_coef_tc_dynamic])

    risk_predictors_df=risk_predictors_df.sort_values(by='Coefficients',ascending=False).reset_index(drop=True)

    risk_predictors_df_ci=pd.concat([str_coef_tc_static_ci,str_coef_tc_dynamic_ci]).reset_index(drop=True)

    risk_predictors_df_ci=risk_predictors_df_ci[['Features','travel_cluster','Coefficients','std','ci95']]

    dataset_suffix = cf.model_suffixes['static_dynamic']
    
    if save_results:

        risk_predictors_df.to_gbq(cf.risk_coef+dataset_suffix, project_id=cf.project_name, if_exists='replace')
        risk_predictors_df_ci.to_gbq(cf.risk_coef_ci+dataset_suffix, project_id=cf.project_name, if_exists='replace')

    return str_coef_tc_dynamic, str_coef_tc_dynamic_ci




