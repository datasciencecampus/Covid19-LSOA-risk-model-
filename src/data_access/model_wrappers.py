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
                 param_search_space = cf.param_search_space,
                 train_weeks = 1,
                 zero_inf_flg_st = cf.zero_infltd_modl,
                 save_results = True):
    
    """
    First step of the two way fixed effects model uses static variables only to predict cases. Function writes output to GCP tables at the end, and outputs dataframes to be picked up by the dynamic_model function.
    
    The data is grouped by the travel_cluster column (to minimise spatial correlation in the data) and a model is created for each time period, as specified in the train_weeks parameter. This defaults to 1, so a model will be trained for each week, using the previous week as training data. Then in the dynamic_model function, the target variable is the week-on-week change in residuals from the static model, and the explanatory variables are the week-on-week changes in the dynamic features.
    
    :param alphas_val: List of alphas for cross-validation. Default is np.logspace(-3, 3, 101), set in config file.
    :type alphas_val: Numpy array
    
    :param param_search_space: Number of different combinations to use in randomised search for hyperparameters. Default is 500, set in config file.
    :type param_search_space: int
    
    :param train_weeks: Number of prior weeks to train the model. Defaults to 1.
    :type train_weeks: int
    
    :param zero_inf_flg_st: Whether to use zero-inflated regressor model. Default to False, set in config.
    :type zero_inf_flg_st: bool
    
    :param save_results: Flag for whether to output results to tables or not. Default True.
    :type: bool
    
    :return: Two dataframes, one each of the coefficients and confidence intervals of models generated.
    :rtype: Pandas DataFrames
    """

    # Combine static and dynamic features for modelling

    df_all_tranches_sbset = factory.get('all_tranches_dynamic_static').create_dataframe()

    df_week_datum = df_all_tranches_sbset[['week', 'Date']].drop_duplicates().reset_index(drop=True)

    date_dict = df_week_datum.set_index('week').to_dict()['Date']

    # Find static risk factors/predictors for each travel cluster
    # Weekly training and predictions

    dynamic_lagged_variables_df = factory.get('dynamic_time_lagged').create_dataframe()
    dynamic_features = list(dynamic_lagged_variables_df.select_dtypes(include='number').columns)
    
    list_of_tc = sorted(df_all_tranches_sbset['travel_cluster'].unique())
    
    # Fit models
    str_pred_tc_static = []
    str_coef_tc_static = []
    
    for sbset_tc in list_of_tc:
        df_chsen = df_all_tranches_sbset[df_all_tranches_sbset['travel_cluster'] == sbset_tc].reset_index(drop=True)
        df_chsen['week_number'] = df_chsen['week'].str.strip('week_').astype(int)
        df_chsen = df_chsen.sort_values(by=['week_number', 'LSOA11CD']).reset_index(drop=True)
        df_chsen = df_chsen[[x for x in df_chsen.columns if x not in ['Date', 'week_number', 'Month']]]

        pred_tc,coef_tc = md.fit_model_one_week_static(df_chsen, train_weeks, zero_inf_flg_st, dynamic_features, alphas_val, param_search_space)
        
        str_pred_tc_static.append(pred_tc)
        str_coef_tc_static.append(coef_tc)

    str_coef_tc_static = pd.concat(str_coef_tc_static).reset_index()
    str_coef_tc_static.rename(columns={'index':'Features'}, inplace=True)

    str_pred_tc_static = pd.concat(str_pred_tc_static).reset_index(drop=True)

    str_pred_tc_static['Date'] = str_pred_tc_static['week_train'].map(date_dict)

    # Confidence interval calculation
    str_coef_tc_static_ci = str_coef_tc_static.groupby(['Features', 'travel_cluster'])['Coefficients'].agg(['mean', 'count', 'std'])
    ci95 = []

    for i in str_coef_tc_static_ci.index:
        m, c, s = str_coef_tc_static_ci.loc[i]
        ci95.append(1.96 * s / math.sqrt(c))

    str_coef_tc_static_ci['ci95'] = ci95

    str_coef_tc_static_ci = str_coef_tc_static_ci.reset_index().sort_values(by='mean', ascending=False).reset_index(drop=True)
    str_coef_tc_static_ci['Features'] = str_coef_tc_static_ci['Features'].str.lower()
    str_coef_tc_static_ci.rename(columns={'mean':'Coefficients'}, inplace=True)

    if save_results:
        
        dataset_suffix = cf.model_suffixes['static_main']

        str_coef_tc_static.to_gbq(cf.risk_coef + dataset_suffix, project_id=cf.project_name, if_exists='replace')
        str_coef_tc_static_ci.to_gbq(cf.risk_coef_ci + dataset_suffix, project_id=cf.project_name, if_exists='replace')
        str_pred_tc_static.to_gbq(cf.risk_pred + dataset_suffix, project_id=cf.project_name, if_exists='replace')

    return str_coef_tc_static, str_coef_tc_static_ci

def dynamic_model(str_coef_tc_static,
                 str_coef_tc_static_ci,
                 alphas_val = cf.alphas_val, 
                 param_search_space = cf.param_search_space,
                 train_weeks = 1,
                 which_clustrng = cf.granularity_for_modelling,
                 save_results = True):
    
    """
    Second step of the two way fixed effects model uses week-on-week changes in dynamic variables to predict the week-on-week changes in residuals output by the first step of the model. The results are written to a GCP table at the end of the function.
    
    :param str_coef_tc_static: Output of static_model function - coefficients.
    :type str_coef_tc_static: Pandas DataFrame
    
    :param str_coef_tc_static_ci: Output of static_model function - 95% confidence intervals. 
    :type str_coef_tc_static_ci: Pandas DataFrame
    
    :param alphas_val: List of alphas for cross-validation. Default is np.logspace(-3, 3, 101), set in config file.
    :type alphas_val: Numpy array
    
    :param param_search_space: Number of different combinations to use in grid search for hyperparameters. Default is 500, set in config file.
    :type param_search_space: int
    
    :param train_weeks: Number of prior weeks to train the model. Defaults to 1.
    :type train_weeks: int
    
    :param which_clustrng: Chosen geography granularity for modelling. Must be a column in the dataset. Defaults to 'Country', set in config file.
    :type which_clustrng: string
    
    :param save_results: Flag for whether to output results to tables or not. Default True.
    :type: bool

    :return: One dataframe with the model coefficients, and one dataframe with the confidence intervals.
    :rtype: Pandas DataFrames
    """
    
    # This function reads in the results of static_model and performs some pre-processing 
    df_dynamic_changes_weekly_with_trgt = factory.get('dynamic_changes_weekly').create_dataframe()

    static_rgn_df = factory.get('static_vars_rgns').create_dataframe()

    df_dynamic_changes_weekly_with_trgt = df_dynamic_changes_weekly_with_trgt.merge(static_rgn_df[['LSOA11CD','RGN19NM']], on=['LSOA11CD'], how='inner').reset_index(drop=True)

    df_dynamic_changes_weekly_with_trgt['Country'] = 'England'

    df_dynamic_changes_weekly_with_trgt.rename(columns={'week_train':'week'}, inplace=True)

    # weekly training and predictions

    list_of_tc = sorted(df_dynamic_changes_weekly_with_trgt[which_clustrng].unique())
    
    str_pred_tc_dynamic = []
    str_coef_tc_dynamic = []
    str_se_coef_tc_dynamic = []
    
    for sbset_tc in list_of_tc:
        print(sbset_tc)
        print('+'*100)
        df_chsen = df_dynamic_changes_weekly_with_trgt[df_dynamic_changes_weekly_with_trgt[which_clustrng] == sbset_tc].reset_index(drop=True)
        df_chsen['week_number'] = df_chsen['week'].str.strip('week_').astype(int)
        df_chsen = df_chsen.sort_values(by=['week_number', 'LSOA11CD']).reset_index(drop=True)
        
        df_chsen = df_chsen[[x for x in df_chsen.columns if x not in ['Date','week_number','lsoa_inflow_volume',
                                                                    'total_vaccinated_second_dose_prop_population_cumsum',
                                                                    'COVID_Cases_per_unit_area_cumsum','commute_inflow_sqkm', 'other_inflow_sqkm']]]
        
        pred_tc, coef_tc, se_coef_tc = md.fit_model_one_week_dynamic(df_chsen, train_weeks, which_clustrng, alphas_val, param_search_space)
        
        str_pred_tc_dynamic.append(pred_tc)
        str_coef_tc_dynamic.append(coef_tc)
        str_se_coef_tc_dynamic.append(se_coef_tc)

    str_coef_tc_dynamic = pd.concat(str_coef_tc_dynamic).reset_index()
    str_coef_tc_dynamic.rename(columns={'index':'Features'}, inplace=True)

    str_pred_tc_dynamic = pd.concat(str_pred_tc_dynamic).reset_index(drop=True)

    str_se_coef_tc_dynamic = pd.concat(str_se_coef_tc_dynamic).reset_index(drop=True)

    # Merge dynamic risk predictors with CI (based on SE for each week)
    str_coef_tc_dynamic = str_coef_tc_dynamic.merge(str_se_coef_tc_dynamic, on=list(str_coef_tc_dynamic.columns & str_se_coef_tc_dynamic.columns), how='inner')

    str_coef_tc_dynamic['ci_95'] = 1.96 * str_coef_tc_dynamic['Standard_error']

    # Calculate confidence interval of dynamic risk predictors 
    str_coef_tc_dynamic_ci = str_coef_tc_dynamic.groupby(['Features',which_clustrng])['Coefficients'].agg(['mean', 'count', 'std'])
    ci95 = []

    for i in str_coef_tc_dynamic_ci.index:
        m, c, s = str_coef_tc_dynamic_ci.loc[i]
        ci95.append(1.96*s/math.sqrt(c))

    str_coef_tc_dynamic_ci['ci95'] = ci95

    str_coef_tc_dynamic_ci = str_coef_tc_dynamic_ci.reset_index().sort_values(by='mean', ascending=False).reset_index(drop=True)
    str_coef_tc_dynamic_ci['Features'] = str_coef_tc_dynamic_ci['Features'].str.lower()
    str_coef_tc_dynamic_ci.rename(columns={'mean':'Coefficients'},inplace=True)

    str_coef_tc_dynamic['week_number'] = str_coef_tc_dynamic['week'].str.split('week_').apply(lambda x: int(x[1]))

    str_coef_tc_dynamic = str_coef_tc_dynamic.sort_values(by='Coefficients', ascending=False).reset_index(drop=True)
    
    # convert to float to allow successful concat and saving of risk_predictors_df if alpha is None
    str_coef_tc_dynamic['regularisation_alpha'] = str_coef_tc_dynamic['regularisation_alpha'].astype(np.float64)
    
    if save_results:

        dataset_suffix = cf.model_suffixes['dynamic']
        
        str_coef_tc_dynamic.to_gbq(cf.risk_coef + dataset_suffix, project_id=cf.project_name, if_exists='replace')
        str_coef_tc_dynamic_ci.to_gbq(cf.risk_coef_ci + dataset_suffix, project_id=cf.project_name, if_exists='replace')
        str_pred_tc_dynamic.to_gbq(cf.risk_pred + dataset_suffix, project_id=cf.project_name, if_exists='replace')

    risk_predictors_df = pd.concat([str_coef_tc_static, str_coef_tc_dynamic])

    risk_predictors_df = risk_predictors_df.sort_values(by='Coefficients', ascending=False).reset_index(drop=True)

    risk_predictors_df_ci = pd.concat([str_coef_tc_static_ci, str_coef_tc_dynamic_ci]).reset_index(drop=True)

    risk_predictors_df_ci = risk_predictors_df_ci[['Features', 'travel_cluster', 'Coefficients', 'std', 'ci95']]
    
    if save_results:
        
        dataset_suffix = cf.model_suffixes['static_dynamic']

        risk_predictors_df.to_gbq(cf.risk_coef+dataset_suffix, project_id=cf.project_name, if_exists='replace')
        risk_predictors_df_ci.to_gbq(cf.risk_coef_ci+dataset_suffix, project_id=cf.project_name, if_exists='replace')

    return str_coef_tc_dynamic, str_coef_tc_dynamic_ci


def tranches_model(use_regularisation = cf.use_regularisation,
                   alphas_val = cf.alphas_val, 
                   param_search_space = cf.param_search_space,
                   zero_inf_flg_st = cf.zero_infltd_modl,
                   save_results = True):
    
    """
    Loads training and test data from BigQuery and runs a linear regression model for each time tranche of
    each travel cluster. Results are output in the form of five dataframes:
    
    1) Coefficient estimates for linear regression with regularisation
    2) Standardised coefficient estimates for linear regression without regularisation
    3) Non-standardised coefficient estimates for linear regression without regularisation
    4) Predictions for each time tranche
    5) Predictions on the test data
    
    :param use_regularisation: Flag indicating which model type to train for the purpose of making predictions.
    Default is True which trains an Elastic Net model. If false, a Linear Regression model is trained.
    :type: bool
    
    :param alphas_val: List of alphas for cross-validation. Default is np.logspace(-3, 3, 101), set in config file.
    :type alphas_val: Numpy array
    
    :param param_search_space: Number of different combinations to use in grid search for optimal hyperparameters. Default is 500, set in config file.
    :type param_search_space: int
    
    :param zero_inf_flg_st: Whether to use zero-inflated regressor model. Default to False, set in config.
    :type zero_inf_flg_st: bool
    
    :param save_results: Flag for whether to output results to tables or not. Default True.
    :type: bool

    :return: Five dataframes as described earlier in this docstring
    :rtype: Pandas DataFrames
    """
    
    # import data sets from BigQuery
    df_all_tranches_sbset = factory.get('tranche_model_input').create_dataframe()
    df_all_tranches_sbset_tst_data = factory.get('tranche_model_test_data').create_dataframe()
    
    # list the travel clusters
    list_of_tc = sorted(df_all_tranches_sbset['travel_cluster'].unique())

    #Store the outputs
    str_pred_tc_static = []
    str_coef_tc_static = []
    str_se_coef_tc_static = []
    str_non_se_coef_tc_static = []
    str_pred_tc_recnt = []
    
    
    for sbset_tc in list_of_tc:
        print(sbset_tc)
        
        df_chsen = df_all_tranches_sbset[df_all_tranches_sbset['travel_cluster']==sbset_tc].reset_index(drop=True)

        df_chsen = df_chsen.sort_values(by=['tranche_order', 'LSOA11CD']).reset_index(drop=True)
        df_chsen = df_chsen[[x for x in df_chsen.columns if x not in ['tranche_desc', 'Date']]]

        pred_tc, coef_tc, se_tc, non_se_tc, pred_tst_tc = md.fit_model_tranche_static_dynamic(use_regularisation,
                                                                                          df_chsen,
                                                                                          zero_inf_flg_st,
                                                                                          alphas_val,
                                                                                          param_search_space,
                                                                                          df_all_tranches_sbset_tst_data)
        
        
        str_pred_tc_static.append(pred_tc)
        str_coef_tc_static.append(coef_tc) 
        str_se_coef_tc_static.append(se_tc)
        str_non_se_coef_tc_static.append(non_se_tc)
        str_pred_tc_recnt.append(pred_tst_tc)
        
    # Store most important features used for making predictions 
    # for each travel cluster and for each tranche
    # This includes both significant and non-significant features
    str_coef_tc_static = pd.concat(str_coef_tc_static).reset_index()

    # store the standard error/p-value of the coefficients of trained model - standardised coefs
    str_se_coef_tc_static = pd.concat(str_se_coef_tc_static).reset_index()

    # store the standard error/p-value of the coefficients of trained model
    str_non_se_coef_tc_static = pd.concat(str_non_se_coef_tc_static).reset_index()
    
    # Store the predictions of trained model
    str_pred_tc_static = pd.concat(str_pred_tc_static).reset_index(drop=True)
    
    # latest prediction on unseen test data
    pred_latest = pd.concat(str_pred_tc_recnt).sort_values(by='Predicted_cases_test', ascending=False).reset_index(drop=True)
    
    result_dfs = [str_coef_tc_static, str_se_coef_tc_static, str_non_se_coef_tc_static]

    # derive a lookup table of dates to tranche numbers to tranche description
    df_tranche_date_lookup = df_all_tranches_sbset[['Date','tranche_desc']].drop_duplicates().reset_index(drop=True)
    tranche_description = cf.tranche_description

    event_dict = dict(zip(df_tranche_date_lookup['Date'].values, df_tranche_date_lookup['tranche_desc'].values))
    tranche_order = list(range(1, cf.n_tranches + 1))

    # store as dictionaries so that they can be mapped to dataframe columns later
    event_order_dict = dict(zip(tranche_description, tranche_order))
    rvse_event_dict = {v: k for k, v in event_order_dict.items()}
    rvse_date_dict = {v: k for k, v in event_dict.items()}
    
    # populate columns using the dictionaries created above
    for df in result_dfs:
        df['Date'] = df['tranche'].map(rvse_event_dict).map(rvse_date_dict)
        df['tranche_desc'] = df['tranche'].map(rvse_event_dict)
        df['Features'] = df['Features'].str.lower()
        
    
    str_pred_tc_static['tranche_desc'] = str_pred_tc_static['tranche_train'].map(rvse_event_dict)
    str_pred_tc_static['Date'] = str_pred_tc_static['tranche_desc'].map(rvse_date_dict)
    str_pred_tc_static.rename(columns={'tranche_train':'tranche'},inplace=True)
    
    str_pred_tc_static['Residual'] = str_pred_tc_static['Actual_cases'] - str_pred_tc_static['Predicted_cases_train']
    
     
    # rename columns for legibility
    str_se_coef_tc_static.rename(columns={'coef':'standardised_coef',
                                          'P>|t|':'P_value',
                                          '[0.025':'lower_bound',
                                          '0.975]':'upper_bound',
                                          'std err':'std_err'}, inplace=True)
    
    str_non_se_coef_tc_static.rename(columns={'P>|t|':'P_value',
                                              '[0.025':'lower_bound',
                                              '0.975]':'upper_bound',
                                              'std err':'std_err'}, inplace=True)
    
    if save_results:

        # write out the results to BigQuery
        str_coef_tc_static.to_gbq(cf.tranche_coefs_regularisation, project_id=cf.project_name, if_exists='replace')
        str_se_coef_tc_static.to_gbq(cf.tranche_coefs_standardised, project_id=cf.project_name, if_exists='replace')
        str_non_se_coef_tc_static.to_gbq(cf.tranche_coefs_non_standardised, project_id=cf.project_name, if_exists='replace')
        str_pred_tc_static.to_gbq(cf.tranche_preds_all_tranches, project_id=cf.project_name, if_exists='replace')
        pred_latest.to_gbq(cf.tranche_preds_latest, project_id=cf.project_name, if_exists='replace')

    return str_coef_tc_static, str_se_coef_tc_static, str_non_se_coef_tc_static, str_pred_tc_static, pred_latest
    
    
    
    
    
    
