# Import Packages
from google.cloud import bigquery

import pandas as pd
import numpy as np
import pandas_gbq

from datetime import date
from datetime import datetime

import random
from random import randint
import math
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats.mstats import zscore

from functools import reduce

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression

from sklego.meta import ZeroInflatedRegressor
from sklego.meta import EstimatorTransformer

from sklearn.metrics import median_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

from numpy.random import seed
from numpy.random import randn

import statsmodels.api as sm
import statsmodels.formula.api as smf


def output_scores(rs, train_r2, test_r2, alpha_val ):
    print("Best estimator:")
    print(rs.best_estimator_)
    print("Best score:")
    print(rs.best_score_)
    print("Best parameters:")
    print(rs.best_params_)
    print("r2_score: train score={0}, test score={1}".format(train_r2,test_r2))
    print('Optimal alpha {}'.format(alpha_val))

def r2(x, y):
    '''This function returns
       pearson correlation between
       two vectors of same length
       x and y
       '''
    return stats.pearsonr(x, y)[0] ** 2

def plot_var_by_travel_cluster(df_to_plot,varbl,wk_or_dat,oprtn):
    '''
    Plot dynamic variables/performance
    indicators aggregated
    for different
    travel clusters
    '''
    df_to_plot=df_to_plot[[varbl,wk_or_dat,'travel_cluster']]
   
    df_to_plot=df_to_plot.dropna()
    if (wk_or_dat=='week')|(wk_or_dat=='week_train'):
        df_to_plot['week_number']=df_to_plot[wk_or_dat].str.strip('week_').astype(int)
        if oprtn=='mean':
            which_tc=df_to_plot.groupby(['travel_cluster','week_number'])[varbl].mean().reset_index()
            fig = px.bar(which_tc, y=varbl, x="week_number",color='travel_cluster',barmode="group")
            fig.show()
            print(df_to_plot.groupby(['travel_cluster'])[varbl].mean())
        elif oprtn=='sum':
            which_tc=df_to_plot.groupby(['travel_cluster','week_number'])[varbl].sum().reset_index()
            fig = px.line(which_tc, y=varbl, x="week_number",color='travel_cluster')
            fig.show()
            print(df_to_plot.groupby(['travel_cluster'])[varbl].sum())
            
    elif wk_or_dat=='Date':
        if oprtn=='mean':
            which_tc=df_to_plot.groupby(['travel_cluster',wk_or_dat])[varbl].mean().reset_index()
            fig = px.bar(which_tc, y=varbl, x="Date",color='travel_cluster',barmode="group")
            fig.show()
            print(df_to_plot.groupby(['travel_cluster'])[varbl].mean())
        elif oprtn=='sum':
            which_tc=df_to_plot.groupby(['travel_cluster',wk_or_dat])[varbl].sum().reset_index()
            fig = px.line(which_tc, y=varbl, x="Date",color='travel_cluster')
            fig.show()
            print(df_to_plot.groupby(['travel_cluster'])[varbl].sum())
    elif (wk_or_dat !='week') |(wk_or_dat=='week_train')| (wk_or_dat!='Date'):
        print('Provided time-frame column does not exist')


def get_train_test_dataframes(df_to_fit,grp_var,week_indx):
    # We fit a different model for each travel_cluster
    # Some of these print ouputs are for visually checking 
    # for any bugs in the code       
    print('travel_cluster {}'.format(df_to_fit['travel_cluster'].unique()))
        
        
    week_index=sorted(list(set(df_to_fit['week'].str.strip('week_').astype(int))))
    # Get the training and testing data
    trng_week=[]
    for incr in range(1,grp_var+1):
        trng_week.append('week_{}'.format(week_index[week_indx+incr-1]))
    testng_week=['week_{}'.format(week_index[week_indx+incr])]
   
    df_train=df_to_fit[df_to_fit.week.isin(trng_week)]
   
    print('Training {}'.format(df_train['week'].unique()))
   
    num_cols_week=df_train.select_dtypes(include=[np.number]).columns
    #TA: is the first non_num_cols_week redundant?
    non_num_cols_week=[x for x in df_train.columns if x not in num_cols_week]
    #CJ: we retain week column
    #non_num_cols_week=[x for x in non_num_cols_week if x not in ['week']]
   
    # Aggregated training data- aggregated over grp_var number of weeks
    df_train=df_train.groupby(non_num_cols_week)[num_cols_week].mean().reset_index()
   
    df_test=df_to_fit[df_to_fit.week.isin(testng_week)]
    
   
    print('Testing {}'.format(df_test['week'].unique()))
    # We have to ensure train and test datasets are in the same order of LSOAs
    df_test = df_test.set_index('LSOA11CD')
    df_test = df_test.reindex(index=df_train['LSOA11CD'])
    df_test = df_test.reset_index()
    return df_train, df_test

def get_scaled_train_test_split(df_train, df_test,trgt_col, dynamic_features=[]):
    
   
    X_train=df_train[[x for x in df_train.columns if x not in ['LSOA11CD',trgt_col,'week','travel_cluster']]]
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint8']
   
    #TA: have you tried np.number?
    X_train = X_train.select_dtypes(include=numerics)
    X_train = X_train[[x for x in X_train.columns if x not in dynamic_features]]
   
    y_train=df_train[trgt_col]
   
    # FIRST FIT THE MODEL ONLY ON STATIC FEATURES: EXPERIMENT WITH THIS
   
    feature_names=X_train.columns
    
    X_test=df_test[feature_names]
    y_test=df_test[trgt_col]
   
    #print(X_train.head())
    # SCALE THE STATIC FEATURES
    scaler = StandardScaler()
    scaler.fit(X_train)  # fit only on training data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, feature_names

def cumulative_data(dfin, dynamic_cols):
    dynamic_df_sbset=dfin[dynamic_cols]

    #split the data for each LSOA
    dynamic_df_sbset_lsoa = [pd.DataFrame(y) for x, y in dynamic_df_sbset.groupby('LSOA11CD', as_index=False)]

    # CUMULATIVE CASES DATA FOR ALL THE LSOA FROM THE PREVIOUS WEEK
    past_week_cum_cases_dynamic_df_sbset=[x.sort_values(by='Date')[dynamic_cols].set_index(['Date','LSOA11CD']).shift(periods=1).dropna()
                            for x in dynamic_df_sbset_lsoa]

    # CONCAT cumulative CASES DATA (FROM THE PREVIOUS WEEK) FOR ALL THE LSOA
    cum_cases_df=pd.concat(past_week_cum_cases_dynamic_df_sbset,axis=0).reset_index()
    return cum_cases_df

def difference_data(dfin, dynamic_cols):
    dynamic_df_sbset=dfin[dynamic_cols]

    #split the data for each LSOA
    dynamic_df_sbset_lsoa = [pd.DataFrame(y) for x, y in dynamic_df_sbset.groupby('LSOA11CD', as_index=False)]

    # CHANGE IN  DATA FOR ALL THE LSOA FROM THE PREVIOUS WEEK
    past_week_cum_cases_dynamic_df_sbset=[x.sort_values(by='Date')[dynamic_cols].set_index(['Date','LSOA11CD']).diff().dropna()
                            for x in dynamic_df_sbset_lsoa]

    # CHANGE IN DATA (FROM THE PREVIOUS WEEK) FOR ALL THE LSOA
    cum_cases_df=pd.concat(past_week_cum_cases_dynamic_df_sbset,axis=0).reset_index()
    return cum_cases_df


def fit_model_one_week_static(df_to_fit,grp_var,zero_inf_flg,dynamic_ftrs,alp_regrlsn,parm_spce):
    
    '''
    This function fits a zero-inflated regression model 
    to the static data for predicting the COVID risk.
    df_to_fit: Input dataframe 
    grp_var: Number of weeks used 
    in training the model (weekly aggregation)
    zero_inf_flg: whether to use zero-inflated regressor model
    dynamic_ftrs: list of dynamic features to be removed from the dataset
    Outputs:
    predictions_lsoa: predictions of COVID risk
    for one week into the future
    coeffs_model: important features picked up
    by the model during the training phase
    '''
    #TA: change to dict with given cols
    # Empty lists for storing the outputs of the model
    
    
    str_tranch_dict = {
        'Actual_cases':[],
        'Predicted_cases_train':[],
        'Predicted_cases':[],
        'week_train':[],
        'week':[],
        'travel_cluster':[],
        'LSOA11CD':[],
        'r2_score':[],
        'Best_cv_score_train':[],
        'RMSE_train':[],
        'Probability_of_COVID_Case':[],
        'Predicted_Class':[],
        'Probability_of_COVID_Case_train':[],
        'Predicted_Class_train':[]
    }
    
    str_tranch_dict_lst_wk = {
        'Actual_cases':[],
        'Predicted_cases_train':[],
        'week_train':[],
        'travel_cluster':[],
        'LSOA11CD':[],
        'Best_cv_score_train':[],
        'RMSE_train':[],
        'Probability_of_COVID_Case_train':[],
        'Predicted_Class_train':[]
    }
    str_coef_risk_tranch=[]

    trgt_col='COVID_Cases_per_unit_area'#COVID_Cases_per_unit_area'

    for week_indx in range(df_to_fit['week'].unique().shape[0]-grp_var):
        
        df_train, df_test = get_train_test_dataframes(df_to_fit,grp_var,week_indx)
        
        
        X_train, X_test, y_train, y_test, feature_names = get_scaled_train_test_split(df_train, df_test,trgt_col, dynamic_features=dynamic_ftrs)
        
        
        which_lsoa_test=df_test['LSOA11CD'].values
        which_tc_test=df_test['travel_cluster'].unique()[0]
        which_week_train=df_train['week'].unique()[0]
        which_week=df_test['week'].unique()[0]
        num_cols_test=X_test.shape[0]
        
        y_train_tmp=[int(1) if y!=0 else int(y) for y in y_train]
        
        
        y_test_tmp=[int(1) if y!=0 else int(y) for y in y_test]
        
        
        
        
        
        clf = LogisticRegression(random_state=1,max_iter=10000)
        rgr=ElasticNet(random_state=1)
        #rgr=LinearRegression()
        
        
        
        #ZERO-INFLATED REGRESSOR
        #https://github.com/koaning/scikit-lego/blob/main/sklego/meta/zero_inflated_regressor.py
        
        # CROSS-VALIDATION, HYPER-PARAMETER OPTIMISATION BY RANDOM SEARCH
        # INCREASE/DECREASE THE VALUE OF n_iter TO EXPAND/REDUCE RANDOM SEARCH TO SEARCH FOR OPTIMAL
        # PARAMETERS..high value of cv takes longer to run 
        
       
        
        if (zero_inf_flg):
            
            param_distributions={'C' : np.logspace(-3, 3, 100),
                                 'class_weight': [{0:x, 1:1.0-x} for x in np.linspace(0.0,0.99,100)]}
            
            rs_cf = RandomizedSearchCV(estimator=clf, param_distributions=param_distributions, 
                                    n_iter=parm_spce, cv=5, scoring="explained_variance", random_state=1)
            print("Zero-inflated model: CV starts for classifier")
            rs_cf.fit(X_train, y_train_tmp)
            print("Zero-inflated model: CV finished for classifier")
            
            sampl_wght_trn=rs_cf.best_estimator_.predict_proba(X_train)[:,1]
            
            
            param_distributions={'alpha' : alp_regrlsn,
                                 'l1_ratio':np.linspace(0.05, 1, 100)}
            
            rs_rg = RandomizedSearchCV(estimator=rgr, param_distributions=param_distributions, 
                                    n_iter=parm_spce, cv=5, scoring="explained_variance", random_state=1)
            print("Zero-inflated model: CV starts for regressor")
            rs_rg.fit(X_train, y_train,sample_weight=sampl_wght_trn)
            print("Zero-inflated model: CV finished for classifier")
            
        else:
            
            param_distributions={'alpha' : alp_regrlsn, 'l1_ratio':np.linspace(0.05, 1, 100)}
            #
            rs = RandomizedSearchCV(estimator=rgr, param_distributions=param_distributions, 
                               n_iter=parm_spce, cv=5, scoring="explained_variance", random_state=1)
            #param_distributions={'n_jobs':[-1]}
            #rs = RandomizedSearchCV(estimator=rgr, param_distributions=param_distributions,
            #                        n_iter=1, cv=5, scoring="explained_variance")
            print("CV starts for regularised regression without zero-inflated model")
            rs.fit(X_train, y_train)
            print("CV finishes for regularised regression without zero-inflated model")
            
        
    
       

        
        # OUTPUTS FROM THE TRAINED MODEL
        if (zero_inf_flg):
            sampl_wght_tst=rs_cf.predict_proba(X_test)[:,1]
            y_pred_train=rs_rg.predict(X_train)
            y_pred_train=np.maximum(0,y_pred_train)
            y_pred=rs_rg.predict(X_test)
            y_pred=np.maximum(0,y_pred)
            
                
            
        else:
            y_pred_train=rs.predict(X_train)
            y_pred_train=np.maximum(0,y_pred_train)
            y_pred=rs.predict(X_test)
            y_pred=np.maximum(0,y_pred)
            
            
        train_r2=r2_score(y_train, y_pred_train)
        test_r2=r2_score(y_test, y_pred)
        
        print("r2_score: train score={0}, test score={1}".format(train_r2,test_r2))
        str_tranch_dict['Predicted_cases'].extend(y_pred)
        str_tranch_dict['Predicted_cases_train'].extend(y_pred_train)
        str_tranch_dict['Actual_cases'].extend(np.ravel(y_test.values))
        str_tranch_dict['week_train'].extend([which_week_train]*len(y_test))
        str_tranch_dict['week'].extend([which_week]*len(y_test))
        str_tranch_dict['LSOA11CD'].extend(which_lsoa_test)
        str_tranch_dict['travel_cluster'].extend([which_tc_test]*len(y_test))
        
        str_tranch_dict['r2_score'].extend([r2_score(y_test, y_pred)]*len(y_test))
        str_tranch_dict['RMSE_train'].extend([mean_squared_error(y_train, y_pred_train, squared=False)]*len(y_train))
        
        if (zero_inf_flg):
            str_tranch_dict['Probability_of_COVID_Case'].extend(rs_cf.best_estimator_.predict_proba(X_test)[:,1])
            str_tranch_dict['Predicted_Class'].extend(rs_cf.best_estimator_.predict(X_test).astype(int))
            
            str_tranch_dict['Probability_of_COVID_Case_train'].extend(rs_cf.best_estimator_.predict_proba(X_train)[:,1])
            str_tranch_dict['Predicted_Class_train'].extend(rs_cf.best_estimator_.predict(X_train).astype(int))
            
            alpha_val=rs_rg.best_estimator_.get_params().get('alpha')
            coefs = pd.DataFrame(rs_rg.best_estimator_.coef_,columns=['Coefficients'], index=feature_names)
            
            str_tranch_dict['Best_cv_score_train'].extend([rs_rg.best_score_]*len(y_train))
            output_scores(rs_rg, train_r2, test_r2, alpha_val )
            
        else:
            str_tranch_dict['Probability_of_COVID_Case'].extend([0]*len(y_test))
            str_tranch_dict['Predicted_Class'].extend([0]*len(y_test))
            
            str_tranch_dict['Probability_of_COVID_Case_train'].extend([0]*len(y_train))
            str_tranch_dict['Predicted_Class_train'].extend([0]*len(y_train))
            
            alpha_val=rs.best_estimator_.get_params().get('alpha')
            coefs = pd.DataFrame(rs.best_estimator_.coef_,columns=['Coefficients'], index=feature_names)
            
            str_tranch_dict['Best_cv_score_train'].extend([rs.best_score_]*len(y_train))
            output_scores(rs, train_r2, test_r2, alpha_val )
    
        # STATIC RISK PREDICTORS
        coefs=coefs[coefs['Coefficients']!=0]
        coefs['week']=which_week_train
        coefs['travel_cluster']=which_tc_test
        coefs['regularisation_alpha']=alpha_val
        str_coef_risk_tranch.append(coefs)
    # The last week in the training data is a special case
    # as we want to fit and predict on the same week
    # and sliding window approach used above
    # will leave out the last week from training: CJ-20/08/2021
    if week_indx==df_to_fit['week'].unique().shape[0]-grp_var-1:
        
        
        X_train=X_test
        y_train_tmp=y_test_tmp
        y_train=y_test
        which_week_train=which_week
        
        which_lsoa_train=which_lsoa_test
        which_tc_train=which_tc_test
        
        if (zero_inf_flg):
            
            param_distributions={'C' : np.logspace(-3, 3, 100),
                                 'class_weight': [{0:x, 1:1.0-x} for x in np.linspace(0.0,0.99,100)]}
            
            rs_cf = RandomizedSearchCV(estimator=clf, param_distributions=param_distributions, 
                                    n_iter=parm_spce, cv=5, scoring="explained_variance", random_state=1)
            print(X_train.shape)
            print(len(y_train_tmp))
            print("Zero-inflated model: CV starts for classifier")
            rs_cf.fit(X_train, y_train_tmp)
            print("Zero-inflated model: CV finished for classifier")
            
            sampl_wght_trn=rs_cf.best_estimator_.predict_proba(X_train)[:,1]
            
            
            param_distributions={'alpha' : alp_regrlsn,
                                 'l1_ratio':np.linspace(0.05, 1, 100)}
            
            rs_rg = RandomizedSearchCV(estimator=rgr, param_distributions=param_distributions, 
                                    n_iter=parm_spce, cv=5, scoring="explained_variance", random_state=1)
            print("Zero-inflated model: CV starts for regressor")
            rs_rg.fit(X_train, y_train,sample_weight=sampl_wght_trn)
            print("Zero-inflated model: CV finished for classifier")
            
        else:
            
            param_distributions={'alpha' : alp_regrlsn, 'l1_ratio':np.linspace(0.05, 1, 100)}
            #
            rs = RandomizedSearchCV(estimator=rgr, param_distributions=param_distributions, 
                               n_iter=parm_spce, cv=5, scoring="explained_variance", random_state=1)
            #param_distributions={'n_jobs':[-1]}
            #rs = RandomizedSearchCV(estimator=rgr, param_distributions=param_distributions,
            #                        n_iter=1, cv=5, scoring="explained_variance")
            print("CV starts for regularised regression without zero-inflated model")
            rs.fit(X_train, y_train)
            print("CV finishes for regularised regression without zero-inflated model")
            
        
    
       

        
        # OUTPUTS FROM THE TRAINED MODEL
        if (zero_inf_flg):
            
            y_pred_train=rs_rg.predict(X_train)
            y_pred_train=np.maximum(0,y_pred_train)
            
            
                
            
        else:
            y_pred_train=rs.predict(X_train)
            y_pred_train=np.maximum(0,y_pred_train)
            
            
        train_r2=r2_score(y_train, y_pred_train)
       
        
        print("r2_score: train score={0}".format(train_r2))
        str_tranch_dict_lst_wk['Predicted_cases_train'].extend(y_pred_train)
        str_tranch_dict_lst_wk['Actual_cases'].extend(np.ravel(y_train.values))
        str_tranch_dict_lst_wk['week_train'].extend([which_week_train]*len(y_train))
        str_tranch_dict_lst_wk['LSOA11CD'].extend(which_lsoa_train)
        str_tranch_dict_lst_wk['travel_cluster'].extend([which_tc_train]*len(y_train))
        str_tranch_dict_lst_wk['RMSE_train'].extend([mean_squared_error(y_train, y_pred_train, squared=False)]*len(y_train))
        
        
        if (zero_inf_flg):
            
            
            str_tranch_dict_lst_wk['Probability_of_COVID_Case_train'].extend(rs_cf.best_estimator_.predict_proba(X_train)[:,1])
            str_tranch_dict_lst_wk['Predicted_Class_train'].extend(rs_cf.best_estimator_.predict(X_train).astype(int))
            
            alpha_val=rs_rg.best_estimator_.get_params().get('alpha')
            coefs = pd.DataFrame(rs_rg.best_estimator_.coef_,columns=['Coefficients'], index=feature_names)
           
            str_tranch_dict_lst_wk['Best_cv_score_train'].extend([rs_rg.best_score_]*len(y_train))
            output_scores(rs_rg, train_r2, train_r2, alpha_val )
            
        else:
           
            
            str_tranch_dict_lst_wk['Probability_of_COVID_Case_train'].extend([0]*len(y_train))
            str_tranch_dict_lst_wk['Predicted_Class_train'].extend([0]*len(y_train))
            
            alpha_val=rs.best_estimator_.get_params().get('alpha')
            coefs = pd.DataFrame(rs.best_estimator_.coef_,columns=['Coefficients'], index=feature_names)
            
            str_tranch_dict_lst_wk['Best_cv_score_train'].extend([rs.best_score_]*len(y_train))
            output_scores(rs, train_r2, train_r2, alpha_val )
    
        # STATIC RISK PREDICTORS
        coefs=coefs[coefs['Coefficients']!=0]
        coefs['week']=which_week_train
        coefs['travel_cluster']=which_tc_test
        coefs['regularisation_alpha']=alpha_val
        str_coef_risk_tranch.append(coefs)

    predictions_lsoa=pd.DataFrame.from_dict(str_tranch_dict)
    predictions_lsoa_lst_wk=pd.DataFrame.from_dict(str_tranch_dict_lst_wk)
    predictions_lsoa=pd.concat([predictions_lsoa,predictions_lsoa_lst_wk])
    coeffs_model=pd.concat(str_coef_risk_tranch)
    
    return predictions_lsoa,coeffs_model


def fit_model_one_week_dynamic(df_to_fit,grp_var,which_clustr_grp,alp_regrlsn,parm_spce):
    '''
    This function fits a regression model 
    to the dynamic data to model the residuals
    obtained from the static risk predictors.
    df_to_fit: Input dataframe 
    grp_var: Number of weeks used 
    in training the model (weekly aggregation)
    Outputs:
    predictions_lsoa: predictions of COVID risk
    for one week into the future
    coeffs_model: important features picked up
    by the model during the training phase
    '''
    
    # Empty lists for storing the outputs of the model
    str_tranch_dict = {
        'Actual_cases':[],
        'Predicted_cases':[],
        'week':[],
         which_clustr_grp:[],
        'LSOA11CD':[],
        'r2_score':[],
        'Best_cv_score':[],
        'RMSE':[],
        'Probability_of_COVID_Case':[],
        'Predicted_Class':[]
    }
    str_se_coeff_all_weeks=[]
    
    str_coef_risk_tranch=[]
    
    trgt_col='Residual'
    # Training is performed as follows:
    # First cycle:(assuming grp_var=1)
    # Target variable is change in residual (obtained from static risk predictors) 
    # from the previous week (so the starting point of training is residual (week_3)-residual (week_2))
    # Train on aggregated weekly data: week_3,
    # Predictions for week_4
    # Second cycle:
    # Train on aggregated weekly data: week_4
    # Predictions for week_(5)
    # ....
    
    for week_indx in range(df_to_fit['week'].unique().shape[0]-grp_var):
        
        df_train, df_test = get_train_test_dataframes(df_to_fit,grp_var,week_indx)
        X_train, X_test, y_train, y_test, feature_names = get_scaled_train_test_split(df_train, df_test,trgt_col)
        
        which_lsoa_test=df_test['LSOA11CD'].values
        which_tc_test=df_test[which_clustr_grp].unique()[0]
        
        which_week_train=df_train['week'].unique()[0]
        which_week_test=df_test['week'].unique()[0]
        num_cols_test=X_test.shape[0]
        
        print("CV starts.")
        
        
        # REGULARISED REGRESSION-- we over-write this with normal regression
        # as the number of features in the dynamic training are not very much
        #rs = RandomizedSearchCV(estimator=ElasticNet(),param_distributions={'alpha' : alp_regrlsn, 'l1_ratio':np.linspace(0.05, 1, 100)},
        #                        n_iter=parm_spce, cv=5, scoring="explained_variance")
        
        # OVER WRITING REGULARISED REGRESSION WITH NORMAL REGRESSION
        rs = RandomizedSearchCV(estimator=LinearRegression(),param_distributions={'n_jobs' :[-1]},
                                n_iter=1, cv=5, scoring="explained_variance" )
        
        rs.fit(X_train, y_train)
        
        y_pred_train=rs.predict(X_train)
        y_pred=rs.predict(X_test)
        train_r2=r2_score(y_train, y_pred_train)
        test_r2=r2_score(y_test, y_pred)
        
        str_tranch_dict['Predicted_cases'].extend(y_pred)
        str_tranch_dict['Actual_cases'].extend(y_test.values)
        str_tranch_dict['week'].extend([which_week_test]*len(y_test))
        str_tranch_dict['LSOA11CD'].extend(which_lsoa_test)
        str_tranch_dict[which_clustr_grp].extend([which_tc_test]*len(y_test))
        str_tranch_dict['Best_cv_score'].extend([rs.best_score_]*len(y_test))
        str_tranch_dict['r2_score'].extend([r2_score(y_test, y_pred)]*len(y_test))
        str_tranch_dict['RMSE'].extend([mean_squared_error(y_test, y_pred, squared=False)]*len(y_test))
        
        str_tranch_dict['Probability_of_COVID_Case'].extend(['NA']*len(y_test))
        str_tranch_dict['Predicted_Class'].extend(['NA']*len(y_test))
        
        coefs = pd.DataFrame(rs.best_estimator_.coef_,columns=['Coefficients'], index=feature_names)
        alpha_val=rs.best_estimator_.get_params().get('alpha')
            
        output_scores(rs, train_r2, test_r2, alpha_val )
    
        # DYNAMIC RISK PREDICTORS
        coefs=coefs[coefs['Coefficients']!=0]
        coefs['week']=which_week_train
        coefs[which_clustr_grp]=which_tc_test
        coefs['regularisation_alpha']=alpha_val
        str_coef_risk_tranch.append(coefs)
        
        if alpha_val==None:
            #Compute Standard error
            N = len(X_train)
            p = len(feature_names) + 1
        
            X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
            X_with_intercept[:, 0] = 1
            X_with_intercept[:, 1:p] = X_train
            #computing the beta regression matrix
            beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y_train.values
             
            #computing residuals (error) while training
            residuals = y_train-y_pred_train
            residual_sum_of_squares = residuals.T @ residuals
            sigma_squared_hat = residual_sum_of_squares / (N - p)
            var_beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * sigma_squared_hat
            str_se=[]
            str_se_coef=[]
            str_se_tc=[]
            
            for p_ in range(p):
                if p_==0:
                    ftr_nam='intercept'
                else:
                    ftr_nam=feature_names[p_-1]
                standard_error = var_beta_hat[p_, p_] ** 0.5
                standard_error = var_beta_hat[p_, p_] ** 0.5
                str_se_coef.append(ftr_nam)
                str_se.append(standard_error)
                str_se_tc.append(which_tc_test)
            se_coef_df=pd.DataFrame(zip(str_se_coef,str_se,str_se_tc),columns=['Features','Standard_error',which_clustr_grp])
            se_coef_df['week']=which_week_train
            str_se_coeff_all_weeks.append(se_coef_df)
            
        
        
        
    predictions_lsoa=pd.DataFrame.from_dict(str_tranch_dict)
    
    coeffs_model=pd.concat(str_coef_risk_tranch)
    
    se_coeffs_model=pd.concat(str_se_coeff_all_weeks)
    
    return predictions_lsoa,coeffs_model,se_coeffs_model


def get_train_tranche(df_train, trgt_col):
    

    X_train=df_train[[x for x in df_train.columns if x not in ['LSOA11CD',trgt_col,'week','travel_cluster']]]
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint8']
   
    X_train = X_train.select_dtypes(include=numerics)
    
   
    y_train=df_train[trgt_col]
   
    feature_names=X_train.columns
    
    return X_train, y_train, feature_names


def fit_model_tranche_static_dynamic(linr_rgr,df_to_fit,zero_inf_flg,alp_regrlsn,parm_spce,df_test):
    
    '''
    This function fits a linear regression model 
    to the data for predicting the COVID cases.
    Params:
    linr_rgr: which regressor to choose- linear or linear regression with regularisation for prediction
    df_to_fit: Input dataframe 
    zero_inf_flg: whether to use zero-inflated regressor model
    alp_regrlsn: list of alpha for regularisation
    parm_spce: number of steps used in random search for finding optimal params
    df_test: test data to obtain predictions
    Outputs:
    dataframes containing the model outputs
    '''
    
    # Empty lists for storing the outputs of the model
    str_tranch_dict = {
        'Actual_cases':[],
        'Predicted_cases_train':[],
        'tranche_train':[],
        'travel_cluster':[],
        'LSOA11CD':[],
        'Best_cv_score_train':[],
        'RMSE_train':[],
        'Probability_of_COVID_Case_train':[],
        'Predicted_Class_train':[]
    }
    
    str_coef_risk_tranch=[]
    
    str_se_coef_risk_tranch=[]
    
    str_non_se_coef_risk_tranch=[]

    trgt_col='COVID_Cases_per_unit_area'

    for trnch_indx in sorted(df_to_fit['tranche_order'].unique()):
        
        df_train=df_to_fit[df_to_fit['tranche_order']==trnch_indx]
        
        
        which_trnch_train=df_train['tranche_order'].unique()[0]
        
        df_train=df_train[[x for x in df_train.columns if x not in ['tranche_order']]]
        
        
        X_train, y_train, feature_names = get_train_tranche(df_train, trgt_col)
        
        X_train=X_train.reset_index(drop=True)
        
        y_train=y_train.reset_index(drop=True)
        
        # we keep unscld data to obtain standardised coefficients-which we obtain using statsmodel
        X_train_unscld=X_train.copy()
        y_train_unscld=y_train.copy()
        
        # For better predictions, scaling the data can be a good choice
        # Different ways to scale the data
        # We scale the data and sklearn to fit the model
        # and to make predictions on the unseen test data
        scaler = RobustScaler()
        #scaler = StandardScaler()
        #scaler = MinMaxScaler()
    
    
        scaler.fit(X_train)  # fit only on training data
        
        X_train = scaler.transform(X_train)
        
        # transform test data (only for the last tranche- as we use the last tranche
        # to make predictions on the test data)
        if trnch_indx==max(sorted(df_to_fit['tranche_order'].unique())):
            df_test=df_test.set_index('LSOA11CD')
            #align the order of LSOAs same as training data
            df_test=df_test.reindex(index=df_train['LSOA11CD'])
            str_lsoa=list(df_test.index)
            str_tc=list(df_test['travel_cluster'].values)
            ftrs_lst=list(feature_names)
            df_test=df_test[ftrs_lst].values
            #transform the test data
            df_test = scaler.transform(df_test)
            
        X_train=pd.DataFrame(X_train,columns=feature_names).reset_index(drop=True)
        
        num_cols_test=X_train.shape[0]
        
        y_train_tmp=[int(1) if y!=0 else int(y) for y in y_train]
        
        
        
        # Classifier for zero-inflated regression model
        clf = LogisticRegression(random_state=1,max_iter=10000)
        
        # we can choose normal regression or regression with regularisation
        
        if linr_rgr:
            rgr=LinearRegression()
            param_distributions_rgrn={'n_jobs':[-1]}
            parm_spce=1
            flg='out'
            
        else:
            rgr=ElasticNet()
            param_distributions_rgrn={'alpha' : alp_regrlsn,'l1_ratio':np.linspace(0.05, 1, 100)}
            parm_spce=parm_spce
            flg=''
        
        
        #ZERO-INFLATED REGRESSOR
        #https://github.com/koaning/scikit-lego/blob/main/sklego/meta/zero_inflated_regressor.py
        
        # CROSS-VALIDATION, HYPER-PARAMETER OPTIMISATION BY RANDOM SEARCH
        # INCREASE/DECREASE THE VALUE OF n_iter TO EXPAND/REDUCE RANDOM SEARCH TO SEARCH FOR OPTIMAL
        # PARAMETERS..high value of cv takes longer to run 
        
        
        if (zero_inf_flg):
            
            param_distributions={'C' : np.logspace(-3, 3, 100),
                                 'class_weight': [{0:x, 1:1.0-x} for x in np.linspace(0.0,0.99,100)]}
            
            rs_cf = RandomizedSearchCV(estimator=clf, param_distributions=param_distributions, 
                                    n_iter=parm_spce, cv=5, scoring="explained_variance")
            print("Zero-inflated model: CV starts for classifier")
            rs_cf.fit(X_train, y_train_tmp)
            print("Zero-inflated model: CV finished for classifier")
            
            sampl_wght_trn=rs_cf.best_estimator_.predict_proba(X_train)[:,1]
            
          
            
            rs = RandomizedSearchCV(estimator=rgr, param_distributions=param_distributions_rgrn,n_iter=parm_spce, cv=5, scoring="explained_variance")
            print("Zero-inflated model: CV starts for regressor")
            rs.fit(X_train, y_train,sample_weight=sampl_wght_trn)
            print("Zero-inflated model: CV finished for classifier")
            
        else:
            
            rs = RandomizedSearchCV(estimator=rgr, param_distributions=param_distributions_rgrn,n_iter=parm_spce, cv=5, scoring="explained_variance")
            print("CV starts without zero-inflated model and with{} regularisation".format(flg))
            #y_train=y_train.values.reshape(-1,1)
            rs.fit(X_train, y_train)
            
            mdl_params = np.append(rs.best_estimator_.intercept_,rs.best_estimator_.coef_)
            ftr_nam=['Intercept']+list(feature_names)
            
            print("CV finishes without zero-inflated model and with{} regularisation".format(flg))
          
            
       
            
        # To get non-standardised coefficients-we dont scale the predictors and target variables 
        X2 = sm.add_constant(X_train_unscld)
        results_summary_non_se = sm.OLS(y_train_unscld, X2).fit().summary()
        results_non_se_as_html = results_summary_non_se.tables[1].as_html()
        #non-standardised regression coefficients
        df_coefs_non_se=pd.read_html(results_non_se_as_html, header=0, index_col=0)[0]    
                
        # To get standardised coefficients-we scale the predictors and target variables
        # we loose the intercept/constant term as a result of this scaling
        df_jnd = X_train[feature_names].reset_index(drop=True)
        df_jnd[trgt_col] =y_train
        df_std=df_jnd.apply(stats.zscore)
        results_summary = sm.OLS(df_std[trgt_col],df_std[feature_names]).fit().summary()
        results_as_html = results_summary.tables[1].as_html()
        #standardised regression coefficients
        df_coefs_se=pd.read_html(results_as_html, header=0, index_col=0)[0]
            
            
        
        # PREDICTIONS  FROM THE TRAINED MODEL ON THE TRAINING DATA
        # BASED ON sklearn
        y_pred_train=rs.predict(X_train)
       

        train_r2=r2_score(y_train, y_pred_train)
        
        print("r2_score: train score={0}".format(train_r2))
        
        # PREDICTIONS  FROM THE TRAINED MODEL ON THE TEST DATA (ONLY FOR THE LAST TRANCHE)
        # BASED ON sklearn
        if trnch_indx==max(sorted(df_to_fit['tranche_order'].unique())):
            
            predtcns_tst=rs.predict(df_test)
                
            pred_tst_df=pd.DataFrame()
            pred_tst_df['LSOA11CD']=str_lsoa
            pred_tst_df['Predicted_cases_test']=predtcns_tst
            pred_tst_df['travel_cluster']= str_tc
        else:
            pred_tst_df=pd.DataFrame()
        
        
        
        
        which_tc_train=df_train['travel_cluster'].unique()[0]
        
        #Model performance obtained from sklearn
        str_tranch_dict['Predicted_cases_train'].extend(y_pred_train)
        str_tranch_dict['Actual_cases'].extend(np.ravel(y_train.values))
        str_tranch_dict['tranche_train'].extend([which_trnch_train]*len(y_train))
       
        str_tranch_dict['LSOA11CD'].extend(df_train['LSOA11CD'].values)
        str_tranch_dict['travel_cluster'].extend([which_tc_train]*len(y_train))
        str_tranch_dict['RMSE_train'].extend([mean_squared_error(y_train, y_pred_train, squared=False)]*len(y_train))
        
        if (zero_inf_flg):
            
            #Model parameters obtained from sklearn
            str_tranch_dict['Probability_of_COVID_Case_train'].extend(rs_cf.best_estimator_.predict_proba(X_train)[:,1])
            str_tranch_dict['Predicted_Class_train'].extend(rs_cf.best_estimator_.predict(X_train).astype(int))
            
            alpha_val=rs.best_estimator_.get_params().get('alpha')
            coefs = pd.DataFrame(rs.best_estimator_.coef_,columns=['Coefficients'], index=feature_names)
            
            str_tranch_dict['Best_cv_score_train'].extend([rs.best_score_]*len(y_train))
           
            
        else:
            #Model parameters obtained from sklearn
            str_tranch_dict['Probability_of_COVID_Case_train'].extend([0]*len(y_train))
            str_tranch_dict['Predicted_Class_train'].extend([0]*len(y_train))
            
            alpha_val=rs.best_estimator_.get_params().get('alpha')
            coefs = pd.DataFrame(rs.best_estimator_.coef_,columns=['Coefficients'], index=feature_names)
            
            str_tranch_dict['Best_cv_score_train'].extend([rs.best_score_]*len(y_train))
            
    
        # RISK PREDICTORS based on sklearn package
        coefs=coefs[coefs['Coefficients']!=0]
        coefs['tranche']=which_trnch_train
        coefs['travel_cluster']=which_tc_train
        coefs['regularisation_alpha']=alpha_val
        str_coef_risk_tranch.append(coefs)
        
    
        
        
        #Model parameters obtained from statsmodel package: standardised coefs    
        df_coefs_se['tranche']=which_trnch_train
        df_coefs_se['travel_cluster']=coefs['travel_cluster'].unique()[0]
        str_se_coef_risk_tranch.append(df_coefs_se)
            
        #Model parameters obtained from statsmodel package: non-standardised coefs     
        df_coefs_non_se['tranche']=which_trnch_train
        df_coefs_non_se['travel_cluster']=coefs['travel_cluster'].unique()[0]
        str_non_se_coef_risk_tranch.append(df_coefs_non_se)
            

           
    predictions_lsoa=pd.DataFrame.from_dict(str_tranch_dict)
    coeffs_model=pd.concat(str_coef_risk_tranch)
    coeffs_model.index.names = ['Features']
    
    
    
   
    se_coeffs=pd.concat(str_se_coef_risk_tranch)
    se_coeffs.index.names = ['Features']
        
    non_se_coeffs=pd.concat(str_non_se_coef_risk_tranch)
    non_se_coeffs.index.names = ['Features']
        
    return predictions_lsoa,coeffs_model,se_coeffs,non_se_coeffs,pred_tst_df