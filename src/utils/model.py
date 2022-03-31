import pandas as pd
import numpy as np
import pandas_gbq

import random
import math
from scipy import stats
from scipy.stats.mstats import zscore
from sklearn.model_selection import RandomizedSearchCV


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


def output_scores(rs, train_r2, test_r2, alpha_val):
    """
    Print evaluation metrics for a set of models trained using a RandomizedSearchCV object
    
    :param rs: A trained RandomizedSearchCV object
    :type rs: 
    
    :param train_r2: r2 score for the best fit on the cross-validated training data
    :type train_r2: float
    
    :param test_r2: r2 score for the fit of the best cross-validated model the test data
    :type test_r2: float
      
    :param alpha_val: The alpha value for the best performing model
    :type alpha_val: float
    
    :returns: None
    :rtype: None
    """
    print("Best estimator:")
    print(rs.best_estimator_)
    print("Best score:")
    print(rs.best_score_)
    print("Best parameters:")
    print(rs.best_params_)
    print("r2_score: train score={0}, test score={1}".format(train_r2,test_r2))
    print('Optimal alpha {}'.format(alpha_val))

def r2(x, y):
    """
    This function returns pearson correlation coefficient between two vectors of same length x and y
    
    :param x: The first vector of values to be evaluated
    :type x: Pandas Series
    
    :param y: The second vector of values to be evaluated
    :type y: Pandas Series
    
    :returns: The Pearson's correlation coefficient between the two vectors
    :rtype: float
    """
    return stats.pearsonr(x, y)[0] ** 2


def get_train_test_dataframes(df_to_fit, grp_var, week_indx):
    """
    This function extracts specific training and test dataframes from a data set
    covering the whole modelled period. The outputs are passed to other
    modelling functions for fitting.
    
    The grp_var parameter defines the time period for the training and test sets.
    This is usually set to 1 week so that in the first iteration, a model trains on week 1 and tests on week 2.
    When the function is called a second time as part of a loop within the fitting functions, this function extracts
    week 2 for training and week 3 for testing etc.
    
    :param df_to_fit: A DataFrame covering the whole modelled period (e.g. January 2021 to December 2021)
    :type df_to_fit: Pandas DataFrame
    
    :param grp_var: The time period, in weeks, covered by each train and test set
    :type grp_var: int
    
    :param week_indx: The week number on which to start the training data extract
    :type week_indx: int
    
    :return df_train: A DataFrame containing grp_var weeks of training data, starting a 
    week number 'week_indx'
    :rtype df_train: Pandas DataFrame
    
    :return df_test: A DataFrame containing grp_var week of testing data, starting on the week
    directly after the end of the testing data
    :rtype df_test: Pandas DataFrame
    """
    
    print('travel_cluster {}'.format(df_to_fit['travel_cluster'].unique()))
        
    week_index = sorted(list(set(df_to_fit['week'].str.strip('week_').astype(int))))
    
    # create lists of strings describing the training and testing weeks to extract
    trng_week = []
    
    for incr in range(1, grp_var + 1):
        trng_week.append('week_{}'.format(week_index[week_indx + incr - 1]))
        
    testng_week = ['week_{}'.format(week_index[week_indx + incr])]
   
    # subset the whole-period dataframe for training week(s) of interest
    df_train = df_to_fit[df_to_fit.week.isin(trng_week)]
   
    print('Training {}'.format(df_train['week'].unique()))
   
    num_cols_week = df_train.select_dtypes(include=[np.number]).columns

    non_num_cols_week = [x for x in df_train.columns if x not in num_cols_week]

    # aggregate the training data over grp_var number of weeks
    df_train = df_train.groupby(non_num_cols_week)[num_cols_week].mean().reset_index()
   
    # subset the whole-period dataframe for testing week(s) of interest
    df_test = df_to_fit[df_to_fit.week.isin(testng_week)]
    
    print('Testing {}'.format(df_test['week'].unique()))
    
    # ensure train and test datasets are in the same order of LSOAs
    df_test = df_test.set_index('LSOA11CD')
    df_test = df_test.reindex(index=df_train['LSOA11CD'])
    df_test = df_test.reset_index()
    
    return df_train, df_test


def get_scaled_train_test_split(df_train, df_test, trgt_col, dynamic_features=[]):
    """
    Split training and test dataframes into X and y arrays, filtering for only variables which are:
    
     - Static
     - Numeric
     - Explanatory
     
    Finally, apply scaling to the features
    
    :param df_train: Training data output from the get_train_test_dataframes() function
    :type df_train: Pandas DataFrame
    
    :param df_test: Testing data output from the get_train_test_dataframes() function
    :type df_test: Pandas DataFrame
    
    :param trgt_col: Target variable to populate y vector
    :type trgt_col: str
    
    :param dynamic_features: List of feature names to not include in the training data
    :type dynamic_features: [str]
    
    :returns X_train, X_test, y_train, y_test: Arrays of explanatory and target variables
    split into train and test. Numeric explanatory variables are scaled. 
    :rtype: Pandas DataFrame/Pandas Series
     
    :returns feature_names: List of column names in the X_train and X_test data sets
    :rtype feature_names: [str]
    """
    
    # filter for only the variables which are numeric, static and explanatory
    X_train = df_train[[x for x in df_train.columns if x not in ['LSOA11CD',trgt_col,'week','travel_cluster']]]
    X_train = X_train.select_dtypes(include=np.number)
    X_train = X_train[[x for x in X_train.columns if x not in dynamic_features]]
   
    y_train = df_train[trgt_col]
   
    # use the same columns for X_test
    feature_names = X_train.columns

    X_test = df_test[feature_names]
    y_test = df_test[trgt_col]
   
    # scale the static features
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, feature_names

def fit_model_one_week_static(df_to_fit, grp_var, zero_inf_flg, dynamic_ftrs, alphas_val, param_search_space):
    
    """
    This is the first stage of the two way fixed effects model. Data sets are wrangled into
    train and test, modelling is performed using ElasticNet regression. Only the static explanatory variables are 
    used as predictors and the target variable is COVID cases per unit area. RandomisedSearchCV is used to find the 
    optimal hyperparameters for the model each week. Predictions are made and metrics are computed and reported. 
    The residuals from the first stage models are used as the target variable in the second stage.
    
    :param df_to_fit: Input dataframe containing all weeks of data for a given travel cluster
    :type df_to_fit: Pandas DataFrame
    
    :param grp_var: Number of weeks of data used to train the model
    :type grp_var: int
    
    :param zero_inf_flg: Flag for whether to use zero-inflated regressor model
    :type zero_inf_flg: bool
    
    :param dynamic_ftrs: list of dynamic features to be removed from the dataset
    :type dynamic_ftrs: [str]
    
    :return predictions_lsoa: predictions of COVID risk for one week into the future
    :rtype: Pandas DataFrame
    
    :return coeffs_model: important features picked up by the model during the training phase
    :rtype coeffs_model: [float]
    """   
    
    # create dictionaries to store results
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
    
    str_coef_risk_tranch = []

    trgt_col = 'COVID_Cases_per_unit_area'

    # for each week
    for week_indx in range(df_to_fit['week'].unique().shape[0] - grp_var):
        
        # make train and test data sets
        df_train, df_test = get_train_test_dataframes(df_to_fit, grp_var, week_indx)    
        X_train, X_test, y_train, y_test, feature_names = get_scaled_train_test_split(df_train, df_test, trgt_col, dynamic_features=dynamic_ftrs)
        
        which_lsoa_test = df_test['LSOA11CD'].values
        which_tc_test = df_test['travel_cluster'].unique()[0]
        which_week_train = df_train['week'].unique()[0]
        which_week = df_test['week'].unique()[0]
        
        
        y_train_tmp = [int(1) if y!=0 else int(y) for y in y_train]
        y_test_tmp = [int(1) if y!=0 else int(y) for y in y_test]
        
        # instantiate classifier and regressor
        clf = LogisticRegression(random_state=1,max_iter=10000)
        rgr = ElasticNet(random_state=1)      
        
        if (zero_inf_flg):
            
            # define hyperparameter search space
            param_distributions = {'C' : np.logspace(-3, 3, 100),
                                 'class_weight': [{0:x, 1:1.0-x} for x in np.linspace(0.0,0.99,100)]}
            
            # instantiate randomised search with cross validation for the classifier
            rs_cf = RandomizedSearchCV(estimator=clf, param_distributions=param_distributions, 
                                    n_iter=param_search_space, cv=5, scoring="explained_variance", random_state=1)
            
            print("Zero-inflated model: CV starts for classifier")
            rs_cf.fit(X_train, y_train_tmp)
            print("Zero-inflated model: CV finished for classifier")
            
            sampl_wght_trn = rs_cf.best_estimator_.predict_proba(X_train)[:,1]
            
            
            param_distributions = {'alpha' : alphas_val,
                                 'l1_ratio':np.linspace(0.05, 1, 100)}
            
            # instantiate randomised search with cross validation for the regressor
            rs_rg = RandomizedSearchCV(estimator=rgr, param_distributions=param_distributions, 
                                    n_iter=param_search_space, cv=5, scoring="explained_variance", random_state=1)
            
            print("Zero-inflated model: CV starts for regressor")
            rs_rg.fit(X_train, y_train, sample_weight=sampl_wght_trn)
            print("Zero-inflated model: CV finished for classifier")
            
        else:
            
            param_distributions = {'alpha' : alphas_val, 'l1_ratio':np.linspace(0.05, 1, 100)}
            
            # instantiate search for just the regressor
            rs = RandomizedSearchCV(estimator=rgr, param_distributions=param_distributions, 
                               n_iter=param_search_space, cv=5, scoring="explained_variance", random_state=1)

            print("CV starts for regularised regression without zero-inflated model")
            rs.fit(X_train, y_train)
            print("CV finishes for regularised regression without zero-inflated model")
            
        # outputs from the trained model
        if (zero_inf_flg):
            sampl_wght_tst = rs_cf.predict_proba(X_test)[:, 1]
            y_pred_train = rs_rg.predict(X_train)
            y_pred_train = np.maximum(0, y_pred_train)
            y_pred = rs_rg.predict(X_test)
            y_pred = np.maximum(0, y_pred)
            
        else:
            y_pred_train = rs.predict(X_train)
            y_pred_train = np.maximum(0, y_pred_train)
            y_pred = rs.predict(X_test)
            y_pred = np.maximum(0, y_pred)
            
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred)
        
        print("r2_score: train score={0}, test score={1}".format(train_r2, test_r2))
        
        # populate the results dictionaries
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
            
            alpha_val = rs_rg.best_estimator_.get_params().get('alpha')
            coefs = pd.DataFrame(rs_rg.best_estimator_.coef_,columns=['Coefficients'], index=feature_names)
            
            str_tranch_dict['Best_cv_score_train'].extend([rs_rg.best_score_]*len(y_train))
            output_scores(rs_rg, train_r2, test_r2, alpha_val )
            
        else:
            str_tranch_dict['Probability_of_COVID_Case'].extend([0]*len(y_test))
            str_tranch_dict['Predicted_Class'].extend([0]*len(y_test))
            
            str_tranch_dict['Probability_of_COVID_Case_train'].extend([0]*len(y_train))
            str_tranch_dict['Predicted_Class_train'].extend([0]*len(y_train))
            
            alpha_val = rs.best_estimator_.get_params().get('alpha')
            coefs = pd.DataFrame(rs.best_estimator_.coef_,columns=['Coefficients'], index=feature_names)
            
            str_tranch_dict['Best_cv_score_train'].extend([rs.best_score_]*len(y_train))
            output_scores(rs, train_r2, test_r2, alpha_val )
    
        # coefficient estimates from the static model
        coefs = coefs[coefs['Coefficients'] != 0]
        coefs['week'] = which_week_train
        coefs['travel_cluster'] = which_tc_test
        coefs['regularisation_alpha'] = alpha_val
        str_coef_risk_tranch.append(coefs)
        
    # The last week in the training data is a special case
    # as we want to fit and predict on the same week
    # and sliding window approach used above
    # will leave out the last week from training
    
    # if the selected week is the last week in the data set
    if week_indx == df_to_fit['week'].unique().shape[0] - grp_var - 1:
        
        # use the same data sets for train and test
        X_train = X_test
        y_train_tmp = y_test_tmp
        y_train = y_test
        which_week_train = which_week
        
        which_lsoa_train = which_lsoa_test
        which_tc_train = which_tc_test
        
        if (zero_inf_flg):
            
            param_distributions = {'C' : np.logspace(-3, 3, 100),
                                 'class_weight': [{0:x, 1:1.0-x} for x in np.linspace(0.0,0.99,100)]}
            
            rs_cf = RandomizedSearchCV(estimator=clf, param_distributions=param_distributions, 
                                    n_iter=param_search_space, cv=5, scoring="explained_variance", random_state=1)
            
            print(X_train.shape)
            print(len(y_train_tmp))
            print("Zero-inflated model: CV starts for classifier")
            rs_cf.fit(X_train, y_train_tmp)
            print("Zero-inflated model: CV finished for classifier")
            
            sampl_wght_trn=rs_cf.best_estimator_.predict_proba(X_train)[:,1]
            
            
            param_distributions = {'alpha' : alphas_val,
                                 'l1_ratio':np.linspace(0.05, 1, 100)}
            
            rs_rg = RandomizedSearchCV(estimator=rgr, param_distributions=param_distributions, 
                                    n_iter=param_search_space, cv=5, scoring="explained_variance", random_state=1)
            
            print("Zero-inflated model: CV starts for regressor")
            rs_rg.fit(X_train, y_train, sample_weight=sampl_wght_trn)
            print("Zero-inflated model: CV finished for classifier")
            
        else:
            
            param_distributions = {'alpha' : alphas_val, 'l1_ratio':np.linspace(0.05, 1, 100)}
            
            rs = RandomizedSearchCV(estimator=rgr, param_distributions=param_distributions, 
                               n_iter=param_search_space, cv=5, scoring="explained_variance", random_state=1)

            print("CV starts for regularised regression without zero-inflated model")
            rs.fit(X_train, y_train)
            print("CV finishes for regularised regression without zero-inflated model")
            
        # outputs from the trained model
        if (zero_inf_flg):
            
            y_pred_train = rs_rg.predict(X_train)
            y_pred_train = np.maximum(0, y_pred_train)
               
        else:
            y_pred_train = rs.predict(X_train)
            y_pred_train = np.maximum(0, y_pred_train)
            
        train_r2 = r2_score(y_train, y_pred_train)
       
        
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
            
            alpha_val = rs_rg.best_estimator_.get_params().get('alpha')
            coefs = pd.DataFrame(rs_rg.best_estimator_.coef_,columns=['Coefficients'], index=feature_names)
           
            str_tranch_dict_lst_wk['Best_cv_score_train'].extend([rs_rg.best_score_]*len(y_train))
            output_scores(rs_rg, train_r2, train_r2, alpha_val)
            
        else:
           
            
            str_tranch_dict_lst_wk['Probability_of_COVID_Case_train'].extend([0]*len(y_train))
            str_tranch_dict_lst_wk['Predicted_Class_train'].extend([0]*len(y_train))
            
            alpha_val = rs.best_estimator_.get_params().get('alpha')
            coefs = pd.DataFrame(rs.best_estimator_.coef_, columns=['Coefficients'], index=feature_names)
            
            str_tranch_dict_lst_wk['Best_cv_score_train'].extend([rs.best_score_]*len(y_train))
            output_scores(rs, train_r2, train_r2, alpha_val )
    
        # coefficient estimates for the static features
        coefs = coefs[coefs['Coefficients'] != 0]
        coefs['week'] = which_week_train
        coefs['travel_cluster'] = which_tc_test
        coefs['regularisation_alpha'] = alpha_val
        str_coef_risk_tranch.append(coefs)

    # concatenate the results from all other weeks the with results from the last week
    predictions_lsoa = pd.DataFrame.from_dict(str_tranch_dict)
    predictions_lsoa_lst_wk = pd.DataFrame.from_dict(str_tranch_dict_lst_wk)
    predictions_lsoa = pd.concat([predictions_lsoa, predictions_lsoa_lst_wk])
    coeffs_model = pd.concat(str_coef_risk_tranch)
    
    return predictions_lsoa, coeffs_model


def fit_model_one_week_dynamic(df_to_fit, grp_var, which_clustr_grp, alphas_val, param_search_space):
    '''
    This function fits a linear regression model to the week-on-week changes in the dynamic features. The
    target variable is the week-on-week change in residuals from the first stage model, which uses only static
    predictors. The coefficient estimates produced by this second stage model show the estimated effect size
    of the dynamic features on the target after controlling for the static features.
    
    :param df_to_fit: Input dataframe 
    :type df_to_fit: 
    
    :param grp_var: Number of weeks used in training the model (weekly aggregation)
    :type grp_var: int
    
    alphas_val: List of values to try for the alpha parameter in training the model
    Outputs:
    predictions_lsoa: predictions of COVID risk
    for one week into the future
    coeffs_model: important features picked up
    by the model during the training phase
    '''
    
    # empty dict for storing the outputs of the model
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
    
    str_se_coeff_all_weeks = []
    
    str_coef_risk_tranch = []
    
    # the second stage uses the residuals from the first stage as the target variable
    trgt_col = 'Residual'
    
    # Training is performed as follows:
    # First cycle: (assuming grp_var=1)
    # Target variable is change in residual (obtained from static risk predictors) 
    # from the previous week (so the starting point of training is residual (week_3) - residual (week_2))
    # Train on aggregated weekly data: week_3,
    # Predictions for week_4
    # Second cycle:
    # Train on aggregated weekly data: week_4
    # Predictions for week_(5)
    # etc...
    
    # for each week
    for week_indx in range(df_to_fit['week'].unique().shape[0] - grp_var):
        
        # derive train and test data sets
        df_train, df_test = get_train_test_dataframes(df_to_fit, grp_var, week_indx)
        X_train, X_test, y_train, y_test, feature_names = get_scaled_train_test_split(df_train, df_test, trgt_col)
        
        which_lsoa_test = df_test['LSOA11CD'].values
        which_tc_test = df_test[which_clustr_grp].unique()[0]
        which_week_train = df_train['week'].unique()[0]
        which_week_test = df_test['week'].unique()[0]
        
        print("CV starts.")
        
        # instantiate randomised search with cross validation for linear regression
        rs = RandomizedSearchCV(estimator=LinearRegression(), param_distributions={'n_jobs' :[-1]},
                                n_iter=1, cv=5, scoring="explained_variance")
        
        # fit and predict
        rs.fit(X_train, y_train)
        y_pred_train = rs.predict(X_train)
        y_pred = rs.predict(X_test)
        
        # compute r2 scores
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred)
        
        # store the results
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
        
        # extract coefficient estimates and alpha values from the best estimator found from randomised search
        coefs = pd.DataFrame(rs.best_estimator_.coef_, columns=['Coefficients'], index=feature_names)
        alpha_val = rs.best_estimator_.get_params().get('alpha')
            
        # print scores
        output_scores(rs, train_r2, test_r2, alpha_val)
    
        # report dynamic predictor coefficient estimates
        coefs = coefs[coefs['Coefficients']!=0]
        coefs['week'] = which_week_train
        coefs[which_clustr_grp] = which_tc_test
        coefs['regularisation_alpha'] = alpha_val
        str_coef_risk_tranch.append(coefs)
        
        if alpha_val == None:
            
            # compute standard error
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
            
            # create empty lists to store standard errors
            str_se = []
            str_se_coef = []
            str_se_tc = []
            
            for p_ in range(p):
                if p_ == 0:
                    ftr_nam = 'intercept'
                else:
                    ftr_nam = feature_names[p_ - 1]
                    
                standard_error = var_beta_hat[p_, p_] ** 0.5
                standard_error = var_beta_hat[p_, p_] ** 0.5
                
                # append results to a list
                str_se_coef.append(ftr_nam)
                str_se.append(standard_error)
                str_se_tc.append(which_tc_test)
                
            # zip the lists together into a dataframe
            se_coef_df = pd.DataFrame(zip(str_se_coef, str_se, str_se_tc), columns=['Features', 'Standard_error', which_clustr_grp])
            se_coef_df['week'] = which_week_train
            str_se_coeff_all_weeks.append(se_coef_df)
        
    predictions_lsoa = pd.DataFrame.from_dict(str_tranch_dict)
    coeffs_model = pd.concat(str_coef_risk_tranch)
    se_coeffs_model = pd.concat(str_se_coeff_all_weeks)
    
    return predictions_lsoa, coeffs_model, se_coeffs_model


def get_train_tranche(df_train, trgt_col):
    
    """
    Separate a dataframe into X_train, y_train and a vector of feature names
    
    :param df_train: A DataFrame containing data training data including a target variables
    :type df_train: Pandas DataFrame
    
    :param trgt_col: Column name of the target variable
    :type trgt_col: string
    
    :return X_train: A DataFrame of predictors
    :rtype: Pandas DataFrame
    
    :return y_train: A Pandas Series of the target variable
    :rtype: Pandas Series
    
    :return feature_names: A list of feature names in the training data
    :rtype: [str]
    """
    

    X_train = df_train[[x for x in df_train.columns if x not in ['LSOA11CD',trgt_col,'week','travel_cluster']]]
   
    X_train = X_train.select_dtypes(include=np.number)  
   
    y_train = df_train[trgt_col]
   
    feature_names= X_train.columns
    
    return X_train, y_train, feature_names


def fit_model_tranche_static_dynamic(use_regularisation, df_to_fit, zero_inf_flg, alphas_val, param_search_space, df_test):
    
    '''
    This function fits a separate linear regression model to each time tranche in the 'tranche_index' column of 'df_to_fit'. 
    The model trained on the final tranche of data is used to make a prediction for the number of cases during the period 
    where mobility data is available but cases data is not available.
    
    :param use_regularisation: Flag indicating whether to use regression with regularisation. If False, a regular linear 
    regression model is used
    :type use_regularisation: bool
    
    :param df_to_fit: DataFrame containing a single time tranche of training data
    :type df_to_fit: Pandas DataFrame
    
    :param zero_inf_flg: Flag indicating whether a zero-inflated regressor 
    :type zero_inf_flg: bool
    
    :param alphas_val: List of values to try for the alpha parameter when training the model
    :type alphas_val: 
    
    :param param_search_space: number of unique combinations of parameters used in random search for finding the 
    optimal combination of parameter values
    :type param_search_space: int
    
    :param df_test: Test data - the time period for which we have mobility data but no cases data
    :type df_test: Pandas DataFrame
    
    :return predictions_lsoa: A DataFrame of predictions for COVID cases over the test data period. Predictions are
    made using the model trained on the final tranche for which cases data is available.
    :rtype predictions_lsoa: Pandas DataFrame
    
    :return coeffs_model: A Pandas DataFrame containing the estimated coefficients from the regression models
    :rtype: Pandas DataFrame
    
    :return se_coeffs: A Pandas DataFrame containing estimated regression coefficients from a linear regression model
    with no regularisation applied, and with standardised features.
    :rtype: Pandas DataFrame
    
    :return non_se_coeffs: A Pandas DataFrame containing estimated regression coefficients from a linear regression model
    with no regularisation applied, and without standardisation of the model features.
    :rtype: Pandas DataFrame
    
    :return pred_tst_df: A Pandas DataFrame containing predictions generated by the model trained on the final tranche. The
    predictions cover the period for which mobility data is available but cases data is not available.
    :rtype: Pandas DataFrame
    
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
    
    # lists for storing model coefficient dataframes
    str_coef_risk_tranch = []
    str_se_coef_risk_tranch = []
    str_non_se_coef_risk_tranch = []

    trgt_col = 'COVID_Cases_per_unit_area'

    # for each time tranche
    for trnch_indx in sorted(df_to_fit['tranche_order'].unique()):
        
        # subset the training data for the tranche of interest
        df_train = df_to_fit[df_to_fit['tranche_order']==trnch_indx]
        
        # drop the tranche_order column
        df_train = df_train[[x for x in df_train.columns if x not in ['tranche_order']]]
        
        # split into X and y 
        X_train, y_train, feature_names = get_train_tranche(df_train, trgt_col)
        
        # reset indices
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        
        # we keep unscld data to obtain standardised coefficients, which we obtain using statsmodel
        X_train_unscld = X_train.copy()
        y_train_unscld = y_train.copy()
        
        # For better predictions, scaling the data can be a good choice
        # Different ways to scale the data
        # We scale the data and sklearn to fit the model
        # and to make predictions on the unseen test data
        scaler = RobustScaler()
   
        # fit only on training data
        scaler.fit(X_train)  
        X_train = scaler.transform(X_train)
        
        # transform test data (only for the last tranche- as we use the last tranche
        # to make predictions on the test data)
        if trnch_indx == max(sorted(df_to_fit['tranche_order'].unique())):
            
            df_test=df_test.set_index('LSOA11CD')
            
            #align the order of LSOAs same as training data
            df_test = df_test.reindex(index=df_train['LSOA11CD'])
            
            str_lsoa = list(df_test.index)
            str_tc = list(df_test['travel_cluster'].values)
            ftrs_lst = list(feature_names)
            df_test = df_test[ftrs_lst].values
            
            #transform the test data
            df_test = scaler.transform(df_test)
            
        # convert to DataFrame and reset index
        X_train = pd.DataFrame(X_train,columns=feature_names).reset_index(drop=True)
                
        y_train_tmp = [int(1) if y!=0 else int(y) for y in y_train]
                
        # classifier for zero-inflated regression model
        clf = LogisticRegression(random_state=1,max_iter=10000)
        
        # we can choose regression with regularisation or standard linear regression
        if use_regularisation:
            
            rgr = ElasticNet()
            param_distributions_rgrn = {'alpha' : alphas_val, 'l1_ratio':np.linspace(0.05, 1, 100)}
            param_search_space = param_search_space
            flg = ''
            
            
        else:

            rgr = LinearRegression()
            param_distributions_rgrn = {'n_jobs':[-1]}
            param_search_space = 1
            flg = 'out'      
        
        if (zero_inf_flg):
            
            param_distributions = {'C' : np.logspace(-3, 3, 100),
                                 'class_weight': [{0:x, 1:1.0-x} for x in np.linspace(0.0,0.99,100)]}
            
            rs_cf = RandomizedSearchCV(estimator=clf, param_distributions=param_distributions, 
                                    n_iter=param_search_space, cv=5, scoring="explained_variance")
            
            print("Zero-inflated model: CV starts for classifier")
            rs_cf.fit(X_train, y_train_tmp)
            print("Zero-inflated model: CV finished for classifier")
            
            sampl_wght_trn=rs_cf.best_estimator_.predict_proba(X_train)[:,1]
            
          
            
            rs = RandomizedSearchCV(estimator=rgr, param_distributions=param_distributions_rgrn,
                                    n_iter=param_search_space, cv=5, scoring="explained_variance")
            
            print("Zero-inflated model: CV starts for regressor")
            rs.fit(X_train, y_train,sample_weight=sampl_wght_trn)
            print("Zero-inflated model: CV finished for classifier")
            
        else:
            
            # find the optimal parameters
            rs = RandomizedSearchCV(estimator=rgr, 
                                    param_distributions=param_distributions_rgrn,
                                    n_iter=param_search_space, 
                                    cv=5, 
                                    scoring="explained_variance")
            
            print("CV starts without zero-inflated model and with{} regularisation".format(flg))
        
            # fit the model
            rs.fit(X_train, y_train)
            
            # extract the parameter values for the best estimator
            mdl_params = np.append(rs.best_estimator_.intercept_, rs.best_estimator_.coef_)
                        
            print("CV finishes without zero-inflated model and with{} regularisation".format(flg))
          
            
       
            
        # to get non-standardised coefficients, we dont scale the predictors and target variables 
        X2 = sm.add_constant(X_train_unscld)
        results_summary_non_se = sm.OLS(y_train_unscld, X2).fit().summary()
        results_non_se_as_html = results_summary_non_se.tables[1].as_html()
        
        #non-standardised regression coefficients
        df_coefs_non_se=pd.read_html(results_non_se_as_html, header=0, index_col=0)[0]    
                
        # to get standardised coefficients, we scale the predictors and target variables
        # we lose the intercept/constant term as a result of this scaling
        df_jnd = X_train[feature_names].reset_index(drop=True)
        df_jnd[trgt_col] = y_train
        df_std = df_jnd.apply(stats.zscore)
        results_summary = sm.OLS(df_std[trgt_col], df_std[feature_names]).fit().summary()
        results_as_html = results_summary.tables[1].as_html()
        
        # standardised regression coefficients
        df_coefs_se = pd.read_html(results_as_html, header=0, index_col=0)[0]
            
                
        # PREDICTIONS  FROM THE TRAINED MODEL ON THE TRAINING DATA
        # BASED ON sklearn
        y_pred_train = rs.predict(X_train)
       

        train_r2 = r2_score(y_train, y_pred_train)
        
        print("r2_score: train score={0}".format(train_r2))
        
        # PREDICTIONS  FROM THE TRAINED MODEL ON THE TEST DATA (ONLY FOR THE LAST TRANCHE)
        # BASED ON sklearn
        if trnch_indx == max(sorted(df_to_fit['tranche_order'].unique())):
            
            predtcns_tst = rs.predict(df_test)
                
            pred_tst_df = pd.DataFrame()
            pred_tst_df['LSOA11CD'] = str_lsoa
            pred_tst_df['Predicted_cases_test'] = predtcns_tst
            pred_tst_df['travel_cluster'] = str_tc
        else:
            pred_tst_df = pd.DataFrame()
        
        
        
        # the data is grouped by travel cluster before if it passed to this modelling function.
        # all values in the 'travel_cluster' column are identical.
        which_tc_train = df_train['travel_cluster'].unique()[0]
        
        #Model performance obtained from sklearn
        str_tranch_dict['Predicted_cases_train'].extend(y_pred_train)
        str_tranch_dict['Actual_cases'].extend(np.ravel(y_train.values))
        str_tranch_dict['tranche_train'].extend([trnch_indx]*len(y_train))
       
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
            
            alpha_val = rs.best_estimator_.get_params().get('alpha')
            coefs = pd.DataFrame(rs.best_estimator_.coef_,columns=['Coefficients'], index=feature_names)
            
            str_tranch_dict['Best_cv_score_train'].extend([rs.best_score_]*len(y_train))
            
    
        # RISK PREDICTORS based on sklearn package
        coefs = coefs[coefs['Coefficients']!=0]
        coefs['tranche'] = trnch_indx
        coefs['travel_cluster'] = which_tc_train
        coefs['regularisation_alpha'] = alpha_val
        str_coef_risk_tranch.append(coefs)
        
    
        
        
        #Model parameters obtained from statsmodel package: standardised coefs    
        df_coefs_se['tranche'] = trnch_indx
        df_coefs_se['travel_cluster'] = which_tc_train
        str_se_coef_risk_tranch.append(df_coefs_se)
            
        #Model parameters obtained from statsmodel package: non-standardised coefs     
        df_coefs_non_se['tranche'] = trnch_indx
        df_coefs_non_se['travel_cluster'] = which_tc_train
        str_non_se_coef_risk_tranch.append(df_coefs_non_se)
            

           
    predictions_lsoa = pd.DataFrame.from_dict(str_tranch_dict)
    coeffs_model = pd.concat(str_coef_risk_tranch)
    coeffs_model.index.names = ['Features']
            
   
    se_coeffs = pd.concat(str_se_coef_risk_tranch)
    se_coeffs.index.names = ['Features']
        
    non_se_coeffs = pd.concat(str_non_se_coef_risk_tranch)
    non_se_coeffs.index.names = ['Features']
        
    return predictions_lsoa, coeffs_model, se_coeffs, non_se_coeffs, pred_tst_df