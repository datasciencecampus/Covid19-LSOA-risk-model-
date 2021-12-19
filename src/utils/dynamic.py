from datetime import date
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

import time
from functools import reduce
import matplotlib.pyplot as plt

import string
import math


#import dash
import pgeocode


from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests

import swifter

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import kpss



import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# import plotly.offline as py
# py.init_notebook_mode()
# import plotly.graph_objs as go
# import plotly.express as px

#import dash_core_components as dcc
#import dash_html_components as html
# from plotly.offline import iplot, init_notebook_mode
from matplotlib.pyplot import figure


from statsmodels.tsa.stattools import adfuller

def plot_var_by_travel_cluster(df_to_plot,varbl):
    '''
    Plot weekly aggregated dynamic variables
    for different travel clusters
    '''
    which_tc=df_to_plot.groupby(['travel_cluster','Date'])[varbl].sum().reset_index()
    fig = px.line(which_tc, y=varbl, color='travel_cluster',x="Date")
    fig.show()

    

def end_of_week(datm, days=6):
    '''
    Function to map daily Date column to the 
    respective Sunday of the week
    '''
    date_obj = datm
    start_of_week = date_obj - timedelta(days=date_obj.weekday())  # Monday
    en_of_week = start_of_week + timedelta(days=days)  # Sunday
    return  en_of_week.date()


def check_for_stationarity(df_to_chck,col_name,travel_cluster_df=None):
    '''
    Function to aggregate dynamic features
    by travel cluster and check for stationarity
    of the feature: vaccination/mobility
    '''
    
    if 'travel_cluster' not in list(df_to_chck.columns):
        df_to_chck=df_to_chck.merge(travel_cluster_df,on='LSOA11CD',how='inner')
        
    df_to_chck=df_to_chck.groupby(['Date','travel_cluster'])[col_name].mean().reset_index()
    
    df_to_chck= [pd.DataFrame(y) for x, y in df_to_chck.groupby('travel_cluster', as_index=False)]
    
    for df_tc in df_to_chck:
        name=col_name
        df_tc=df_tc.sort_values(by='Date').reset_index(drop=True)
        print('***'*100)
        print(df_tc['travel_cluster'].unique()[0])
        adfuller_test(df_tc[name],signif=0.05, name=name)
        print('AFTER DIFFERENCING')
        print('+++'*100)
        # We are not removing seasonality from Cases 
        if col_name=='COVID_Cases':
            df_tc_diff=df_tc[['Date',name]].set_index('Date').dropna()
        else:
            df_tc_diff=df_tc[['Date',name]].set_index('Date').diff().dropna()
        adfuller_test(df_tc_diff[name],signif=0.05, name=name)

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """
    Perform ADFuller to test for Stationarity
    of given series and print report
    """
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 
              'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")


def df_derived_by_shift(df,lag=0,NON_DER=[],trgt_col=[]):
    '''
    This function returns a df
    with index shifted by desired
    number of periods.
    '''
    df = df.copy()
    if not lag:
        return df
    cols ={}
    for i in range(1,lag+1):
        for x in list(df.columns):
            if x not in NON_DER+trgt_col:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k,v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)    
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i+=1
        df = pd.concat([df, dfn], axis=1)
    return df

class TimeLag():
    def __init__(self):
        self.name="Class  which holds functions to calculate time lag"
    def compute_time_lagged_corr(self,df_list_rgns,col_a,col_b,col_c,which_one_rgn,window_days,start_date=2020-30-12,n_lag=12,plt_Flg=True,moblty_flg=False):
    
        '''
        Computes time-lagged cross correlation
        between two columns of the df:col_a,col_b

        df_list_rgns: input df list based on the spatial split
        lsoa/msoa/regions/travel_cluster/country

        col_a: the target column: could be cases/mortality data

        col_b/col_c: feature for which optimal time lag
        with col_a needs to be computed. This could be 
        vacciantion data/ Mobility indicator.

        which_one_rgn: group variable in the multi-group modelling,
        could be travel_cluster/UTLA/LTLA/MSOA/LSOA

        window_days: number of days from the start of the year
        after which one starts comparing the two time series.
        Note, if window_days<0, then one starts the 
        comparison before the start of the year (might only make sense
        for mobility as vaccination only kicked in from December onwards
        and each of this indicator (Mob/vacc) is expected to influence
        target variable after a certain lag)

        n_lag: for how many historical weeks one wants to evaluate 
        time-lagged cross correlation.

        start_date=Date to use for cut off in format (YYYY-MM-DD)
        '''
        str_rgn=[]
        str_ftr=[]
        str_corr=[]
        str_optimal_lag=[]


        # Vaccination influence: start comparing the
        # two time series mid-January onwards
        # Mobility influence:window_days can be negative 
        # starting the signal before January 2021
        cut_off_datum_1=pd.to_datetime(f'{start_date}')+timedelta(window_days)

        print(cut_off_datum_1)

        for which_indx in range(len(df_list_rgns)):


            df=df_list_rgns[which_indx]

            trvl_clustr_nam=df[which_one_rgn].unique()[0]

            df=df.groupby(['Date',which_one_rgn])[[col_a,col_b,col_c]].sum().reset_index()

            df=df.set_index('Date')

            df=df.dropna()
            print(trvl_clustr_nam)

            df=df[[x for x in df.columns if x not in [which_one_rgn]]]



            print('==================before differencing ====================')
            # ADF Test on each column
            for name, column in df.iteritems():
                print(column.name)
                adfuller_test(column, name=column.name)
                print('\n')

            # different indicators might be stationary after different transformations
            # we start with first order difference
            #TA: deduplicated code here
            whch_col  =col_c if moblty_flg else col_b
            df=df[[col_a,whch_col]].diff().dropna()

            print('==================after differencing ====================')
            # ADF Test on each column
            for name, column in df.iteritems():
                print(column.name)
                adfuller_test(column, name=column.name)
                print('\n')

            df[which_one_rgn]=trvl_clustr_nam

            #TA: simplified code
            if not moblty_flg:
                df=df[df.index>pd.to_datetime(cut_off_datum_1)]


            df.sort_index(ascending=True, inplace=True)


            if plt_Flg:
                plt.rcParams["figure.figsize"] = (15,5)
                fig,ax = plt.subplots()
                ax.plot(df[col_a], color="blue", marker="o",alpha=0.5)
                ax.set_ylabel(col_a,color="blue",fontsize=14)
                ax.set_title(trvl_clustr_nam)

                ax2=ax.twinx()
                ax2.plot(df[whch_col],color="green",marker="d",alpha=0.5)
                ax2.set_ylabel(whch_col,color="green",fontsize=14)
                plt.show()





            # Compute time lagged cross correlation between
            # col_a and col_b/col_c for which the correlation is optimal
            # We are trying to find the optimal lag for col_b/col_c 
            # for which its either correlated/anti_correlated with the col_a


            NON_DER = ['Date',]

            trgt_col=[col_a,]

            df_new = df_derived_by_shift(df,n_lag, NON_DER,trgt_col)

            df_new = df_new.dropna()

            if plt_Flg:
                df_new.corr()[df_new.corr().index.isin([whch_col]+[whch_col+'_'+str(x+1) for x in range(n_lag)])][col_a].plot(style='-o')
                plt.xlabel('Time lag')
                plt.ylabel('Cross correlation')
                plt.title('Time lagged correlation of {} with {} for {}'.format(col_a,whch_col,trvl_clustr_nam))
                plt.xticks(rotation=60)
                plt.show()


            chsen_vectr=df_new.corr()[df_new.corr().index.isin([whch_col]+[whch_col+'_'+str(x+1) for x in range(n_lag)])][col_a]

            #TA: deduplicate:
            find_extrema = np.greater if moblty_flg else np.less

            radius=1 #local maximum: one data point on either side for vaccination
            vect=argrelextrema(chsen_vectr.values, find_extrema,order=radius)[0]
            vect_values=chsen_vectr.values[vect]
            print(vect)
            print(vect_values)
            print(vect[0])
            print(vect_values[0])
            str_optimal_lag.append(vect[0])
            str_corr.append(vect_values[0])

            str_rgn.append(trvl_clustr_nam)
            str_ftr.append(col_b)
            print(which_indx)

        nam_col='week'
        if moblty_flg:
            use_nam='inflow'
        else:
            use_nam='vaccination'
        df_optimal_corr=pd.DataFrame(np.column_stack([str_rgn,str_corr,str_optimal_lag]),\
                                     columns=[which_one_rgn,'Optimal_corr_'+use_nam,'Optimal_lag_'+nam_col+'_'+use_nam])
        return df_optimal_corr
    def get_time_lag_value(self,dfs, trgt, vacc, mobility, region,window_days,start_date, n_lag, plt_flg, moblty_flag):
        """
        This class will return the int value of timelags across all selected regions provided by the compute time lag function
        
        Parameters:
        dfs: A pandas dataframe split by regions e.g. ([pd.DataFrame(y) for x, y in df.groupby(region, as_index=False)])
        trgt: Target variable for calculating lag e.g. cases
        vacc: Vaccination column 
        mobility: mobility column
        region: column name of region, this should be the same as the granularity that the dataframe was split by.
        
        window_days: number of days from the start of the year
        after which one starts comparing the two time series.
        Note, if window_days<0, then one starts the 
        comparison before the start of the year (might only make sense
        for mobility as vaccination only kicked in from December onwards
        and each of this indicator (Mob/vacc) is expected to influence
        target variable after a certain lag)

        n_lag: for how many historical weeks one wants to evaluate 
        time-lagged cross correlation.

        start_date=Date to use for cut off in format (YYYY-MM-DD)
        
        plt_flg: bool, Flag for if plotting wanted.
        
        moblty_flag: bool, True if lag for mobility is being calculated, False if vaccination lag is to be calculated
        """
        

        optimal_corr_df=self.compute_time_lagged_corr(dfs,trgt,vacc,mobility,region,window_days,start_date,n_lag,plt_flg,
                                                      moblty_flag)  #calculate time lag

        if moblty_flag:
            optimal_corr_df=optimal_corr_df.rename(columns={'Optimal_lag_week_inflow':f'Optimal_lag_week_{mobility}',
                                                                                       'Optimal_corr_inflow':f'Optimal_corr_{mobility}'})      #rename to mobility column
            lag_value=np.rint(optimal_corr_df[f'Optimal_lag_week_{mobility}'].astype(float).mean()).astype(int)  #get lag value
            print (f"Lag for {mobility} calculated: {lag_value} weeks")        
        else:
            optimal_corr_df=optimal_corr_df.rename(columns={'Optimal_lag_week_vaccination':f'Optimal_lag_week_{vacc}',
                                                                                       'Optimal_corr_vaccination':f'Optimal_corr_{vacc}'}) #rename to the vaccinatin column
            lag_value=np.rint(optimal_corr_df[f'Optimal_lag_week_{vacc}'].astype(float).mean()).astype(int)
            print (f"Lag for {vacc} calculated: {lag_value} weeks")

        return lag_value

    def split_df_apply_time_lag(self,df_to_split,cols_to_retain, lag_value=None, apply_lag=True)->list:
        """
        This function splits a dataframe by the variable and LSOA's which then can apply a time lag 
        
        df_to_split: pandas dataframe to be split
        cols_to_retain: Column which are to be kept in e.g. the column that lag is being applied to
        lag_value: int, the value to lag values in columns per lsoa by
        apply_lag: bool, choose if lag value is to be applied. If false, only a split dataframe will be be returned
        """
        
        split_df= [pd.DataFrame(y)[['LSOA11CD','Date']+cols_to_retain].\
                                  sort_values(by='Date').set_index('Date')\
                                  for x, y in df_to_split.\
                                  groupby('LSOA11CD', as_index=False)]  #split dataframe by LSOA and retain specified columns
        
        
        if apply_lag:
            lagged_df=[x.shift(periods=lag_value).dropna() for x in split_df]  #apply time lag by lag value
        else:
            lagged_df=split_df #do nothing
        
        return lagged_df

