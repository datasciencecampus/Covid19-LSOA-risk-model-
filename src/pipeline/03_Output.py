import sys
import os

# import from local data files
current_path = os.path.abspath('.')
sys.path.append(os.path.dirname(current_path))

from data_access.data_factory import DataFactory as factory
from utils import config as cf
from utils import dashboard as dash

##### LOAD DATA SETS #######

# Coefficients from the regression with regularisation
df_reg_coefs = factory.get('tranche_regularised_coefs').create_dataframe()

# Standardised coefficients from the regression without regularisation
df_non_reg_std_coefs = factory.get('tranche_non_reg_std_coefs').create_dataframe()

# Non-standardised coefficients from the regression without regularisation
df_non_reg_non_std_coefs = factory.get('tranche_non_reg_non_std_coefs').create_dataframe()

# Model predictions for each tranche including residuals - the regularised regression model is used for this
df_pred_residuals = factory.get('tranche_residuals').create_dataframe()

# Model predictions for the lastest tranche - the regularised regression model is used for this
df_latest_pred = factory.get('tranche_latest_predictions').create_dataframe()

# Model features
df_features = factory.get('tranche_model_input').create_dataframe()

###### PROCESS ###########

# Subset for the columns required in the dashboard
df_reg_coefs = df_reg_coefs[['Features','Coefficients','tranche', 'travel_cluster','Date']]
df_non_reg_std_coefs = df_non_reg_std_coefs[['Features','standardised_coef','P_value','lower_bound','upper_bound','tranche', 'travel_cluster','Date']]
df_non_reg_non_std_coefs = df_non_reg_non_std_coefs[['Features','coef','P_value','lower_bound','upper_bound','tranche', 'travel_cluster','Date']]
df_pred_residuals = df_pred_residuals[['LSOA11CD','tranche','Residual']]
df_latest_pred = df_latest_pred[['LSOA11CD','Predicted_cases_test','MSOA11NM']]

# Filter the features for the latest tranche for plotting
df_features = df_features[df_features['tranche_order'] == cf.n_tranches]

# Drop the column to allow for quintile calculation on all numerical columns
df_features.drop(['Date','tranche_order','tranche_desc','COVID_Cases_per_unit_area'], axis=1, inplace=True)

# compute quintiles for plotting
df_quint = dash.make_quintiles(df_features)

# encode travel clusters as integers for data studio plotting
df_encoded = dash.encode_column(df_quint, 'travel_cluster')

# pivot the data set
df_features_pivot = dash.pivot_results(df_encoded)

#### PRETTY NAMES ####

# Rename column to fit into the for loop below
df_features_pivot.rename({'feature':'Features'}, axis='columns', inplace=True)

df_resid_list = [df_reg_coefs, df_non_reg_std_coefs, df_non_reg_non_std_coefs, df_features_pivot]

for df in df_resid_list:
    
    df['tc_short_name'] = df['travel_cluster']
    
    df = dash.pretty_rename(df, 'Features', cf.feature_pretty_names)
    df = dash.pretty_rename(df, 'travel_cluster', cf.tc_pretty_names)
    df = dash.pretty_rename(df, 'tc_short_name', cf.tc_short_names)
    

# filter for rows containing quintile scores
search_term = 'Quintile'
df_features_pivot = df_features_pivot[df_features_pivot['Features'].str.contains("|".join(search_term))]


# Write to BigQuery
df_reg_coefs.to_gbq('review_ons.dashboard_tranche_reg_coefs', project_id = cf.project_name, if_exists = 'replace')
df_non_reg_std_coefs.to_gbq('review_ons.dashboard_tranche_non_reg_std_coefs', project_id = cf.project_name, if_exists = 'replace')
df_non_reg_non_std_coefs.to_gbq('review_ons.dashboard_tranche_non_reg_non_std_coefs', project_id = cf.project_name, if_exists = 'replace')
df_pred_residuals.to_gbq('review_ons.dashboard_tranche_residuals', project_id = cf.project_name, if_exists = 'replace')
df_latest_pred.to_gbq('review_ons.dashboard_tranche_latest_preds', project_id = cf.project_name, if_exists = 'replace')
df_features_pivot.to_gbq('review_ons.dashboard_tranche_model_features', project_id = cf.project_name, if_exists = 'replace')