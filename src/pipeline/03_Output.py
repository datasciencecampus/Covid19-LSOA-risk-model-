## This script ingests the modelling results and transforms them to 

# import modules
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
df_features = factory.get('tranche_model_features').create_dataframe()

###### PROCESS ###########

# TEST REQUIREMENT - MAKE SURE THAT ONLY OBJECT AND FLOAT32 DTYPES EXIST IN THE DATAFRAME BEFORE WRITING TO BIGQUERY
# drop the 'Area' column - do this further back in the modelling process if possible

# Subset for the columns required in the dashboard
df_reg_coefs = df_reg_coefs[['Features','Coefficients','tranche', 'travel_cluster','Date']]
df_non_reg_std_coefs = df_non_reg_std_coefs[['Features','standardised_coef','P_value','lower_bound','upper_bound','tranche', 'travel_cluster','Date']]
df_non_reg_non_std_coefs = df_non_reg_non_std_coefs[['Features','coef','P_value','lower_bound','upper_bound','tranche', 'travel_cluster','Date']]
df_pred_residuals = df_pred_residuals[['LSOA11CD','tranche','Residual']]
df_latest_pred = df_latest_pred[['LSOA11CD','Predicted_cases_test','MSOA11NM']]

# No temporal element is needed for plotting this data set
df_features.drop('Date', axis=1, inplace=True)

# Filter the features for the latest tranche for plotting
df_features = df_features[df_features['tranche_order'] == cf.n_tranches]

# Drop the column to allow for quintile calculation on all numerical columns
df_features.drop(['tranche_order','tranche_desc','COVID_Cases_per_unit_area'], axis=1, inplace=True)

# compute quintiles for plotting
# The ready_meals feature is dropped - too many zeroes to compute quintiles with unique bin edges. Rank is used instead
df_quint = dash.make_quintiles(df_features.drop('ready_meals', axis=1))

# Make a separate dataframe for just the ready_meals column
df_ready_meals = df_features[['LSOA11CD', 'ready_meals']]

# rank LSOAs by ready meals
df_ready_meals['ready_meals_rank'] = df_ready_meals['ready_meals'].rank(method='max', ascending=False)

# Join ready meals with the other features
df_features = df_quint.merge(df_ready_meals, how='inner', on='LSOA11CD')

# encode travel clusters as integers for data studio plotting
df_encoded = dash.encode_column(df_features, 'travel_cluster')

# pivot the data set
df_features_pivot = dash.pivot_results(df_encoded)

### PRETTY NAMES ###

# Rename column to fit into the for loop below
df_features_pivot.rename({'feature':'Features'}, axis='columns', inplace=True)

df_resid_list = [df_reg_coefs, df_non_reg_std_coefs, df_non_reg_non_std_coefs, df_features_pivot]

for df in df_resid_list:
    
    df['tc_short_name'] = df['travel_cluster']
    
    df = dash.pretty_rename(df, 'Features', cf.feature_pretty_names)
    df = dash.pretty_rename(df, 'travel_cluster', cf.tc_pretty_names)
    df = dash.pretty_rename(df, 'tc_short_name', cf.tc_short_names)
    

# filter df_features DataFrame for just the ranks and quintile scores
search_terms = ['Rank','Quintile']

df_features_pivot = df_features_pivot[df_features_pivot['Features'].str.contains("|".join(search_terms))]


# Write to BigQuery
df_reg_coefs.to_gbq('review_ons.dashboard_tranche_reg_coefs', project_id = cf.project_name, if_exists = 'replace')
df_non_reg_std_coefs.to_gbq('review_ons.dashboard_tranche_non_reg_std_coefs', project_id = cf.project_name, if_exists = 'replace')
df_non_reg_non_std_coefs.to_gbq('review_ons.dashboard_tranche_non_reg_non_std_coefs', project_id = cf.project_name, if_exists = 'replace')
df_pred_residuals.to_gbq('review_ons.dashboard_tranche_residuals', project_id = cf.project_name, if_exists = 'replace')
df_latest_pred.to_gbq('review_ons.dashboard_tranche_latest_preds', project_id = cf.project_name, if_exists = 'replace')
df_features_pivot.to_gbq('review_ons.dashboard_tranche_model_features', project_id = cf.project_name, if_exists = 'replace')