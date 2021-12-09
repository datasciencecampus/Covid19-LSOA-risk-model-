## This script

# import modules
import sys
import os

# import from local data files
current_path = os.path.abspath('.')
sys.path.append(os.path.dirname(current_path))

from data_access.data_factory import DataFactory as factory
from utils import config as cf
from utils import dashboard as dash

# import the LSOA level features
df_lsoa_features=factory.get('LSOA_feature_distribution').create_dataframe()

# import the coefficient size time series
df_coefs = factory.get('coefficients').create_dataframe()

# TEST REQUIREMENT - MAKE SURE THAT ONLY OBJECT AND FLOAT32 DTYPES EXIST IN THE DATAFRAME BEFORE WRITING TO BIGQUERY
# drop the 'Area' column - do this further back in the modelling process if possible

df_lsoa_features.drop('Area', axis='columns', inplace=True)

# compute quintiles for plotting
df_quint = dash.make_quintiles(df_lsoa_features)

# encode travel clusters as integers for data studio plotting
df_encoded = dash.encode_column(df_quint, 'travel_cluster')

# pivot the data set
df_pivot = dash.pivot_results(df_encoded)

# pretty names
df = dash.pretty_rename(df_pivot, 'feature', cf.feature_pretty_names)
df = dash.pretty_rename(df, 'travel_cluster', cf.tc_pretty_names)

# write to BigQuery
#df.to_gbq('wip.multi_grp_pred_static_vars_incl_fa', project_id='ons-hotspot-prod',if_exists='replace')

print(df.head())
