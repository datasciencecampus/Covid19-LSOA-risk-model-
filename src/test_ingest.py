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
from utils import model as md

from functools import reduce

### INGESTS ####

# static variables
print("static")
static_df = factory.get('static_vars_for_modelling').create_dataframe()

print("area")
# LSOA Area information to normalise footfall
area_df = factory.get('static_subset_for_norm').create_dataframe()

print("cases")
# Cases data
cases_df = factory.get('aggregated_tests_lsoa').create_dataframe()

print("mobility")
# Mobility data
deimos_footfall_df = factory.get('lsoa_daily_footfall').create_dataframe()