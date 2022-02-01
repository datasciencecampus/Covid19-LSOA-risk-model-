# Import Packages
import os
import sys

# Import from local data files
current_path = os.path.abspath('.')
sys.path.append(os.path.dirname(current_path))

from data_access import model_wrappers as mw
from utils import config as cf

# for the purposes of testing, separate the two out using the config file
# when ready for production, remove this distinction. We want to run both models at the same time.

if cf.model_type == "twfe":
    
    str_coef_tc_static, str_coef_tc_static_ci = mw.static_model()

    str_coef_tc_dynamic, str_coef_tc_dynamic_ci = mw.dynamic_model(str_coef_tc_static, str_coef_tc_static_ci)
    
else:
    
    # do the time tranches model
    str_coef_tc_static, str_se_coef_tc_static, str_non_se_coef_tc_static = mw.tranches_model()

