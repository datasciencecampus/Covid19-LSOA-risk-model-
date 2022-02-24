import os
import sys

# add local path to allow module imports
current_path = os.path.abspath('.')
sys.path.append(os.path.dirname(current_path))

from data_access import model_wrappers as mw
from utils import config as cf

if cf.model_type == "two_way_fixed_effects":
    
    str_coef_tc_static, str_coef_tc_static_ci = mw.static_model()

    str_coef_tc_dynamic, str_coef_tc_dynamic_ci = mw.dynamic_model(str_coef_tc_static, str_coef_tc_static_ci)
    
else:
    
    str_coef_tc_static, str_se_coef_tc_static, str_non_se_coef_tc_static, str_pred_tc_static, pred_latest = mw.tranches_model()

