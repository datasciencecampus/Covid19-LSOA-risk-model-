# Import Packages
import os
import sys

# Import from local data files
current_path = os.path.abspath('.')
sys.path.append(os.path.dirname(current_path))

from data_access import two_way_model as twm

str_coef_tc_static, str_coef_tc_static_ci = twm.static_model()

str_coef_tc_dynamic, str_coef_tc_dynamic_ci = twm.dynamic_model(str_coef_tc_static, str_coef_tc_static_ci)

