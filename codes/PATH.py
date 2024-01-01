# --------------------------------------REVISIONS---------------------------------------
# Date        Name        Ver#    Description
# --------------------------------------------------------------------------------------
# 08/15/21    J. Li       [1]     Initial creation
# **************************************************************************************

"""
create global variable across files.
"""

import os


# Absolute path of the repo
# global package_directory
# global inputs_path
# global results_path

package_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
inputs_path = os.path.join(os.path.dirname(package_directory), 'datasets/video_asl/inputs')
results_path = os.path.join(os.path.dirname(package_directory), 'datasets/video_asl/results')
