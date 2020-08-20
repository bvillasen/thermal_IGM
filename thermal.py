import os, sys
import numpy as np

thermal_dir = os.getcwd()
subDirectories = [x[0] for x in os.walk(thermal_dir)]
sys.path.extend(subDirectories)
from tools import *
from cosmology import Cosmology


# Initialize Cosmology
cosmo = Cosmology()