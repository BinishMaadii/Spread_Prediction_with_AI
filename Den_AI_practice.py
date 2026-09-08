import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import statsmodels.api as sm 


import statsmodels.formula.api as smf
import lightgbm as lgb


### Defining some parameters behind it ###

ROW_PER_CITY = 250 

LAG_WEEKS = [1,2,4] # I will try with [1,2,4,8]


