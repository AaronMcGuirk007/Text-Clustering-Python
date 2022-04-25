import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:24:15 2022

@author: Aaron-PC
"""

"""
Project2-AaronMcGuirk.py

Created on Sun Feb 20 22:26:15 2022

Text clustering HW#2 for CSIS320

@author: Aaron McGuirk
@version: Spring 2022
"""


n_row, n_col = 1, 4
data = np.loadtxt("descriptions.txt", dtype="str", delimiter="\t", skiprows=1)