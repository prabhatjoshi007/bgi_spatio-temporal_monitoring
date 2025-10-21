# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:49:07 2025

@author: joshipra
"""
import numpy as np

def Ks_to_por(Ks, fc, wp):
    """
    Convert sat. hydraulic conductivity [mm.h-1] to porosity [-].

    Parameters:
    - Ks: sat. hydraulic conductivity [mm.h-1]
    - fc: field capacity
    - wp: wilting point
    """
    lambda1 = 0.262 * np.log(fc/wp)
    calc = Ks/25.4 # conversion to in.h-1
    por = ((calc/76)**(1/(3 - lambda1))) + fc    
    return por
    
   