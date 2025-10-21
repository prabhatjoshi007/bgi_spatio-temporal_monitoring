# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:49:07 2025

@author: joshipra
"""
import numpy as np

def por_to_Ks(por, fc, wp):
    """
    Convert porosity to sat. hydraulic conductivity [mm.h-1].

    Parameters:
    - por: soil porosity
    - fc: field capacity
    - wp: wilting point
    """
    lambda1 = 0.262 * np.log(fc/wp)
    Ks = 76 * (por - fc)**(3 - lambda1) * 25.4
    return Ks
    
   