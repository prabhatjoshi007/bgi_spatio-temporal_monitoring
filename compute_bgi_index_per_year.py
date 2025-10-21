# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 13:29:10 2025

@author: joshipra
"""

import numpy as np
import pandas as pd

def compute_bgi_index_per_year(n_bgis, n_years, weights, current_states, max_state):
    """
    Compute BGI Index per year using weighted normalized condition states.

    Parameters:
        n_bgis (int): Number of BGIs
        n_years (int): Number of years
        weights (ndarray): Array of shape (n_years, n_bgis) with BGI weights
        current_states (ndarray): Array of shape (n_years, n_bgis) with condition states
        max_state (int): Maximum condition state (e.g., 5)

    Returns:
        pd.Series: BGI index per year (values between 0 and 1)
    """

    # Ensure arrays are numpy arrays
    weights = np.array(weights)
    current_states = np.array(current_states)

    # Normalized condition performance (1 = new, 0 = worst)
    normalized_perf = 1 - (current_states - 1) / (max_state - 1)

    # Weighted sum per year
    bgi_index = np.sum(weights * normalized_perf, axis=1)

    # Convert to pandas Series with year index
    years = np.arange(1, n_years + 1)
    return pd.Series(bgi_index, index=years, name="BGI_Index")
