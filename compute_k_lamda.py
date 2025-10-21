# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math
import random

def compute_k_lamda(num_states=4):
    """
    Computes the formula for multiple randomly generated (x1, x2) pairs
    and calculates an additional formula for each state:
    value = x1 / (-ln(0.5))^(1/result)
    
    Args:
    num_states (int): Number of random (x1, x2) pairs to generate. Default is 4.
    
    Returns:
    list: A list of dictionaries containing x1, x2, result, and the additional value.
    """
    k = []
    lambdas = []
    numerator = math.log(-math.log(0.2)) - math.log(-math.log(0.5))
    ln_negative_half = -math.log(0.5)
    
    for _ in range(num_states):
        x1 = random.uniform(2.0, 5.0)  # Random x1 in the range [2.0, 5.0]
        x2 = x1 + random.uniform(2.0, 3.0)  # Random x2 in the range [x1 + 1.0, x1 + 3.0]
        
        denominator = math.log(x2) - math.log(x1)
        result = numerator / denominator
        
        # Calculate additional value
        additional_value = x1 / (ln_negative_half ** (1 / result))
        
        k.append(result)
        
        lambdas.append(additional_value)
        
        
      
    return k, lambdas




