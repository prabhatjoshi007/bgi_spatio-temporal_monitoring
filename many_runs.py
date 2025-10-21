# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:05:00 2025

@author: joshipra
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from datetime import datetime, timedelta
from pyswmm import Simulation, Output, SystemSeries
import glob
import math
import random

# Define reusable functions

def weibull_pdf(q, lambda_j, k_j):
    return (k_j / lambda_j) * (q / lambda_j) ** (k_j - 1) * np.exp(- (q / lambda_j) ** k_j)

def compute_cumulative_pdfs(pdfs, q, delta):
    cumulative_pdfs = [pdfs[0]]
    cumulative_pdf = pdfs[0]
    for pdf in pdfs[1:]:
        cumulative_pdf = fftconvolve(cumulative_pdf, pdf, mode="full")[:len(q)] * delta
        cumulative_pdfs.append(cumulative_pdf)
    return cumulative_pdfs

def compute_cumulative_survivals(cumulative_pdfs, q, delta):
    return [1 - np.cumsum(cum_pdf) * delta for cum_pdf in cumulative_pdfs]

def compute_transition_matrices(num_states, num_years, lambdas, k):
    q = np.linspace(0, num_years, num_years * 365)
    delta = q[1] - q[0]
    pdfs = [weibull_pdf(q, lambdas[j], k[j]) for j in range(num_states)]
    cumulative_pdfs = compute_cumulative_pdfs(pdfs, q, delta)
    cumulative_survivals = compute_cumulative_survivals(cumulative_pdfs, q, delta)
    
    transition_matrices = []
    for x in range(num_years * 365):
        P = np.zeros((num_states + 1, num_states + 1))
        for j in range(num_states):
            num = cumulative_pdfs[j][x]
            den = (cumulative_survivals[j][x] if j == 0 else cumulative_survivals[j][x] - cumulative_survivals[j - 1][x])
            if den > 0:
                P[j, j + 1] = min(num * delta / den, 1)
                P[j, j] = 1 - P[j, j + 1]
            else:
                P[j, j + 1] = 1
                P[j, j] = 0
        P[-1, -1] = 1
        transition_matrices.append(P)
    return transition_matrices

def run_simulation(model_name, start_date, end_date, porosity, K_sat):
    with open(model_name, 'r') as file:
        lines = file.readlines()

    with open(model_name, 'w') as file:
        in_lid_controls = False
        for line in lines:
            if line.startswith('START_DATE'):
                file.write(f"START_DATE            {start_date}\n")
            elif line.startswith("END_DATE"):
                file.write(f"END_DATE             {end_date}\n")
            elif line.startswith("REPORT_START_DATE"):
                file.write(f"REPORT_START_DATE     {start_date}\n")
            elif '[LID_CONTROLS]' in line:
                in_lid_controls = True
                file.write(line)
            elif in_lid_controls and line.strip().startswith('[END_LID_CONTROLS]'):
                in_lid_controls = False
                file.write(line)
            elif in_lid_controls and 'SOIL' in line:
                parts = line.split()
                if len(parts) >= 7:
                    parts[3] = str(porosity)
                    parts[6] = str(K_sat)
                file.write(' '.join(parts) + '\n')
            else:
                file.write(line)

def process_flood_files(output_dir):
    flood_files = glob.glob(os.path.join(output_dir, "flood_*.csv"))
    merged_df = pd.concat([pd.read_csv(file) for file in flood_files], ignore_index=True)
    if 'Time' in merged_df.columns:
        merged_df['Time'] = pd.to_datetime(merged_df['Time'])
        merged_df = merged_df.sort_values(by='Time')
    merged_df.to_csv(os.path.join(output_dir, "merged_flood_data.csv"), index=False)
    print("Merged flood data saved.")
    
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
        x1 = random.uniform(3.0, 5.0)  # Random x1 in the range [1.0, 5.0]
        x2 = x1 + random.uniform(2.0, 3.0)  # Random x2 in the range [x1 + 1.0, x1 + 3.0]
        
        denominator = math.log(x2) - math.log(x1)
        result = numerator / denominator
        
        # Calculate additional value
        additional_value = x1 / (ln_negative_half ** (1 / result))
        
        k.append(result)
        
        lambdas.append(additional_value)       
        
      
    return k, lambdas

def main_simulation(base_dir, main_folder, files_to_copy, num_iterations, num_states, num_years):
    for i_iter in range(1, num_iterations + 1):
        folder_path = os.path.join(base_dir, f"Simulation_{i_iter}")
        os.makedirs(folder_path, exist_ok=True)
        os.chdir(folder_path)
        for file_name in files_to_copy:
            shutil.copy(os.path.join(main_folder, file_name), folder_path)
        k, lambdas = compute_k_lamda(num_states=4)
        df = pd.DataFrame({
            "k": k,
            "lambdas": lambdas
        })
        
        # Export to CSV
        df.to_csv("k_and_lambdas.csv", index=False)
        print("CSV file 'k_and_lambdas.csv' saved successfully.")
        transition_matrices = compute_transition_matrices(num_states, num_years, lambdas, k)
        print(f"Transition matrices computed for iteration {i_iter}.")

        for i_cs in range(1, 6):
            inp_file = os.path.join(folder_path, f"basicModel_CS{i_cs}.inp")
            rpt_file = os.path.join(folder_path, f"basicModel_CS{i_cs}.rpt")
            out_file = os.path.join(folder_path, f"basicModel_CS{i_cs}.out")

            if os.path.exists(inp_file):
                with Simulation(inputfile=inp_file, reportfile=rpt_file, outputfile=out_file) as sim:
                    for step in sim:
                        pass
                with Output(out_file) as out:
                    ts11 = SystemSeries(out).flood_losses
                    pd.DataFrame(list(ts11.items()), columns=["Time", "Flood_Volume"]).to_csv(
                        os.path.join(folder_path, f"flood_{i_cs}.csv"), index=False
                    )
    process_flood_files(folder_path)

# Set paths and parameters
base_dir = r"C:\Users\joshipra\switchdrive\BGI_simulator\04_workspace\SWMM_model"
main_folder = base_dir
files_to_copy = [f"basicModel_CS{i}.inp" for i in range(1, 6)]
main_simulation(base_dir, main_folder, files_to_copy, num_iterations=3, num_states=4, num_years=25)
