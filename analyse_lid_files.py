# -*- coding: utf-8 -*-
"""
Created on Fri May  2 17:09:08 2025

@author: joshipra
"""

import os
import pandas as pd
import numpy as np
from glob import glob
import re

# --- CONFIG ---
base_dir = r"Q:\Abteilungsprojekte\eng\SWWData\Prabhat_Joshi\PhD\BGI_simulator\04_workspace\SWMM_model" 
num_simulations = 3  # Number of folders Simulation_1 to Simulation_j
file_pattern = "LID_S2_Tr*.txt"  # Match all LID time series files

def extract_number(f):
    match = re.search(r'Tr(\d+)', os.path.basename(f))
    return int(match.group(1)) if match else -1  # fallback for safety

# --- MASTER STORAGE ---


for sim_id in range(0, num_simulations + 1):
    all_dfs = []
    folder = os.path.join(base_dir, f"Simulation_{sim_id}")
    if not os.path.exists(folder):
        print(f"Skipping missing folder: {folder}")
        continue

    print(f"Processing: {folder}")
    
    file_paths = glob(os.path.join(folder, file_pattern))
    file_paths2 = sorted(file_paths, key=extract_number)
    
    # OPTIONAL: preview sorted files
    print("  Files:", [os.path.basename(f) for f in file_paths2])
    
    #file_paths = glob(os.path.join(folder, files_sorted))
    
    for file_path in file_paths2:
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            start_index = 0
            for i, line in enumerate(lines):
                if line.startswith('Date        Time'):
                    start_index = i
                    break

            header = lines[start_index].strip().split()
            data_lines = lines[start_index + 2:]
            data = [line.strip().split() for line in data_lines if line.strip()]

            df = pd.DataFrame(data, columns=header)
            df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])           
            
            df.set_index('Datetime', inplace=True)
            df.drop(columns=['Date', 'Time'], inplace=True)

            df.columns = ['Elapsed_hours', 'Total_inflow_mmh-1', 'Total_evap_mmh-1', 'Surf_inf_mmh-1', 'Pav_perc_mmh-1',
                          'Soil_perc_mmh-1', 'Sto_exf_mmh-1', 'Surf_runoff_mmh-1', 'Drain_outflow_mmh-1',
                          'Surf_level_mm', 'Pav_level_mm', 'Soil_moisture_mm', 'Sto_level_mm']

            full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='10T')
            df = df.reindex(full_index).sort_index()
            df.index.name = 'Datetime'
            df['Soil_moisture_mm'] = df['Soil_moisture_mm'].fillna(method='ffill')
            df.loc[:, df.columns != 'Soil_moisture_mm'] = df.loc[:, df.columns != 'Soil_moisture_mm'].fillna(0)

            df['Simulation'] = sim_id
            df['SourceFile'] = os.path.basename(file_path)

            all_dfs.append(df)

        except Exception as e:
            print(f"Could not read {file_path}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, axis=0)
        full_index = pd.date_range(start=combined_df.index.min(), end=combined_df.index.max(), freq='10T')
        combined_df = combined_df.reindex(full_index).sort_index()
        combined_df.index.name = 'Datetime'
        combined_df['Soil_moisture_mm'] = combined_df['Soil_moisture_mm'].fillna(method='ffill')
        combined_df.loc[:, combined_df.columns != 'Soil_moisture_mm'] = combined_df.loc[:, df.columns != 'Soil_moisture_mm'].fillna(0)

        #combined_df['Simulation'] = sim_id
        #combined_df['SourceFile'] = os.path.basename(file_path)              
               
        
        output_path = os.path.join(folder, "all_lid_timeseries_combined.csv")
        combined_df.to_csv(output_path)
        print(f"Combined data saved to: {output_path}")
    else:
        print("No valid data found.")