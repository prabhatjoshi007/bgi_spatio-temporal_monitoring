# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:48:33 2025

@author: joshipra
"""

import os
import sys
import csv
import shutil
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pyswmm import Simulation, Output, SystemSeries, NodeSeries
import glob

sys.path.append(r'C:\Users\joshipra\switchdrive\BGI_simulator\08_codes_and_functions')
from compute_k_lamda import compute_k_lamda
from Ks_to_por import Ks_to_por


#%%
def weibull_pdf(q, lambda_j, k_j):
    """Weibull probability density function."""
    return (k_j / lambda_j) * (q / lambda_j) ** (k_j - 1) * np.exp(- (q / lambda_j) ** k_j)

def weibull_survival(q, lambda_j, k_j):
    """Weibull survival function."""
    return np.exp(- (q / lambda_j) ** k_j)


def compute_cumulative_pdfs(pdfs, q, delta):
    """Compute cumulative PDFs using convolution."""
    cumulative_pdfs = [pdfs[0]]
    cumulative_pdf = pdfs[0]  # Start with the first state's PDF
    for pdf in pdfs[1:]:
        # Convolve the current cumulative PDF with the next state's PDF
        cumulative_pdf = fftconvolve(cumulative_pdf, pdf, mode="full")[:len(q)] * delta
        cumulative_pdfs.append(cumulative_pdf)
    return cumulative_pdfs

def compute_cumulative_survivals(cumulative_pdfs, q, delta):
    """Compute cumulative survival curves from cumulative PDFs."""
    cumulative_survivals = []
    for cum_pdf in cumulative_pdfs:
        cdf = np.cumsum(cum_pdf) * delta  # Cumulative Distribution Function
        survival = 1 - cdf  # Survival Function
        cumulative_survivals.append(survival)
    return cumulative_survivals


def compute_transition_probability_matrices(num_states, num_years, lambdas, k):
    """
    Compute transition probability matrices for each time step based on Eq 12,
    ensuring P[j, j] does not decrease across consecutive timesteps.

    Parameters:
    - num_states: Number of condition states.
    - num_years: Total number of years to compute.
    - lambdas: List of scale parameters for each state.
    - k: List of shape parameters for each state.

    Returns:
    - transition_matrices: List of transition matrices for each time step.
    """
    # Time vector
    q = np.linspace(0, num_years, num_years * 365)  # Daily timesteps
    delta = q[1] - q[0]  # Time step size

    # Compute PDFs and survival functions for each state
    pdfs = [weibull_pdf(q, lambdas[j], k[j]) for j in range(num_states)]
    #surv = [weibull_survival(q, lambdas[j], k[j]) for j in range(num_states)]
    cumulative_pdfs = compute_cumulative_pdfs(pdfs, q, delta)
    cumulative_survivals = compute_cumulative_survivals(cumulative_pdfs, q, delta)

    # Initialize list to hold transition matrices
    transition_matrices = []
    #previous_P = None  # To store the previous transition matrix

    # Compute transition matrices for each time step
    for x in range(num_years * 365):
        P = np.zeros((num_states + 1, num_states + 1))  # Transition matrix

        for j in range(num_states):
            if j == 0:  # First state
                num = np.round(cumulative_pdfs[j][x], 6)
                den = np.round(cumulative_survivals[j][x], 6)
                if den > 0:
                    P[j, j + 1] = num * delta/ den if num * delta / den < 1 else 1
                    P[j, j] = 1 - P[j, j + 1]
                else:
                    P[j, j + 1] = 1
                    P[j, j] = 0
            else:  # Subsequent states
                num = cumulative_pdfs[j][x]
                den = cumulative_survivals[j][x] - cumulative_survivals[j - 1][x]
                if den > 0:
                    P[j, j + 1] = num * delta / den if num *delta / den < 1 else 1
                    P[j, j] = 1 - P[j, j + 1]
                else:
                    P[j, j + 1] = 1
                    P[j, j] = 0

        # Final absorbing state
        P[-1, -1] = 1
        
        
        '''
        # Ensure P[j, j] does not decrease compared to the previous step
        if previous_P is not None:
            for j in range(num_states):
                if P[j, j] > previous_P[j, j]:
                    P[j, j] = previous_P[j, j]
                    P[j, j + 1] = 1 - P[j, j]
                    
                    '''                    

        # Save the current transition matrix and update the previous
        transition_matrices.append(P)
        #previous_P = P.copy()

    return transition_matrices

#%%
# Function to execute simulation for a given condition state
def run_simulation(model_name, start_date, end_date, porosity, K_sat):
    print(f"Simulating {model_name}")
    print(f"Start Date: {start_date}, End Date: {end_date}")
    
    with open(model_name, 'r') as file:
        lines = file.readlines()

    with open(model_name, 'w') as file:
        in_lid_controls = False  # Flag to track if inside LID_CONTROLS section

        for line in lines:
            # Modify the start and report dates
            if line.startswith('START_DATE'):
                file.write(f"START_DATE            {start_date}\n")
            elif line.startswith("END_DATE"):
                file.write(f"END_DATE             {end_date}\n")
            elif line.startswith("REPORT_START_DATE"):
                file.write(f"REPORT_START_DATE     {start_date}\n")
            elif line.startswith("START_TIME"):
                file.write("START_TIME            00:00:00\n")
            elif line.startswith("REPORT_START_TIME"):
                file.write("REPORT_START_TIME     00:00:00\n")
            
            # Check if inside the LID_CONTROLS section
            elif '[LID_CONTROLS]' in line:
                in_lid_controls = True
                file.write(line)
            
            elif in_lid_controls and line.strip().startswith('[END_LID_CONTROLS]'):
                in_lid_controls = False
                file.write(line)
            
            elif in_lid_controls and 'SOIL' in line:
                # Modify the porosity and K_sat in the SOIL line
                parts = line.split()
                if len(parts) >= 7:  # Ensure the line has enough fields
                    parts[3] = str(porosity)  # Assuming porosity is the 4th value
                    parts[6] = str(K_sat)     # Assuming K_sat is the 7th value
                file.write(' '.join(parts) + '\n')
            
            else:
                # Write the unchanged line
                file.write(line)

#%%

# Base directory
base_dir = r"C:\Users\joshipra\switchdrive\BGI_simulator\04_workspace\SWMM_model"

# Main folder containing the files
main_folder = r"C:\Users\joshipra\switchdrive\BGI_simulator\04_workspace\SWMM_model"

# Files to copy
files_to_copy = ["basicModel_catchment_CS1.inp", "basicModel_catchment_CS2.inp", "basicModel_catchment_CS3.inp", 
                 "basicModel_catchment_CS4.inp", "basicModel_catchment_CS5.inp"]

# List to store paths
simulation_paths = []

# Number of iterations
n_iter = 1

# Loop through iterations
for i_iter in range(1, n_iter + 1):
    folder_name = f"Simulation_{i_iter}"
    folder_path = os.path.join(base_dir, folder_name)
    
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    os.chdir(folder_path)
    
    for file_name in files_to_copy:
        src_file = os.path.join(main_folder, file_name)
        dest_file = os.path.join(folder_path, file_name)
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)
            
    #k, lambdas = compute_k_lamda(num_states = 4)
    k = [2.042, 2.042, 2.042, 2.042]
    lambdas = [2.393, 3.590, 4.188, 3.590]

    df = pd.DataFrame({
        "k": k,
        "lambdas": lambdas
    })
    
    # Export to CSV
    df.to_csv("k_and_lambdas.csv", index=False)
    print("CSV file 'k_and_lambdas.csv' saved successfully.")
    
    delta = 1/365              # Time step (in d)
    num_years = 25
    num_days = 365 * num_years         # 25 years of daily transitions
    num_states = 4
    q =  np.linspace(0, num_years, num_days)


    # Compute the transition matrices for each day ; (lambdas, k, delta, max_age, num_days)
    transition_matrices = compute_transition_probability_matrices(num_states, num_years, lambdas, k)

    pdfs = [weibull_pdf(q, lambdas[j], k[j]) for j in range(num_states)]
    cumulative_pdfs = compute_cumulative_pdfs(pdfs, q, delta)
    cumulative_survivals = compute_cumulative_survivals(cumulative_pdfs, q, delta)
    
    with open("cumulative_survivals.csv", mode = "w", newline = "") as ff:
        writer = csv.writer(ff)
        writer.writerows(cumulative_survivals)
        
    
    
    state_vector = np.array([1, 0, 0, 0, 0])  # Initial state


    # Transition matrices (list of matrices for each timestep)
    # Assume `transition_matrices` is already computed and contains matrices for each timestep
    # transition_matrices[i] is the transition matrix at timestep i

    # Storage for state vectors over time
    state_vectors = np.zeros((num_days, len(state_vector)))
    state_vectors[0] = state_vector
    state_progression = np.zeros(num_days, dtype=int)  # Tracks the actual state over time

    # Iterate through each timestep
    for t in range(1, num_days):
        #print("Old state vector:", state_vector)
        state_vector = np.dot(state_vector, transition_matrices[t])
        #print("New state vector:", state_vector)
        state_vectors[t] = state_vector
        
        
    # Step 1: Determine the most likely state for each day
    most_likely_states = np.argmax(state_vectors, axis=1)  # Get column index of max probability
    
    for tt in range(1, num_days):  # Use weeks instead of days if updated
        state_progression[0] = 0  # Initial state is always 0

        current_state = state_progression[tt - 1]  # Get previous state

    # Zero out probabilities of lower states
        allowed_probabilities = state_vectors[tt].copy()
        allowed_probabilities[:current_state] = 0  # Disallow lower states

    # Normalize the remaining probabilities to sum to 1
        if allowed_probabilities.sum() > 0:
            allowed_probabilities /= allowed_probabilities.sum()
        else:
            allowed_probabilities[current_state] = 1  # Stay in the current state if no valid transition
    
        # Sample next state from the adjusted probability distribution
        state_progression[tt] = np.random.choice(len(state_vector), p=allowed_probabilities)

    # Output the results
    print("Most Likely States for Each Day:", most_likely_states)
    


    change_indices = np.where(np.diff(most_likely_states) != 0)[0] + 1
    Ks_init = 300

    Ks = []
    por = []
    for tt in range(0, len(change_indices) + 1):
        if tt == 0:
            Ks.append(300)
            por.append(Ks_to_por(Ks[tt], fc=0.15, wp=0.08))
        elif tt == 1:
            Ks.append(np.random.uniform(high=0.500 * Ks_init, low=0.301 * Ks_init))
            por.append(Ks_to_por(Ks[tt], fc=0.15, wp=0.08))
        elif tt == 2:
            Ks.append(np.random.uniform(high=0.300 * Ks_init, low=0.201 * Ks_init))
            por.append(Ks_to_por(Ks[tt], fc=0.15, wp=0.08))
        elif tt == 3:
            Ks.append(np.random.uniform(high=0.200 * Ks_init, low=0.151 * Ks_init))
            por.append(Ks_to_por(Ks[tt], fc=0.15, wp=0.08))
        else:
            Ks.append(np.random.uniform(high=0.150 * Ks_init, low=0.1 * Ks_init))
            por.append(Ks_to_por(Ks[tt], fc=0.15, wp=0.08))

    
    start_date = datetime(1985, 1, 1)
    x = len(most_likely_states)  # Replace with your desired number of days

    # Generate a list of dates in the specified format
    dates = [(start_date + timedelta(days=ijk)).strftime('%m/%d/%Y %H:%M') for ijk in range(x)]

    # Print the formatted dates
    change_date = []
    for date in change_indices:
        change_date.append(dates[date])
        print(change_date)
        
    a = np.vstack((Ks, por))
    a = a.T
    a_str = a.astype(str)
    tm = np.hstack(('01/01/1985 00:00', change_date))
    tm_reshaped = tm.reshape(-1, 1)
    k_por_joined = np.hstack((tm_reshaped, a_str))
    
    with open('k_por_joined.csv', mode='w', newline='', encoding='utf-8') as file2:
        writer = csv.writer(file2)
        writer.writerows(k_por_joined)
    

    # Loop through the condition states
    #models = ['basicModel_CS1.inp', 'basicModel_CS2.inp', 'basicModel_CS3.inp', 'basicModel_CS4.inp', 'basicModel_CS5.inp']
    models = files_to_copy
    for ii in range(len(por)):
        print(f"Running condition state: CS {ii + 1}")
        
        if ii == 0:
            start_date = (datetime.strptime('01/01/1985 00:00', '%m/%d/%Y %H:%M')).strftime('%m/%d/%Y %H:%M')
        else:
            start_date = change_date[ii - 1]
        
        if ii < len(change_date):
            end_date = (datetime.strptime(change_date[ii], '%m/%d/%Y %H:%M') - timedelta(minutes=10)).strftime('%m/%d/%Y %H:%M')
        else:
            # For the last state, set a fixed end date
            end_date = (datetime.strptime('12/31/2009 23:59', '%m/%d/%Y %H:%M')).strftime('%m/%d/%Y %H:%M')
        
        # Run the simulation
        run_simulation(models[ii], start_date, end_date, por[ii], Ks[ii])

    # Define the base name for the SWMM input files
    base_name = "basicModel_catchment_CS"
    report_name = "rpt_basicModel_catchment_CS"

    # Loop through the files and run simulations
    for i_cs in range(1,6):
        inp_file = f"{base_name}{i_cs}.inp"
        rpt_file = f"{report_name}{i_cs}.rpt"
        out_file = f"{base_name}{i_cs}.out"  # Default output file
        
    
        if os.path.exists(inp_file):
            with Simulation(inputfile=inp_file, reportfile=rpt_file, outputfile=out_file) as sim:
                print(f"Running simulation for {inp_file}...")
                
                
                
                for step in sim:
                    pass
                
            with Output(out_file) as out:
                ts11 = SystemSeries(out).flood_losses
                ts11_df = pd.DataFrame(list(ts11.items()), columns=["Time", "Flood_Volume"])
                ts11_df.to_csv(f"catchment_flood_{i_cs}.csv")
                #print(ts11)
                
                
                ts12 = NodeSeries(out)['CSO'].total_inflow
                ts12_df = pd.DataFrame(list(ts12.items()), columns=["Time", "Outfall_Volume"])
                ts12_df.to_csv(f"catchment_outfall_{i_cs}.csv")
                
            print(f"Simulation for {inp_file} completed.")
            
            # Check and remove the .out file if it exists
            if os.path.exists(out_file):
                os.remove(out_file)
                print(f"Removed .out file: {out_file}")
        else:
            print(f"File {inp_file} not found.")
            
        # #%% Remove the out file
        
        # for i_cs in range(1, 6):
        #     out_file = f"{base_name}{i_cs}.out"  # Default output file
            
        #     if os.path.exists(out_file):
        #         os.remove(out_file)
        #         print(f"Removed .out file: {out_file}")  
            
#%%            
# Define the file pattern for all five files
for i_iter in range(1, n_iter + 1):
    folder_name = f"Simulation_{i_iter}"
    folder_path = os.path.join(base_dir, folder_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    os.chdir(folder_path)
    
    start_time = "01/01/1985 00:00"
    end_time = "12/31/2009 23:59"
    full_datetime_index_full = pd.date_range(start=start_time, end=end_time, freq="10min")
    file_pattern = "basicModel_catchment_CS[1-5].txt"
    
    # List all files matching the pattern
    file_list = glob.glob(file_pattern)
    
    # Initialize an empty list to store processed DataFrames
    processed_dataframes = []
    merged_data = []
    merged_data_full = []
    
    # Process each file
    for file_path in file_list:
        print(f"Processing file: {file_path}")
        
        # Read the file content
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Find the start index of the table
        start_index = 0
        for iii, line in enumerate(lines):
            if line.startswith('Date        Time'):
                start_index = iii
                break
    
        # Extract the header and data lines
        header = lines[start_index].strip().split()
        data_lines = lines[start_index + 2:]
    
        # Process the data lines
        data = []
        for line in data_lines:
            if line.strip():
                data.append(line.strip().split())
    
        # Create a DataFrame
        df = pd.DataFrame(data, columns=header)
    
        # Convert relevant columns to numeric types
        df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    
        # Combine the 'Date' and 'Time' columns into a single datetime column
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
        # Drop the original 'Date' and 'Time' columns
        df.drop(columns=['Date', 'Time'], inplace=True)
    
        # Rename columns
        df.columns = [
            'Elasped_hours', 'Total_inflow_mmh-1', 'Total_evap_mmh-1', 
            'Surf_inf_mmh-1', 'Pav_perc_mmh-1', 'Soil_perc_mmh-1', 
            'Sto_exf_mmh-1', 'Surf_runoff_mmh-1', 'Drain_outflow_mmh-1', 
            'Surf_level_mm', 'Pav_level_mm', 'Soil_moisture_mm', 'Sto_level_mm', 'Datetime'
        ]
    
        # Set the 'Datetime' column as the index
        df.set_index('Datetime', inplace=True)
    
        # Fill missing timesteps
        full_datetime_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='10min')
        df = df.reindex(full_datetime_index).fillna(0)
        df = df.rename_axis('Datetime')    
        # Add a source column to indicate the file source
        #df['source_file'] = file_path.split('/')[-1]
    
        # Append the processed DataFrame
        processed_dataframes.append(df)
    
    # Concatenate all processed DataFrames
    merged_data = pd.concat(processed_dataframes)
    merged_data = merged_data[~merged_data.index.duplicated(keep='first')]
    merged_data_full = merged_data.reindex(full_datetime_index_full).fillna(0)
    merged_data_full = merged_data_full.rename_axis('Datetime')
    
    # Save the merged DataFrame to a file
    output_name = f"merged_completed_dataset_{i_iter}.csv"
    merged_data_full.to_csv(output_name, index = True)
    
    print("All files processed, missing timesteps filled, and merged dataset saved to:", output_name)

    
#%%
for i_iter in range(1, n_iter + 1):
    folder_name = f"Simulation_{i_iter}"
    folder_path = os.path.join(base_dir, folder_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    os.chdir(folder_path)
    # Define the file pattern for all five files
    file_pattern_flood = "catchment_flood_[1-5].csv"
    
    # List all files matching the pattern
    file_list_flood = glob.glob(file_pattern_flood)
    
    # Initialize an empty list to store processed DataFrames
    merged_df = pd.DataFrame()
    
    # Process each file
    for file in file_list_flood:
    # Read each file
        temp_df = pd.read_csv(file)
        
    
    # Optionally, keep track of the source file
        temp_df['Source_File'] = file
        #plt.plot(temp_df['Time'], temp_df['Flood_Volume'])
    
    # Append to the merged DataFrame
        merged_df = pd.concat([merged_df, temp_df], ignore_index=True)

    # Ensure 'Time' column is in datetime format and sort by it
    if 'Time' in merged_df.columns:
        merged_df['Time'] = pd.to_datetime(merged_df['Time'])
        merged_df = merged_df.sort_values(by='Time')
        merged_df = merged_df[['Time', 'Flood_Volume']]
        #merged_df2 = merged_df.reindex(full_datetime_index_full, fill_value = 0)
        
        
    # Save the merged data to a new CSV file
    output_file = f"merged_flood_data_{i_iter}.csv"
    merged_df = merged_df[['Time', 'Flood_Volume']]
    #merged_df = merged_df.reindex(full_datetime_index_full).fillna(0)
    merged_df.to_csv(output_file, index=False)

    print(f"Merged data has been saved to: {output_file}")
      
    plt.plot(merged_df['Time'], merged_df['Flood_Volume'])
    
    
#%%
for i_iter in range(1, n_iter + 1):
    folder_name = f"Simulation_{i_iter}"
    folder_path = os.path.join(base_dir, folder_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    os.chdir(folder_path)
    # Define the file pattern for all five files
    file_pattern_outfall = "catchment_outfall_[1-5].csv"
    
    # List all files matching the pattern
    file_list_outfall = glob.glob(file_pattern_outfall)
    
    # Initialize an empty list to store processed DataFrames
    merged_df_outfall = pd.DataFrame()
    
    # Process each file
    for file in file_list_outfall:
    # Read each file
        temp_df_2 = pd.read_csv(file)
    
    # Optionally, keep track of the source file
        temp_df_2['Source_File'] = file  
    
    # Append to the merged DataFrame
        merged_df_outfall = pd.concat([merged_df_outfall, temp_df_2], ignore_index=True)

    # Ensure 'Time' column is in datetime format and sort by it
    if 'Time' in merged_df_outfall.columns:
        merged_df_outfall['Time'] = pd.to_datetime(merged_df_outfall['Time'])
        merged_df_outfall = merged_df_outfall.sort_values(by='Time')
    
    # Save the merged data to a new CSV file
    output_file_2 = f"merged_outfall_data_{i_iter}.csv"
    merged_df_outfall = merged_df_outfall[['Time', 'Outfall_Volume']]
    merged_df_outfall.to_csv(output_file_2, index=False)

    print(f"Merged data has been saved to: {output_file_2}")
    
    plt.plot(merged_df_outfall['Time'], merged_df_outfall['Outfall_Volume'])
