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
from pyswmm import Simulation, Output, SystemSeries, NodeSeries, LidGroups, Subcatchments
import glob

sys.path.append(r'C:\Users\joshipra\switchdrive\BGI_simulator\08_codes_and_functions')
from compute_k_lamda import compute_k_lamda
from Ks_to_por import Ks_to_por


#%%
def weibull_pdf(q, lambda_j, k_j):
    """Weibull probability density function. lambda_j is the scale parameter (eta); k_j is the shape parameter (beta)"""
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
    # Time vector
    q = np.linspace(0, num_years, num_years * 365)  # Daily timesteps
    delta = num_years / (num_years * 365)  # Time step size

    # Compute PDFs and survival functions for each state
    pdfs = [weibull_pdf(q, lambdas[j], k[j]) for j in range(num_states)]
    cumulative_pdfs = compute_cumulative_pdfs(pdfs, q, delta)
    cumulative_survivals = compute_cumulative_survivals(cumulative_pdfs, q, delta)

    # Initialize list to hold transition matrices
    transition_matrices = []

    # Compute transition matrices for each time step
    for x in range(num_years * 365):
        P = np.zeros((num_states + 1, num_states + 1))  # Transition matrix

        for j in range(num_states):
            total_prob = 0
            remaining_survival = cumulative_survivals[j][x] if j == 0 else cumulative_survivals[j][x] - cumulative_survivals[j - 1][x]
            
            for next_state in range(j + 1, num_states + 1):
                num = cumulative_pdfs[next_state - 1][x]
                den = remaining_survival
                if den > 0 and num > 0:
                    P[j, next_state] = min(1 - total_prob, num * delta / den)
                    total_prob += P[j, next_state]
            
            # Probability of remaining in state j
            P[j, j] = max(0, 1 - total_prob)

        # Final absorbing state remains unchanged
        P[-1, -1] = 1

        transition_matrices.append(P)

    return transition_matrices

def plot_cumulative_survivals(cumulative_survivals, q):
    """
    Plot cumulative survival functions for all condition states.

    Parameters:
    - cumulative_survivals: List of cumulative survival functions for all condition states.
    - q: Time vector (discretized time steps).
    """
    plt.figure(figsize=(8, 6))
    for idx, survival in enumerate(cumulative_survivals):
        plt.plot(q, survival, label=f'Survival {idx + 1}')
    plt.xlabel('Time [years]')
    plt.ylabel('Survival Probability')
    plt.title('Cumulative Survival Functions for Condition States')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_cumulative_pdfs(cumulative_pdfs, q):
    """
    Plot cumulative PDFs for all condition states.

    Parameters:
    - cumulative_pdfs: List of cumulative PDFs for all condition states.
    - q: Time vector (discretized time steps).
    """
    plt.figure(figsize=(8, 6))
    for idx, cum_pdf in enumerate(cumulative_pdfs):
        plt.plot(q, cum_pdf, label=f'Cumulative PDF {idx + 1}')
    plt.xlabel('Time [years]')
    plt.ylabel('Probability Density')
    plt.title('Cumulative PDFs for Condition States')
    plt.legend()
    plt.grid(True)
    plt.show()

#%%
def edit_inpfile(model_name, start_date, end_date, save_hotstart, use_hotstart,
                                    new_lid_process, iteration, start_line=109, end_line=109, update_row_index=0):
    """
    Updates simulation parameters and, in a specified block of lines (start_line to end_line),
    updates the "RptFile" column for all rows and updates the "LID Process" column only for one row.

    For each row in the block, the "RptFile" column is set to:
        "LID_{subcatchment}_Tr{iteration}"
    where subcatchment is the first token on the line.

    Parameters:
      model_name (str): The input file name.
      start_date (str): New simulation start date.
      end_date (str): New simulation end date.
      save_hotstart (str): New value for SAVE HOTSTART.
      use_hotstart (str): New value for USE HOTSTART.
      new_lid_process (str): New LID Process value to be applied only to one row in the block.
      iteration (int or str): The iteration number used to form the RptFile field.
      start_line (int): The starting line number of the block (1-indexed; default 118).
      end_line (int): The ending line number of the block (1-indexed; default 121).
      update_row_index (int): Which row (0-indexed relative to start_line) gets its LID Process updated.
                                For example, 0 means only update the LID Process in line 118.
    """
    print(f"Editing {model_name}")
    print(f"Simulation Start Date: {start_date}, End Date: {end_date}, Iteration: {iteration}")

    # Read all lines from the file
    with open(model_name, 'r') as file:
        lines = file.readlines()

    # Process and update the lines
    with open(model_name, 'w') as file:
        for i, line in enumerate(lines):
            # Update simulation parameters based on keywords
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
            elif line.startswith("SAVE HOTSTART"):
                file.write(f"SAVE HOTSTART            {save_hotstart}\n")
            elif line.startswith("USE HOTSTART"):
                file.write(f"USE HOTSTART             {use_hotstart}\n")
            elif '[FILES]' in line:
                file.write(line)
            # Update lines in the given block (assumed to be part of LID_USAGE)
            elif start_line - 1 <= i < end_line:
                tokens = line.split()
                if len(tokens) >= 9:
                    # Compute the new RptFile field based on the subcatchment name (first token)
                    subcatchment = tokens[0]
                    tokens[8] = f'"LID_{subcatchment}_Tr{iteration}.txt"'
                    # Only update the "LID Process" column (2nd token) for the specified row
                    if i == start_line - 1 + update_row_index:
                        tokens[1] = new_lid_process
                    file.write(' '.join(tokens) + '\n')
                else:
                    print(f"Warning: Line {i+1} does not have enough columns to update.")
                    file.write(line)
            else:
                # Write any other lines unchanged.
                file.write(line)

    return update_row_index, new_lid_process

#%%
# ----- Parameters -----
n_iter = 10             # Number of iterations (simulation runs)
n_lid = 1              # Number of subcatchments with LIDs per iteration
base_dir = r"C:\Users\joshipra\switchdrive\BGI_simulator\04_workspace\SWMM_model"
main_folder = r"C:\Users\joshipra\switchdrive\BGI_simulator\04_workspace\SWMM_model"
files_to_copy = ["basicModel_catchment.inp", "basicModel_catchment_Tr0.hsf"]

num_years = 25
num_days = 365 * num_years
delta = 1/365
num_states = 4
start_date = (datetime.strptime('01/01/1986 00:00', '%m/%d/%Y %H:%M'))

# Save k and lambdas to CSV
k = [2.643, 2.438, 2.769, 4.000]
lambdas = [4.595, 6.392, 3.995, 2.192]
#lambdas = [300, 597.5, 1195, 1493.751]
df = pd.DataFrame({"k": k, "lambdas": lambdas})
df.to_csv("k_and_lambdas.csv", index=False)
print("CSV file 'k_and_lambdas.csv' saved successfully.")

# Containers for overall results
all_iterations_results = {}
all_merged_change_dates = {}  # To store merged change dates per iteration

# ----- Outer Loop: Iterations -----
for i_iter in range(1, n_iter + 1):
    folder_name = f"Simulation_{i_iter}"
    folder_path = os.path.join(main_folder, folder_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    os.chdir(folder_path)
    
    # Copy necessary files
    for file_name in files_to_copy:
        src_file = os.path.join(main_folder, file_name)
        dest_file = os.path.join(folder_path, file_name)
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)           
    
    
    # Precompute common elements for simulation
    q = np.linspace(0, num_years, num_days)
    transition_matrices = compute_transition_probability_matrices(num_states, num_years, lambdas, k)
    pdfs = [weibull_pdf(q, lambdas[j], k[j]) for j in range(num_states)]
    cumulative_pdfs = compute_cumulative_pdfs(pdfs, q, delta)
    cumulative_survivals = compute_cumulative_survivals(cumulative_pdfs, q, delta)
    plot_cumulative_pdfs(cumulative_pdfs, q)
    plot_cumulative_survivals(cumulative_survivals, q)
    
    with open("cumulative_survivals.csv", mode="w", newline="") as ff:
        writer = csv.writer(ff)
        writer.writerows(cumulative_survivals)
    
    # ----- Inner Loop: Subcatchments -----
    # We'll store the state progression for each subcatchment in this list.
    subcatchment_results = []
    
    for sub in range(1, n_lid + 1):
        # Change random seed so that each subcatchment simulation is different
        np.random.seed(i_iter * 100 + sub)
        
        # Initial state vector (adjust size if needed)
        state_vector = np.array([1, 0, 0, 0, 0])
        state_vectors = np.zeros((num_days, len(state_vector)))
        state_vectors[0] = state_vector
        state_progression = np.zeros(num_days, dtype=int)
        
        # Iterate over each day to update state
        for t in range(1, num_days):
            state_vector = np.dot(state_vector, transition_matrices[t])
            state_vectors[t] = state_vector
            
            # Choose the next state based on the day's probabilities
            state_progression[t] = np.random.choice(len(state_vector), p=state_vectors[t])
            # Enforce non-decreasing state (if needed)
            if state_progression[t] < state_progression[t-1]:
                state_progression[t] = state_progression[t-1]
            
        # Append this subcatchment's daily condition state
        subcatchment_results.append(state_progression)
    
    # ----- Create Output Table -----
    # Generate date series (first column)
    dates = [(start_date + timedelta(days=ijk)).strftime('%m/%d/%Y %H:%M') for ijk in range(num_days)]
    df_output = pd.DataFrame({'Date': dates})
    
    
    # Add each subcatchment's condition state as a new column.
    for idx, states in enumerate(subcatchment_results, start=1):
        df_output[f'S{idx}'] = states
    
    # Save the combined table to a CSV file for this iteration.
    output_csv = f'iteration_{i_iter}_condition_states.csv'
    df_output.to_csv(output_csv, index=False)
    print(f"Iteration {i_iter}: Condition states for {n_lid} subcatchments saved in {output_csv}.")
    
    # Optionally, store in an overall container for later analysis.
    all_iterations_results[f'Iteration_{i_iter}'] = df_output.copy()
    
    # ----- Calculate Change Dates for Each Subcatchment -----
    change_info = {}

    for col in df_output.columns[1:]:
        # Compute the day-to-day difference
        diff = df_output[col].diff()
        # Identify indices where the condition state increases (by 1 or more)
        change_mask = diff >= 1
        # Get the corresponding dates and new condition state values
        dates_where_change = df_output.loc[change_mask, 'Date'].tolist()
        new_states = df_output.loc[change_mask, col].tolist()
        # Store both the date and the new condition state as tuples
        change_info[col] = list(zip(dates_where_change, new_states))
    
    # ----- Merge Change Dates Across All Subcatchments with Subcatchment Info -----
    # First, get all unique change dates (as strings)
    unique_dates = sorted(
        list({date for tuples in change_info.values() for (date, state) in tuples}),
        key=lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M')
        )

    
        # 2. For each unique date, determine which subcatchments changed and record their new state.
    merged_change_info = []
    for d in unique_dates:
        changed_info = []
        for subcatchment, changes in change_info.items():
            # Find if this subcatchment had a change on date d
            for (date_str, new_state) in changes:
                if date_str == d:
                    changed_info.append(f"{subcatchment}: {new_state}")
                    break
        merged_change_info.append({'Date': d, 'Subcatchments': ', '.join(changed_info)})
    
    # 3. Convert the merged change info to a DataFrame
    merged_change_df = pd.DataFrame(merged_change_info)
    merged_change_df["New_State"] = merged_change_df["Subcatchments"].str.extract(r'(\d+)$').astype(int)
    merged_change_df.index = merged_change_df.index + 1
    
    # Print results for the iteration
    print("Change info per subcatchment:")
    print(change_info)
    print("\nMerged change dates with subcatchment and condition state info (sorted):")
    print(merged_change_df)
    
    # Store merged change dates DataFrame for this iteration in the overall container
    all_merged_change_dates[f'Iteration_{i_iter}'] = merged_change_df.copy()  
    
    for jj in range(0,len(merged_change_df)+1):
        
        rpt_file = f"basicModel_catchment_{i_iter}_{jj+1}.rpt"
        out_file = f"basicModel_catchment_{i_iter}_{jj+1}.out"
        
        if jj == 0:
            st_date = start_date.strftime('%m/%d/%Y %H:%M')
            end_date = (datetime.strptime(merged_change_df.Date[jj+1], '%m/%d/%Y %H:%M') - timedelta(days = 1)).strftime('%m/%d/%Y %H:%M')
            save_hotstart = f"basicModel_catchment_Tr{jj+1}.hsf"
            use_hotstart = f"basicModel_catchment_Tr{jj}.hsf"
            lid_name = "BR_CS1"
            row_number = 0
                        
           
        elif jj == len(merged_change_df):
            st_date = (datetime.strptime(merged_change_df.Date[jj], '%m/%d/%Y %H:%M')).strftime('%m/%d/%Y %H:%M')
            end_date = "12/31/1999 23:50"
            save_hotstart = f"basicModel_catchment_Tr{jj+1}.hsf"
            use_hotstart = f"basicModel_catchment_Tr{jj}.hsf"
            lid_name = f"BR_CS{merged_change_df['New_State'][jj] + 1}"
            row_number = int(merged_change_df.Subcatchments[jj][-4])-1
           
            
        else:
            st_date = (datetime.strptime(merged_change_df.Date[jj], '%m/%d/%Y %H:%M')).strftime('%m/%d/%Y %H:%M')
            end_date = (datetime.strptime(merged_change_df.Date[jj+1], '%m/%d/%Y %H:%M') - timedelta(days = 1)).strftime('%m/%d/%Y %H:%M')
            save_hotstart = f"basicModel_catchment_Tr{jj+1}.hsf"
            use_hotstart = f"basicModel_catchment_Tr{jj}.hsf"
            lid_name = f"BR_CS{merged_change_df['New_State'][jj] + 1}"
            row_number = int(merged_change_df.Subcatchments[jj][-4])-1
            
        #it = jj+1
               
       
        xx, vv = edit_inpfile(model_name = "basicModel_catchment.inp", 
                     start_date = st_date, 
                     end_date = end_date, 
                     save_hotstart = save_hotstart, 
                     use_hotstart = use_hotstart,
                     new_lid_process = lid_name,
                     iteration = jj,
                     start_line = 109,
                     end_line = 109,
                     update_row_index = row_number)
        
        print(f"Subcatchment: S{xx+1}, LID: {vv}")
        
        with Simulation(inputfile="basicModel_catchment.inp", reportfile = rpt_file, outputfile = out_file) as sim:           
                                     
            for step in sim:
                pass    
                        
        rpt_file = f"basicModel_catchment_{i_iter}_{jj+1}.rpt"
        out_file = f"basicModel_catchment_{i_iter}_{jj+1}.out"
        
        with Output(out_file) as out:
            ts11 = SystemSeries(out).flood_losses
            ts11_df = pd.DataFrame(list(ts11.items()), columns=["Time", "Flood_Volume"])
            ts11_df.to_csv(f"catchment_flood_{jj+1}.csv", index = False)
            #print(ts11)
            
            
            ts12 = NodeSeries(out)['CSO_1'].total_inflow
            ts12_df = pd.DataFrame(list(ts12.items()), columns=["Time", "Outfall_Volume"])
            ts12_df.to_csv(f"catchment_outfall_{jj+1}.csv", index = False)
       
    
    if i_iter > 1:
        folder_name_prev = f"Simulation_{i_iter-1}"
        folder_path_prev = os.path.join(main_folder, folder_name_prev)
        
        os.chdir(folder_path_prev)
        
        for kk in range(1,17):            
               
            out_file_prev = f"basicModel_catchment_{i_iter-1}_{kk}.out"
            if os.path.exists(out_file_prev):
                os.remove(out_file_prev)
                print(f"Removed .out file: {out_file_prev}")   
    
    
    # Return to base directory before next iteration
    os.chdir(main_folder)



#%%

def process_catchment_data(main_folder, sim_count, file_prefix, value_column, output_filename):
    os.chdir(main_folder)

    for i_iter in range(1, sim_count + 1):
        print(f"Reading Simulation: {i_iter}")
        folder_name = f"Simulation_{i_iter}"
        folder_path = os.path.join(main_folder, folder_name)
        os.chdir(folder_path)

        dfs = []

        # Read and collect individual files
        for i in range(1, len(merged_change_df)+2):
            filename = f"{file_prefix}_{i}.csv"
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                dfs.append(df)
            else:
                print(f"File not found: {filename}, skipping.")

        # Combine and process DataFrame
        combined_df = pd.concat(dfs, ignore_index=True)

        # Handle Time column
        combined_df["Time"] = pd.to_datetime(combined_df["Time"], errors="coerce")
        combined_df = combined_df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

        # Infer time frequency
        time_diffs = combined_df["Time"].diff().dropna().mode()
        if not time_diffs.empty:
            freq = time_diffs[0]
        else:
            raise ValueError("Could not determine time frequency. Check your input data.")

        # Create full time range
        full_time_range = pd.DataFrame({"Time": pd.date_range(
            start=combined_df["Time"].min(),
            end=combined_df["Time"].max(),
            freq=freq
        )})

        # Merge and fill missing values
        merged_df = pd.merge(full_time_range, combined_df, on="Time", how="left")
        merged_df[value_column] = merged_df[value_column].fillna(0)

        # Save to CSV
        merged_df.to_csv(output_filename, index=False)
        print(f"Concatenation complete. File saved as {output_filename}")


# Define parameters
main_folder = main_folder  
sim_count = n_iter

# Process Outfall Data
process_catchment_data(
    main_folder=main_folder,
    sim_count=sim_count,
    file_prefix="catchment_outfall",
    value_column="Outfall_Volume",
    output_filename="catchment_outfall_combined.csv"
)

# Process Flooding Data
process_catchment_data(
    main_folder=main_folder,
    sim_count=sim_count,
    file_prefix="catchment_flood",
    value_column="Flood_Volume",
    output_filename="catchment_flood_combined.csv"
)


#%%
os.chdir(folder_path)
cso_annual_list = []  # To collect annual data from each iteration

for i_iter in range(0, n_iter+1):
    folder_name = f"Simulation_{i_iter}"
    folder_path = os.path.join(main_folder, folder_name)
    os.chdir(folder_path)

    # Choose correct CSV file
    csv_name = "catchment_outfall_ref.csv" if i_iter == 0 else "catchment_outfall_combined.csv"
    
    # Read and process data
    read_df = pd.read_csv(csv_name, header=(0))
       

    read_df["Time"] = pd.to_datetime(read_df["Time"], errors="coerce")
    cso_dt = read_df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    cso_dt["Outfall_Volume"] = cso_dt["Outfall_Volume"] * 0.6  # Convert to m³
    cso_dt.set_index("Time", inplace=True)

    # Resample annually
    cso_annual = cso_dt.resample('YE').sum()
    cso_annual["Iteration"] = i_iter  # Optional: track which iteration the data came from
    
    cso_annual_list.append(cso_annual)
    
    
# Combine all annual data
cso_all = pd.concat(cso_annual_list)

# Extract year and keep Iteration column
cso_all["Year"] = cso_all.index.year

# Split into baseline and others
baseline_df = cso_all[cso_all["Iteration"] == 0].copy()
others_df = cso_all[cso_all["Iteration"] != 0].copy()

# Group by year
baseline = baseline_df.groupby("Year")["Outfall_Volume"].mean()
others_mean = others_df.groupby("Year")["Outfall_Volume"].mean()
others_sem = others_df.groupby("Year")["Outfall_Volume"].sem()

# Align the data to ensure consistent year indexing
years = sorted(set(cso_all["Year"]))
x = np.arange(len(years))  # X locations for the groups

# Bar width and figure
bar_width = 0.35
plt.figure(figsize=(10, 6))

# Plot bars
plt.bar(x - bar_width/2, baseline.loc[years], width=bar_width, label='Ref. (stationary)')
plt.bar(x + bar_width/2, others_mean.loc[years], width=bar_width, yerr=others_sem.loc[years], 
        capsize=5, label='Dynamic')

# Axis formatting
plt.xlabel("Year")
plt.ylabel("Annual Outfall Volume (m³)")
plt.title("Annual CSO Volume: Ref. (stationary) vs Dynamic systems")
plt.xticks(x, [str(year) for year in years], rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show() 


#%%
os.chdir(folder_path)
flood_annual_list = []  # To collect annual data from each iteration

for i_iter in range(0, n_iter+1):
    folder_name = f"Simulation_{i_iter}"
    folder_path = os.path.join(main_folder, folder_name)
    os.chdir(folder_path)

    # Choose correct CSV file
    csv_name = "catchment_flood_ref.csv" if i_iter == 0 else "catchment_flood_combined.csv"
    
    # Read and process data
    read_df = pd.read_csv(csv_name, header=(0))
       

    read_df["Time"] = pd.to_datetime(read_df["Time"], errors="coerce")
    flood_dt = read_df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    flood_dt["Flood_Volume"] = flood_dt["Flood_Volume"] * 0.6  # Convert to m³
    flood_dt.set_index("Time", inplace=True)

    # Resample annually
    flood_annual = flood_dt.resample('YE').sum()
    flood_annual["Iteration"] = i_iter  # Optional: track which iteration the data came from
    
    flood_annual_list.append(flood_annual)
    
    
# Combine all annual data
flood_all = pd.concat(flood_annual_list)

# Extract year and keep Iteration column
flood_all["Year"] = flood_all.index.year

# Split into baseline and others
baseline_df = flood_all[flood_all["Iteration"] == 0].copy()
others_df = flood_all[flood_all["Iteration"] != 0].copy()

# Group by year
baseline = baseline_df.groupby("Year")["Flood_Volume"].mean()
others_mean = others_df.groupby("Year")["Flood_Volume"].mean()
others_sem = others_df.groupby("Year")["Flood_Volume"].sem()

# Align the data to ensure consistent year indexing
years = sorted(set(flood_all["Year"]))
x = np.arange(len(years))  # X locations for the groups

# Bar width and figure
bar_width = 0.35
plt.figure(figsize=(10, 6))

# Plot bars
plt.bar(x - bar_width/2, baseline.loc[years], width=bar_width, label='Ref. (stationary)')
plt.bar(x + bar_width/2, others_mean.loc[years], width=bar_width, yerr=others_sem.loc[years], 
        capsize=5, label='Dynamic')

# Axis formatting
plt.xlabel("Year")
plt.ylabel("Annual Flood Volume (m³)")
plt.title("Annual Flood Volume: Ref. (stationary) vs Dynamic systems")
plt.xticks(x, [str(year) for year in years], rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()     