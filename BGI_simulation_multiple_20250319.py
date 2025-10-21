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
import re
import random

sys.path.append(r'Q:\Abteilungsprojekte\eng\SWWData\Prabhat_Joshi\PhD\BGI_simulator\08_codes_and_functions')
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

    
def plot_cumulative_survivals2(cumulative_survivals, q):
    """
    Plot cumulative survival functions for all condition states,
    shading the area between successive curves with different colours.

    Parameters:
    - cumulative_survivals: List of cumulative survival arrays for each state.
    - q: Time vector (discretized time steps).
    """
    plt.figure(figsize=(8, 6))

    #cmap = plt.get_cmap('tab10')
    #colours = cmap(np.linspace(0, 1, len(cumulative_survivals) + 1))
    colours = ['#a6611a', '#dfc27d', '#abd9e9', '#80cdc1', '#018571']

    # First fill between 0 and first survival curve
    lower = np.zeros_like(q)
    upper = cumulative_survivals[0]
    plt.fill_between(q, lower, upper, color=colours[0], alpha=0.4,
                     label='1')
    plt.plot(q, upper, color=colours[0], linestyle='-', linewidth=2)
    
    # Fill between successive survival curves
    for idx in range(1, len(cumulative_survivals)):
        lower = cumulative_survivals[idx - 1]
        upper = cumulative_survivals[idx]
        plt.fill_between(q, lower, upper, color=colours[idx], alpha=0.4,
                         label=f'{idx + 1}')
        plt.plot(q, upper, color=colours[idx], linestyle='-', linewidth=2)

    # Optionally: fill between last curve and 1 (top of plot)
    if np.any(cumulative_survivals[-1] < 1):
        lower = cumulative_survivals[-1]
        upper = np.ones_like(q)
        idx += 1  # increment for colour
        plt.fill_between(q, lower, upper, color=colours[idx], alpha=0.4,
                         label=f'{idx + 1}')
        plt.plot(q, upper, color='grey', linestyle=':', linewidth=1.5)

    plt.xlabel('Time [years]', fontsize = 16)
    plt.ylabel('Survival probability', fontsize = 16)
    #plt.title('Cumulative Survival Functions for Condition States')
    plt.legend(title='States', loc = 'upper right', facecolor='white', framealpha=1.0, fontsize = 16, title_fontsize = 18)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_cumulative_pdfs(cumulative_pdfs, q):
    """
    Plot cumulative PDFs for all condition states with cumulative legend labels.

    Parameters:
    - cumulative_pdfs: List of cumulative PDFs for all condition states.
    - q: Time vector (discretized time steps).
    """
    plt.figure(figsize=(8, 6))
    
    colours = ['#a6611a', '#dfc27d', '#abd9e9', '#80cdc1', '#018571']

    for idx, cum_pdf in enumerate(cumulative_pdfs):
        # Build cumulative label: State 1, State 1+2, ..., State 1+2+...+N
        if idx == 0:
            lab = 'State 1'
        else:
            lab = 'States ' + '+'.join(str(i + 1) for i in range(idx + 1))
        color = colours[idx % len(colours)]  # Wrap around if needed
        plt.plot(q, cum_pdf, label=lab, color=color)
    
    plt.xlabel('Time [years]', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    plt.legend(title='Cumulative States', title_fontsize=14, fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%%
def edit_inpfile(model_name, start_date, end_date, save_hotstart, use_hotstart,
                                    new_lid_process, iteration, start_line=847, end_line=899, update_row_index=0):
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
                    #tokens[8] = f'"LID_{subcatchment}_Tr{iteration}.txt"'
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
n_iter = 5            # Number of iterations (simulation runs)
i_start = 6
n_lid = 53              # Number of subcatchments with LIDs per iteration
base_dir = r"Q:\Abteilungsprojekte\eng\SWWData\Prabhat_Joshi\PhD\BGI_simulator\04_workspace\SWMM_model\Fehraltorf"
main_folder = r"Q:\Abteilungsprojekte\eng\SWWData\Prabhat_Joshi\PhD\BGI_simulator\04_workspace\SWMM_model\Fehraltorf"
files_to_copy = ["basicModel_catchment.inp", "basicModel_catchment_Tr0.hsf"]
number = random.choice([1, 2, 3, 4, 5, 6])
print(number)

num_years = 25
num_days = 365 * num_years
delta = 1/365
num_states = 4
start_date = (datetime.strptime('01/01/1987 00:00', '%m/%d/%Y %H:%M'))

# Save k and lambdas to CSV
k = [2.042, 2.042, 2.042, 2.042]
lambdas = [4.188, 3.590, 2.393, 3.590]
#lambdas = [300, 597.5, 1195, 1493.751]
df = pd.DataFrame({"k": k, "lambdas": lambdas})
df.to_csv("k_and_lambdas.csv", index=False)
print("CSV file 'k_and_lambdas.csv' saved successfully.")

# Containers for overall results
all_iterations_results = {}
all_merged_change_dates = {}  # To store merged change dates per iteration

# ----- Outer Loop: Iterations -----
for i_iter in range(i_start, i_start + n_iter):
    print(f"Running iteration {i_iter} out of {i_start + n_iter}")
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
    plot_cumulative_survivals2(cumulative_survivals, q)
    
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
        
        proposed_state = state_progression[0]
        streak = 1
        
        # Iterate over each day to update state
        for t in range(1, num_days):
            state_vector = np.dot(state_vector, transition_matrices[t])
            state_vectors[t] = state_vector
        
            states = np.arange(len(state_vectors[t]))
            transition_prob = 1 - state_vectors[t][proposed_state]
        
            probs = np.round(state_vectors[t], 4)
            probs = probs / probs.sum()
        
            if np.random.rand() < transition_prob:
                new_proposal = np.random.choice(states, p=probs)
            else:
                new_proposal = proposed_state
           
        
            if new_proposal == proposed_state:
                streak += 1
            else:
                streak = 1
                proposed_state = new_proposal
        
            # Only commit change if it's persisted for 5 days
            if streak >= number:
                state_progression[t] = proposed_state
            else:
                state_progression[t] = state_progression[t - 1]
                
                
        for t in range(1, num_days):
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
    occupied_dates = set()
    merged_change_info = []

    for d in unique_dates:
        # Collect subcatchments scheduled to change on this date
        changes_today = []
        for subcatchment, changes in change_info.items():
            for (date_str, new_state) in changes:
                if date_str == d:
                    changes_today.append((subcatchment, new_state))
                    break

        # Sort for consistent processing
        changes_today.sort()

        for subcatchment, new_state in changes_today:
            current_date = datetime.strptime(d, '%m/%d/%Y %H:%M')
            while current_date.strftime('%m/%d/%Y %H:%M') in occupied_dates:
                current_date += timedelta(days=1)
            date_str = current_date.strftime('%m/%d/%Y %H:%M')
            occupied_dates.add(date_str)

            merged_change_info.append({
                'Date': date_str,
                'Subcatchments': f"{subcatchment}:{new_state}"
            })
    
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
            end_date = "12/31/2012 23:50"
            save_hotstart = f"basicModel_catchment_Tr{jj+1}.hsf"
            use_hotstart = f"basicModel_catchment_Tr{jj}.hsf"
            lid_name = f"BR_CS{merged_change_df['New_State'][jj] + 1}"
            row_number = int(merged_change_df.Subcatchments[jj].split(':')[0][1:])-1
           
            
        else:
            st_date = (datetime.strptime(merged_change_df.Date[jj], '%m/%d/%Y %H:%M')).strftime('%m/%d/%Y %H:%M')
            end_date = (datetime.strptime(merged_change_df.Date[jj+1], '%m/%d/%Y %H:%M') - timedelta(days = 1)).strftime('%m/%d/%Y %H:%M')
            save_hotstart = f"basicModel_catchment_Tr{jj+1}.hsf"
            use_hotstart = f"basicModel_catchment_Tr{jj}.hsf"
            lid_name = f"BR_CS{merged_change_df['New_State'][jj] + 1}"
            row_number = int(merged_change_df.Subcatchments[jj].split(':')[0][1:])-1
            
        #it = jj+1
               
       
        xx, vv = edit_inpfile(model_name = "basicModel_catchment.inp", 
                     start_date = st_date, 
                     end_date = end_date, 
                     save_hotstart = save_hotstart, 
                     use_hotstart = use_hotstart,
                     new_lid_process = lid_name,
                     iteration = jj,
                     start_line = 847,
                     end_line = 899,
                     update_row_index = row_number)
        
        print(f"Subcatchment: S{xx+1}, LID: {vv}")
        
                
        with Simulation(inputfile="basicModel_catchment.inp", reportfile = rpt_file, outputfile = out_file) as sim:           
                                     
            for step in sim:
                pass    
                        
        rpt_file = f"basicModel_catchment_{i_iter}_{jj+1}.rpt"
        out_file = f"basicModel_catchment_{i_iter}_{jj+1}.out"
        
        with Output(out_file) as out:
            ts_flood = SystemSeries(out).flood_losses
            ts_outfall = SystemSeries(out).outfall_flows  
            ts_ara = NodeSeries(out)['ARA_Fehraltorf'].total_inflow
            
        ts_flood_df = pd.DataFrame(list(ts_flood.items()), columns=["Time", "Flood_Volume"])
        #ts_flood_df.to_csv("catchment_flood_ref.csv", index = False)
        ts_flood_df["Time"] = pd.to_datetime(ts_flood_df["Time"], errors = "coerce")
        flood_dt = ts_flood_df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
        flood_dt["Flood_Volume"] = flood_dt["Flood_Volume"] * 0.6 # Convert to m3
        flood_dt["Time"] = pd.to_datetime(flood_dt["Time"])
        flood_dt.set_index("Time", inplace=True)
        flood_annual_sum = flood_dt.resample('YE').sum()   
        flood_annual_sum.to_csv(f"catchment_flood_{jj+1}.csv", index = True)
          
           
        ts_outfall_df = pd.DataFrame(list(ts_outfall.items()), columns=["Time", "Outfall_Volume"])
        #ts_outfall_df.to_csv("catchment_outfall_ref.csv", index = False)
        ts_outfall_df["Time"] = pd.to_datetime(ts_outfall_df["Time"], errors = "coerce")
        cso_dt = ts_outfall_df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
        cso_dt["Outfall_Volume"] = cso_dt["Outfall_Volume"] * 0.6 # Convert to m3
        cso_dt["Time"] = pd.to_datetime(cso_dt["Time"])
        cso_dt.set_index("Time", inplace=True)
        cso_annual_sum = cso_dt.resample('YE').sum()
        cso_annual_sum["Time"] = cso_annual_sum.index.year
        cso_annual_sum.to_csv(f"catchment_outfall_{jj+1}.csv", index = False) 

        ts_ara_df = pd.DataFrame(list(ts_ara.items()), columns=["Time", "ARA"])
        #ts_ara_df.to_csv("ara_outfall_ref.csv", index = False)
        ts_ara_df["Time"] = pd.to_datetime(ts_ara_df["Time"], errors = "coerce")
        ts_ara_dt = ts_ara_df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
        ts_ara_dt["ARA"] = ts_ara_dt["ARA"] * 0.6 # Convert to m3
        ts_ara_dt["Time"] = pd.to_datetime(ts_ara_dt["Time"])
        ts_ara_dt.set_index("Time", inplace=True)
        ara_annual_sum = ts_ara_dt.resample('YE').sum()
        ara_annual_sum["Time"] = ara_annual_sum.index.year
        ara_annual_sum.to_csv(f"ara_outfall_{jj+1}.csv", index = False) 

        net_outfall = cso_annual_sum['Outfall_Volume'] - ara_annual_sum['ARA']
        net_outfall.name = "Outfall_Volume"
        net_outfall.index = net_outfall.index.year
        net_outfall.to_csv(f"net_annual_outfall_{jj+1}.csv", index = True)
               
        
    if i_iter > 1:
        folder_name_prev = f"Simulation_{i_iter-1}"
        folder_path_prev = os.path.join(main_folder, folder_name_prev)
    
    if not os.path.exists(folder_path_prev):
        print(f"Skipping cleanup: Folder does not exist — {folder_path_prev}")
    else:
        os.chdir(folder_path_prev)
        
        for kk in range(1, 300):
            out_file_prev = f"basicModel_catchment_{i_iter-1}_{kk}.out"
            if os.path.exists(out_file_prev):
                os.remove(out_file_prev)
                print(f"Removed .out file: {out_file_prev}")   
        
    
    # Return to base directory before next iteration
    os.chdir(main_folder)




#%%

# def process_catchment_data(main_folder, sim_count, file_prefix, value_column, output_filename):
#     for i_iter in range(1, sim_count + 1):
#         print(f"\nReading Simulation: {i_iter}")
#         folder_name = f"Simulation_{i_iter}"
#         folder_path = os.path.join(main_folder, folder_name)

#         if not os.path.isdir(folder_path):
#             print(f"Folder not found: {folder_path}, skipping.")
#             continue

#         print(f"Looking in: {folder_path}")
#         print(f"n_lid = {n_lid}, num_states = {num_states}")
#         expected_file_count = n_lid * (num_states + 1) - 1
#         print(f"Expecting {expected_file_count} files with prefix {file_prefix}_#.csv")

#         dfs = []

#         for i in range(1, (n_lid * (num_states + 1))):
#             filename = os.path.join(folder_path, f"{file_prefix}_{i}.csv")
#             if os.path.exists(filename):
#                 df = pd.read_csv(filename)
#                 dfs.append(df)
#             else:
#                 print(f"File not found: {filename}, skipping.")

#         if not dfs:
#             print(f"No valid CSV files loaded for Simulation {i_iter}. Skipping to next.")
#             continue

#         combined_df = pd.concat(dfs, ignore_index=True)

#         if "Time" not in combined_df.columns:
#             print("Missing 'Time' column in data. Skipping this simulation.")
#             continue

#         combined_df["Time"] = pd.to_datetime(combined_df["Time"], errors="coerce")
#         combined_df = combined_df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

#         if combined_df.empty:
#             print("DataFrame is empty after removing invalid 'Time' values. Skipping.")
#             continue

#         # Infer time frequency
#         time_diffs = combined_df["Time"].diff().dropna().mode()
#         if not time_diffs.empty:
#             freq = time_diffs[0]
#         else:
#             print("Could not determine time frequency. Skipping this simulation.")
#             continue

#         # Create full time range
#         try:
#             full_time_range = pd.DataFrame({
#                 "Time": pd.date_range(
#                     start=combined_df["Time"].min(),
#                     end=combined_df["Time"].max(),
#                     freq=freq
#                 )
#             })
#         except Exception as e:
#             print(f"Failed to create time range: {e}")
#             continue

#         # Merge and fill missing values
#         merged_df = pd.merge(full_time_range, combined_df, on="Time", how="left")
#         merged_df[value_column] = merged_df[value_column].fillna(0)

#         # Create simulation-specific output filename
#         output_path = os.path.join(main_folder, output_filename.replace(".csv", f"_{i_iter}.csv"))
#         merged_df.to_csv(output_path, index=False)
#         print(f"Concatenation complete. File saved as {output_path}")




# # Process Outfall Data
# process_catchment_data(
#     main_folder=main_folder,
#     sim_count=sim_count,
#     file_prefix="net_annual_outfall",
#     value_column="Outfall_Volume",
#     output_filename="catchment_outfall_combined.csv"
# )

# # Process Flooding Data
# process_catchment_data(
#     main_folder=main_folder,
#     sim_count=sim_count,
#     file_prefix="catchment_flood",
#     value_column="Flood_Volume",
#     output_filename="catchment_flood_combined.csv"
# )

#%%

def combine_outfall_volumes(main_folder, sim_count, output_filename="outfall_combined.csv"):
   

    for i in range(1, sim_count + 1):
        all_sim_data = []
        folder_name = f"Simulation_{i}"
        folder_path = os.path.join(main_folder, folder_name)

        if not os.path.isdir(folder_path):
            print(f"Folder not found: {folder_path}, skipping.")
            continue

        print(f"Processing: {folder_name}")
        pattern = os.path.join(folder_path, "net_annual_outfall_*.csv")
        files = glob.glob(pattern)

        if not files:
            print(f"No files found in {folder_name}, skipping.")
            continue

        dfs = []
        for file in files:
            df = pd.read_csv(file, usecols=["Time", "Outfall_Volume"])
            #df["Time"] = pd.to_datetime(df["Time"], errors="coerce").dt.year  # Convert to year
            df = df.dropna(subset=["Time"])
            df = df.groupby("Time", as_index=False).sum()
            dfs.append(df)

        if dfs:
            sim_df = pd.concat(dfs, ignore_index=True)
            sim_df = sim_df.groupby("Time", as_index=False).sum()
            all_sim_data.append(sim_df)

        if not all_sim_data:
            print("No valid data found across simulations.")
            return
    
        final_df = []
        final_df = pd.concat(all_sim_data, ignore_index=True)
        final_df = final_df.groupby("Time", as_index=False).sum()
        final_df = final_df.sort_values("Time")
    
        #output_path = os.path.join(main_folder, output_filename)
        os.chdir(folder_path)
        final_df.to_csv(f"outfall_combined_{i}.csv", index=False)
        print(f"\n Combined outfall saved to: {folder_path}")
        
# # Define parameters
main_folder = main_folder  
sim_count = n_iter
combine_outfall_volumes(main_folder, sim_count)

#%%
def combine_flood_volumes(main_folder, sim_count, output_filename="flood_combined.csv"):
   

    for i in range(1, sim_count + 1):
        all_sim_data = []
        folder_name = f"Simulation_{i}"
        folder_path = os.path.join(main_folder, folder_name)

        if not os.path.isdir(folder_path):
            print(f"Folder not found: {folder_path}, skipping.")
            continue

        print(f"Processing: {folder_name}")
        pattern = os.path.join(folder_path, "net_annual_outfall_*.csv")
        files = glob.glob(pattern)

        if not files:
            print(f"No files found in {folder_name}, skipping.")
            continue

        dfs = []
        for file in files:
            df = pd.read_csv(file, usecols=["Time", "Outfall_Volume"])
            #df["Time"] = pd.to_datetime(df["Time"], errors="coerce").dt.year  # Convert to year
            df = df.dropna(subset=["Time"])
            df = df.groupby("Time", as_index=False).sum()
            dfs.append(df)

        if dfs:
            sim_df = pd.concat(dfs, ignore_index=True)
            sim_df = sim_df.groupby("Time", as_index=False).sum()
            all_sim_data.append(sim_df)

        if not all_sim_data:
            print("No valid data found across simulations.")
            return
    
        final_df = []
        final_df = pd.concat(all_sim_data, ignore_index=True)
        final_df = final_df.groupby("Time", as_index=False).sum()
        final_df = final_df.sort_values("Time")
    
        #output_path = os.path.join(main_folder, output_filename)
        os.chdir(folder_path)
        final_df.to_csv(f"outfall_combined_{i}.csv", index=False)
        print(f"\n Combined outfall saved to: {folder_path}")
        
# # Define parameters
main_folder = main_folder  
sim_count = n_iter
combine_outfall_volumes(main_folder, sim_count)

#%%
# os.chdir(folder_path)
# Area = 17 # area in hectare
# rain_data = r"Q:\Abteilungsprojekte\eng\SWWData\Prabhat_Joshi\PhD\BGI_simulator\04_workspace\SWMM_model\Simulation_0\rainfall.csv"
# rain_df = pd.read_csv(rain_data)
# rain_df["Time"] = pd.to_datetime(rain_df["Time"], errors = "coerce")
# rain_dt = rain_df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
# rain_dt.set_index("Time", inplace=True)
# rain_dt["rainfall"] = rain_dt["rainfall"] * Area*10/60

# rain_annual = rain_dt.resample('YE').sum()
# rain_annual["Year"] = rain_annual.index.year


#%%
os.chdir(folder_path)
cso_annual_list = []  # To collect annual data from each iteration

for i_iter in range(0, n_iter+1):
    folder_name = f"Simulation_{i_iter}"
    folder_path = os.path.join(main_folder, folder_name)
    os.chdir(folder_path)

    # Choose correct CSV file
    csv_name = "net_annual_outfall.csv" if i_iter == 0 else f"outfall_combined_{i_iter}.csv"
    
    # Read and process data
    read_df = pd.read_csv(csv_name, header=(0))
       

    #read_df["Time"] = pd.to_datetime(read_df["Time"], errors="coerce")
    cso_dt = read_df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    cso_dt["Iteration"] = i_iter

    #cso_dt["Outfall_Volume"] = cso_dt["Outfall_Volume"] * 0.6  # Convert to m³
    #cso_dt.set_index("Time", inplace=True)
    cso_annual_list.append(cso_dt)


    # # Resample annually
    # cso_annual = cso_dt.resample('YE').sum()
    # cso_annual["Iteration"] = i_iter  # Optional: track which iteration the data came from
    
    # cso_annual_list.append(cso_annual)
    # cso_annual_list.append(cso_dt)
    
    
# Combine all annual data
cso_all = pd.concat(cso_annual_list)

# Extract year and keep Iteration column
#cso_all["Time"] = cso_all.index.Time

# Split into baseline and others
baseline_df = cso_all[cso_all["Iteration"] == 0].copy()
others_df = cso_all[cso_all["Iteration"] != 0].copy()

# Group by year
# rain_mean = rain_annual.groupby("Year")["rainfall"].mean()
# rain_sem = rain_annual.groupby("Year")["rainfall"].sem()
baseline = baseline_df.groupby("Time")["Outfall_Volume"].mean()
others_mean = others_df.groupby("Time")["Outfall_Volume"].mean()
others_sem = others_df.groupby("Time")["Outfall_Volume"].sem()

# Align the data to ensure consistent year indexing
years = sorted(set(cso_all["Time"]))
years = years[1:26]
x = np.arange(len(years))  # X locations for the groups

# Bar width and figure
bar_width = 0.35
plt.figure(figsize=(10, 6))

# Plot bars
plt.bar(x - bar_width/2, baseline.loc[years], width=bar_width, label='Ref. (stationary)')
plt.bar(x + bar_width/2, others_mean.loc[years], width=bar_width, yerr=others_sem.loc[years], 
        capsize=5, label='Dynamic')
plt.xticks(x, [str(year) for year in years], rotation=45)

# Axis formatting
plt.xlabel("Year",  fontsize = 14, color = 'black')
plt.ylabel("Annual CSO Volume (m³)",  fontsize = 14, color = 'black')
#plt.title("Annual CSO Volume: Ref. (stationary) vs Dynamic systems")
#plt.xticks(x, [str(year) for year in years], rotation=0)
plt.legend(fontsize = 14)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show() 


plt.figure(figsize=(10, 6))

plt.plot(np.cumsum(baseline.loc[years]),'k-', label = 'Ref. (stationary)')
plt.plot(np.cumsum(others_mean.loc[years]),'r-', label = 'Dynamic')
plt.plot(np.cumsum(others_mean.loc[years] + others_sem.loc[years]),'r--', label = 'Dynamic +', alpha = 0.5)
plt.plot(np.cumsum(others_mean.loc[years] - others_sem.loc[years]),'r--', label = 'Dynamic -', alpha = 0.5)

#plt.plot(others_mean - baseline, 'r--', label = 'Dynamic')
plt.xlabel("Year", fontsize = 10)
plt.ylabel("Annual CSO Volume (m³)", fontsize = 10)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show() 

#%%
# os.chdir(folder_path)
# flood_annual_list = []  # To collect annual data from each iteration

# for i_iter in range(0, n_iter+1):
#     folder_name = f"Simulation_{i_iter}"
#     folder_path = os.path.join(main_folder, folder_name)
#     os.chdir(folder_path)

#     # Choose correct CSV file
#     csv_name = "catchment_flood_ref.csv" if i_iter == 0 else "catchment_flood_combined.csv"
    
#     # Read and process data
#     read_df = pd.read_csv(csv_name, header=(0))
       

#     read_df["Time"] = pd.to_datetime(read_df["Time"], errors="coerce")
#     flood_dt = read_df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

#     flood_dt["Flood_Volume"] = flood_dt["Flood_Volume"] * 0.6  # Convert to m³
#     flood_dt.set_index("Time", inplace=True)

#     # Resample annually
#     flood_annual = flood_dt.resample('YE').sum()
#     flood_annual["Iteration"] = i_iter  # Optional: track which iteration the data came from
    
#     flood_annual_list.append(flood_annual)
    
    
# # Combine all annual data
# flood_all = pd.concat(flood_annual_list)

# # Extract year and keep Iteration column
# flood_all["Year"] = flood_all.index.year

# # Split into baseline and others
# baseline_df = flood_all[flood_all["Iteration"] == 0].copy()
# others_df = flood_all[flood_all["Iteration"] != 0].copy()

# # Group by year
# baseline = baseline_df.groupby("Year")["Flood_Volume"].mean()
# others_mean = others_df.groupby("Year")["Flood_Volume"].mean()
# others_sem = others_df.groupby("Year")["Flood_Volume"].sem()

# # Align the data to ensure consistent year indexing
# years = sorted(set(flood_all["Year"]))
# x = np.arange(len(years))  # X locations for the groups

# # Bar width and figure
# bar_width = 0.35
# plt.figure(figsize=(10, 6))

# # Plot bars
# plt.bar(x - bar_width/2, baseline.loc[years], width=bar_width, label='Ref. (stationary)')
# plt.bar(x + bar_width/2, others_mean.loc[years], width=bar_width, yerr=others_sem.loc[years], 
#         capsize=5, label='Dynamic')

# # Axis formatting
# plt.xlabel("Year")
# plt.ylabel("Annual Flood Volume (m³)")
# plt.title("Annual Flood Volume: Ref. (stationary) vs Dynamic systems")
# plt.xticks(x, [str(year) for year in years], rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()     



