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
from itertools import cycle
import seaborn as sns
import matplotlib as mpl

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
    #colours = ['#2c7bb6', '#abd9e9', '#dfc27d', '#fdae61', '#d7191c']
    colours = ['#66c2a4','#f1b6da','#969696','#c2a5cf','#dfc27d']  # Set2

    plt.rcParams['font.family'] = 'Calibri'

    # First fill between 0 and first survival curve
    lower = np.zeros_like(q)
    upper = cumulative_survivals[0]
    plt.fill_between(q, lower, upper, color=colours[0], alpha=0.6,
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
        plt.fill_between(q, lower, upper, color=colours[idx], alpha=0.5,
                         label=f'{idx + 1}')
        plt.plot(q, upper, color='grey', linestyle=':', linewidth=1.5)

    plt.xlabel('Cumulative waiting time [a]', fontsize = 24)
    plt.ylabel('Probability', fontsize = 24)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    #plt.title('Cumulative Survival Functions for Condition States')
    plt.legend(title='Condition \n state (CS)', loc = 'upper right', facecolor='white', framealpha=1.0, fontsize = 14, title_fontsize = 18)
    plt.grid(True, alpha=0.3)
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
    plt.rcParams['font.family'] = 'Calibri'
    colours = ['#66c2a4','#f1b6da','#969696','#c2a5cf','#dfc27d']

    for idx, cum_pdf in enumerate(cumulative_pdfs):
        # Build cumulative label: State 1, State 1+2, ..., State 1+2+...+N
        if idx == 0:
            lab = 'Condition state (CS) 1'
        else:
            lab = 'CS ' + '+'.join(str(i + 1) for i in range(idx + 1))
        color = colours[idx % len(colours)]  # Wrap around if needed
        plt.plot(q, cum_pdf, label=lab, color=color)
    
    plt.xlabel('Time [years]', fontsize=18)
    plt.ylabel('Probability', fontsize=18)
    plt.legend(title='Cumulative States', title_fontsize=18, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_individual_pdfs(pdfs, q, delta, normalise=True):
    """
    Plot one PDF per condition state (no convolution)
    with distinct colours and linestyles.
    """
    # Distinct linestyles & colours (will cycle if states > list length)
    plt.rcParams['font.family'] = 'Calibri'
    linestyles = cycle(['-', '-', '-', '--'])
    #colours = cycle(plt.cm.tab10.colors)  # tab10 gives 10 distinct colours
    colours = cycle(['#66c2a4','#f1b6da','#969696','#c2a5cf'])
    linwid = cycle([2, 1, 1, 1])

    for i, p in enumerate(pdfs, start=1):
        p = np.maximum(np.nan_to_num(p, nan=0.0), 0.0)
        if normalise:
            s = p.sum() * delta
            if s > 0:
                p = p / s

        ls = next(linestyles)
        c = next(colours)
        #lw = next(linwid)
        

        plt.plot(q, p, label=f"{i}", linestyle=ls, color=c)

    plt.xlabel("Waiting time [a]", fontsize = 18)
    plt.xlim(-1, 15)
    plt.ylabel("Probability", fontsize = 18)
    plt.legend(title="Condition \n state (CS)",  title_fontsize=14, fontsize=12)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
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
n_iter = 1            # Number of iterations (simulation runs)
i_start = 41
n_lid = 53              # Number of subcatchments with LIDs per iteration
base_dir = r"Q:\Abteilungsprojekte\eng\SWWData\Prabhat_Joshi\PhD\BGI_simulator\04_workspace\SWMM_model\Fehraltorf"
main_folder = r"Q:\Abteilungsprojekte\eng\SWWData\Prabhat_Joshi\PhD\BGI_simulator\04_workspace\SWMM_model\Fehraltorf"
files_to_copy = ["basicModel_catchment.inp", "basicModel_catchment_Tr0.hsf"]
number = random.choice([2, 3, 4, 5, 6, 7, 8])
print(f"The number of consecutive values required: {number}")

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
    plot_individual_pdfs(pdfs, q, delta)
    
    N = 1
    
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
        current = 0  # integer: current state
        for t in range(1, num_days):
            row = transition_matrices[t][current, :].astype(float)

            # enforce single-step dynamics (j -> j or j+1 only)
            row[:current] = 0.0
            row[current+2:] = 0.0
            s = row.sum()
            if s == 0.0:
                row = np.zeros_like(row); row[current] = 1.0
            else:
                row /= s

            proposal = np.random.choice(np.arange(len(row)), p=row)

            # optional persistence rule (need N consecutive “move” days to commit)
            if proposal > current:
                streak += 1
                if streak >= N:
                    current += 1
                    streak = 0
            else:
                streak = 0

            state_progression[t] = current     
        
                             
                            
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
sim_count = 35
combine_outfall_volumes(main_folder, sim_count)

#%% don't run this section
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
        final_df.to_csv(f"flood_combined_{i}.csv", index=False)
        print(f"\n Combined flood saved to: {folder_path}")
        
# # Define parameters
main_folder = main_folder  
sim_count = 35
combine_outfall_volumes(main_folder, sim_count)

#%%
os.chdir(folder_path)
Area = 17 # area in hectare
rain_data = r"Q:\Abteilungsprojekte\eng\SWWData\Prabhat_Joshi\PhD\BGI_simulator\04_workspace\SWMM_model\Fehraltorf\Simulation_0\rainfall.csv"
rain_df = pd.read_csv(rain_data)
rain_df["Time"] = pd.to_datetime(rain_df["Time"], errors = "coerce")
rain_dt = rain_df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
rain_dt.set_index("Time", inplace=True)
rain_dt["rainfall"] = rain_dt["rainfall"] * Area*10/60

rain_annual = rain_dt.resample('YE').sum()
rain_annual["Year"] = rain_annual.index.year


#%%
#os.chdir(folder_path)
cso_annual_list = []  # To collect annual data from each iteration

for i_iter in range(0, sim_count + 1):    
    print(i_iter)
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
baseline = (baseline_df.groupby("Time")["Outfall_Volume"].mean())/1000
others_mean = (others_df.groupby("Time")["Outfall_Volume"].mean())/1000
others_sem = (others_df.groupby("Time")["Outfall_Volume"].sem())/1000
others_sd = (others_df.groupby("Time")["Outfall_Volume"].std())/1000

# Align the data to ensure consistent year indexing
years = sorted(set(cso_all["Time"]))
years = years[0:26]
x = np.arange(len(years))  # X locations for the groups
#%%

# Bar width and figure
bar_width = 0.35
plt.figure(figsize=(10, 6))

# Plot bars
plt.bar(x - bar_width/2, baseline.loc[years], width=bar_width, label='Ref. (stationary)', color = 'red', alpha = 0.60)
plt.bar(x + bar_width/2, others_mean.loc[years], width=bar_width, yerr=others_sem.loc[years], 
        capsize=5, label='Dynamic (mean +/- SEM)', color = 'yellow', alpha = 0.60, error_kw=dict(ecolor='yellow'))
plt.xticks(x, [str(year) for year in years], rotation=45)

# Axis formatting
plt.xlabel("Year",  fontsize = 18, color = 'black')
plt.ylabel("Annual CSO Volume [10³ m³]",  fontsize = 18, color = 'black')
#plt.title("Annual CSO Volume: Ref. (stationary) vs Dynamic systems")
#plt.xticks(x, [str(year) for year in years], rotation=0)
plt.legend(fontsize = 16)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.xticks(fontsize = 12) 

# Bar width and figure
bar_width = 0.35
plt.figure(figsize=(10, 6))

# Plot bars
plt.bar(x - bar_width/2, baseline.loc[years], width=bar_width, label='Ref. (stationary)', color = '#80cdc1')
plt.bar(x + bar_width/2, others_mean.loc[years], width=bar_width, yerr=others_sd.loc[years], 
        capsize=5, label='Dynamic (mean +/- SD)', color = '#2b83ba')
plt.xticks(x, [str(year) for year in years], rotation=45)
plt.xticks(fontsize = 14) 


# Axis formatting
plt.xlabel("Year",  fontsize = 18, color = 'black')
plt.ylabel("Annual CSO Volume [10³ m³]",  fontsize = 18, color = 'black')
#plt.title("Annual CSO Volume: Ref. (stationary) vs Dynamic systems")
#plt.xticks(x, [str(year) for year in years], rotation=0)
plt.legend(fontsize = 16)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize = 14) 
plt.show() 



plt.figure(figsize=(10, 6))

plt.plot(np.cumsum(baseline.loc[years]), color = 'red', linestyle = '-', label = 'Ref. (stationary)')
plt.plot(np.cumsum(others_mean.loc[years]), color ='yellow', linestyle = '-', label = 'Dynamic (mean)')
plt.plot(np.cumsum(others_mean.loc[years] + others_sem.loc[years]), color ='yellow', linestyle = '--', label = 'Dynamic  (mean +/- SEM)', alpha = 0.5)
plt.plot(np.cumsum(others_mean.loc[years] - others_sem.loc[years]),color ='yellow', linestyle = '--', alpha = 0.5)

#plt.plot(others_mean - baseline, 'r--', label = 'Dynamic')
plt.xlabel("Year", fontsize = 18,  color = 'black')
plt.ylabel("Annual CSO Volume [10³ m³]", fontsize = 18,  color = 'black')
plt.legend(fontsize = 16)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize = 14) 
plt.show() 

plt.figure(figsize=(10, 6))

plt.plot(np.cumsum(baseline.loc[years]), color = '#80cdc1', linestyle = '-', label = 'Ref. (stationary)')
plt.plot(np.cumsum(others_mean.loc[years]), color = '#2b83ba', linestyle = '-', label = 'Dynamic (mean)')
plt.plot(np.cumsum(others_mean.loc[years] + others_sd.loc[years]), color ='#2b83ba', linestyle = '--', label = 'Dynamic  (mean +/- SD)', alpha = 0.5)
plt.plot(np.cumsum(others_mean.loc[years] - others_sd.loc[years]),color ='#2b83ba', linestyle = '--', alpha = 0.5)

#plt.plot(others_mean - baseline, 'r--', label = 'Dynamic')
plt.xlabel("Year", fontsize = 18,  color = 'black')
plt.ylabel("Annual CSO Volume [10³ m³]", fontsize = 18,  color = 'black')
plt.legend(fontsize = 16)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize = 14) 
plt.show() 

#%% build data
all_iterations_results = []
points_data = []             # for BGI_12 / BGI_50 dots (per-year, per-iter)
counts_records = []          # for stacked bars (per-year, per-state counts across all BGIs)

for i_iter in range(1, sim_count + 1):
    print(f"Processing iteration {i_iter}...")

    folder_name = f"Simulation_{i_iter}"
    folder_path = os.path.join(main_folder, folder_name)
    state_csv = f"iteration_{i_iter}_condition_states.csv"
    state_path = os.path.join(folder_path, state_csv)

    state = pd.read_csv(state_path, header=0)
    state['Date'] = pd.to_datetime(state['Date'], format='mixed')
    state = state.sort_values('Date')

    state_cols = [c for c in state.columns if c.startswith('S')]

    # Build state time series: zeros in raw mean "no event recorded on that date".
    # We convert zeros to NaN, carry last known state forward, then set initial state to CS1 (code 0).
    ts = (state.set_index('Date')[state_cols]
            .replace(0, np.nan)
            .ffill()
         )

    INITIAL_STATE_CODE = 0   # CS1
    ts = ts.fillna(INITIAL_STATE_CODE)

    # End-of-year state (codes 0..4) for every BGI
    final_by_year = ts.groupby(ts.index.year).last()
    final_by_year.index.name = 'Year'

    # ---- dots: collect S12 & S50 (we'll average across iterations later)
    if 'S12' in final_by_year.columns:
        for year, val in final_by_year['S12'].items():
            points_data.append({'Iteration': i_iter, 'Year': int(year), 'BGI': 'BGI_12', 'StateCode': float(val)})
    if 'S50' in final_by_year.columns:
        for year, val in final_by_year['S50'].items():
            points_data.append({'Iteration': i_iter, 'Year': int(year), 'BGI': 'BGI_50', 'StateCode': float(val)})

    # ---- stacked bars: count how many BGIs are in each state code 0..4 per year
    for year, row in final_by_year.iterrows():
        vals = pd.to_numeric(row, errors='coerce').dropna().values
        # round to be safe, then clip to 0..4
        vals = np.clip(np.rint(vals).astype(int), 0, 4)
        # counts for codes 0..4
        counts = np.bincount(vals, minlength=5)[:5]
        for code, count in enumerate(counts):
            counts_records.append({
                'Iteration': i_iter,
                'Year': int(year),
                'StateCode': code,     # 0..4
                'Count': int(count)
            })

# ---- tidy frames
points_df = pd.DataFrame(points_data)
counts_df = pd.DataFrame(counts_records)

if counts_df.empty:
    raise ValueError("No condition-state counts found. Check input/state coding (expecting codes 0..4).")

# proportions per iteration-year
counts_df['Total'] = counts_df.groupby(['Iteration', 'Year'])['Count'].transform('sum')
counts_df['Prop']  = counts_df['Count'] / counts_df['Total']

# average proportions across iterations → per year & state code
mean_prop = (counts_df.groupby(['Year', 'StateCode'], as_index=False)['Prop']
             .mean())

# pivot to wide for stacked bars
stack = (mean_prop
         .pivot(index='Year', columns='StateCode', values='Prop')
         .reindex(columns=[0,1,2,3,4])        # ensure order CS1..CS5
         .fillna(0.0))

# ---- compute per-year means for BGI_12 & BGI_50 dots (convert codes to CS numbers 1..5)
bgi12_mean = (points_df.query("BGI=='BGI_12'")
              .groupby('Year')['StateCode'].mean()
              .add(1)) if not points_df.empty else pd.Series(dtype=float)
bgi50_mean = (points_df.query("BGI=='BGI_50'")
              .groupby('Year')['StateCode'].mean()
              .add(1)) if not points_df.empty else pd.Series(dtype=float)
bgi12_median = (points_df.query("BGI=='BGI_12'")
              .groupby('Year')['StateCode'].median()
              .add(1)) if not points_df.empty else pd.Series(dtype=float)
bgi50_median = (points_df.query("BGI=='BGI_50'")
              .groupby('Year')['StateCode'].median()
              .add(1)) if not points_df.empty else pd.Series(dtype=float)

# --- duplicate 2011 as 2012 for BGI_12 / BGI_50 means AND medians ---
# --- duplicate 2011 as 2012 for the stacked bars ---
if 2011 in stack.index:
    stack.loc[2012] = stack.loc[2011].values
    stack = stack.sort_index()
else:
    print("NB: 2011 not found in 'stack' – nothing to copy.")


for s_name, s in {
    'bgi12_mean': bgi12_mean,
    'bgi50_mean': bgi50_mean,
    'bgi12_median': bgi12_median,
    'bgi50_median': bgi50_median
}.items():
    if s.size and (2011 in s.index):
        s.loc[2012] = s.loc[2011]
        s.sort_index(inplace=True)
    else:
        print(f"NB: 2011 not found in series '{s_name}' (or series empty).")

# Global style (Calibri + sizes)
mpl.rcParams.update({
    "font.family": "Calibri",
    "font.size": 16,
    "axes.titlesize": 15,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# Ensure full year range on stacked proportions (0..1)
years_full = np.arange(1987, 2013)
stack = stack.reindex(years_full, fill_value=0.0)

# Nice categorical palette for CS1..CS5 (colour-blind-friendly)
cs_cols = ['#66c2a4','#f1b6da','#969696','#c2a5cf','#dfc27d']  # Set2
cs_labels = {0:'CS1',1:'CS2',2:'CS3',3:'CS4',4:'CS5'}

# Prepare BGI dots (mean CS number, already computed as bgi12_mean/bgi50_mean)
# small x-offset so dots don’t sit exactly on top of each other
x12 = bgi12_median.index.values if not bgi12_median.empty else np.array([])
y12 = bgi12_median.values        if not bgi12_median.empty else np.array([])
x50 = bgi50_median.index.values if not bgi50_median.empty else np.array([])
y50 = bgi50_median.values        if not bgi50_median.empty else np.array([])

#%%

fig, ax = plt.subplots(figsize=(14, 6))

# Stacked bars (percent)
bottom = np.zeros(len(stack), dtype=float)
for code, col in zip(stack.columns, cs_cols):
    vals = stack[code].values
    ax.bar(stack.index, vals, bottom=bottom, color=col,
           edgecolor='white', linewidth=0.3, label=cs_labels.get(code, f'CS{code+1}'))
    bottom += vals

ax.set_xlabel("Year")
ax.set_ylabel("Proportion of the condition states (CS) \n of the BGIs [%]")
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_yticklabels([f"{int(v*100)}" for v in np.linspace(0, 1, 6)])
ax.grid(True, axis='y', alpha=0.25)

# Right axis for CS numbers (1–5) and dots only
ax2 = ax.twinx()
if x12.size:
    ax2.scatter(x12 - 0.15, y12, s=26, marker='o', label='BGI_12 (median CS)', color='#df65b0')
if x50.size:
    ax2.scatter(x50 + 0.15, y50, s=26, marker='o', label='BGI_50 (median CS)', color='#980043')

ax2.set_ylabel("Median CS of BGI_12 and BGI_50")
ax2.set_ylim(0.8, 5.2)
ax2.set_yticks([1,2,3,4,5])
ax2.set_yticklabels(['CS1','CS2','CS3','CS4','CS5'])

# Combined legend (compact, above plot)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2,
          loc='lower right',
          bbox_to_anchor=(0.98, 0.06), bbox_transform=ax.transAxes,  # tweak as needed
          ncol=2, frameon=True, framealpha=0.9)

plt.tight_layout()
plt.show()
# plt.savefig("bgi_states_stacked.png", dpi=300, bbox_inches="tight")


#%%
# --- 1) One x for everything ---
# Global style (Calibri + sizes)
mpl.rcParams.update({
    "font.family": "Calibri",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})


years = np.asarray(years, dtype=int)           # e.g. 1987..2012
x = np.arange(len(years))                      # 0..N-1 positions

# --- 2) Reindex ALL series to 'years' and use .values ---
base     = baseline.reindex(years).values
dyn_mean = others_mean.reindex(years).values
#dyn_sem  = others_sem.reindex(years).values
dyn_sd   = others_sd.reindex(years).values

# cumulative
cum_base      = np.cumsum(base)
cum_dyn       = np.cumsum(dyn_mean)
#cum_dyn_sem_u = np.cumsum(dyn_mean + dyn_sem)
#cum_dyn_sem_l = np.cumsum(dyn_mean - dyn_sem)
cum_dyn_sd_u  = np.cumsum(dyn_mean + dyn_sd)
cum_dyn_sd_l  = np.cumsum(dyn_mean - dyn_sd)

# stacked proportions table aligned to years
stack_plot = stack.reindex(years).fillna(0.0)  # columns = state codes 0..4

# BGI dots (medians or means), aligned + turned into x positions
bgi12_med = bgi12_median.reindex(years) if 'bgi12_median' in globals() else None
bgi50_med = bgi50_median.reindex(years) if 'bgi50_median' in globals() else None
x12 = np.where(~bgi12_med.isna())[0] if bgi12_med is not None else np.array([])
y12 = bgi12_med.values[~bgi12_med.isna()]     if bgi12_med is not None else np.array([])
x50 = np.where(~bgi50_med.isna())[0] if bgi50_med is not None else np.array([])
y50 = bgi50_med.values[~bgi50_med.isna()]     if bgi50_med is not None else np.array([])

# --- 3) Plot (3 rows, shared x) ---
fig, ax = plt.subplots(3, 1, figsize=(13.5, 16), constrained_layout=True, sharex=True, gridspec_kw={'height_ratios': [1, 0.6, 0.8]})  # Panel 2 double height)

# Panel 1: bars ± SD
bar_w = 0.38
ax[0].bar(x - bar_w/2, base,     width=bar_w, label='Ref. (stationary)', color='#018571', alpha=1)
ax[0].bar(x + bar_w/2, dyn_mean, width=bar_w, yerr=dyn_sd, capsize=4,
          label='Dynamic (mean ± SD)', color='#0571b0', alpha=1,
          error_kw=dict(ecolor='black', alpha=1, lw=1))
#ax[0].set_title("Annual CSO volume (± SD)")
ax[0].set_ylabel("Annual CSO Volume [10³ m³]")
ax[0].grid(axis='y', linestyle='--', alpha=0.4)
ax[0].legend(loc = 'lower right')

# Panel 2: cumulative + bands
ax[1].plot(x, cum_base, color='#018571', lw=1, label='Ref. (stationary)')
ax[1].plot(x, cum_dyn,  color='#0571b0', lw=1.2, label='Dynamic (mean)')
#ax[1].fill_between(x, cum_dyn_sem_l, cum_dyn_sem_u, color='#2b83ba', alpha=0.18, label='± SEM')
ax[1].fill_between(x, cum_dyn_sd_l,  cum_dyn_sd_u,  color='black', alpha=0.25, label='± SD')
#ax[1].set_title("Cumulative CSO volume")
ax[1].set_ylabel("Cumulative CSO volume [10³ m³]")
ax[1].grid(axis='y', linestyle='--', alpha=0.4)
ax[1].legend(loc = 'lower right')

# Panel 3: stacked proportions + dots
bottom = np.zeros(len(years), dtype=float)
for code, col in zip(stack_plot.columns, cs_cols):   # cs_cols from your palette
    vals = stack_plot[code].values
    ax[2].bar(x, vals, bottom=bottom, color=col,
              edgecolor='white', linewidth=0.3, label=cs_labels.get(code, f'CS{code+1}'))
    bottom += vals

#ax[2].set_title("BGI condition states (stacked proportions)")
ax[2].set_xlabel("Year")
ax[2].set_ylabel("Proportion of the condition \n states (CS) of the BGIs [%]")
ax[2].set_ylim(0, 1)
ax[2].set_yticks(np.linspace(0, 1, 6))
ax[2].set_yticklabels([f"{int(v*100)}" for v in np.linspace(0, 1, 6)])
ax[2].grid(True, axis='y', alpha=0.25)

# dots on twin y-axis; use same x positions
ax2 = ax[2].twinx()
if x12.size:
    ax2.scatter(x12 - 0.15, y12, s=28, marker='o', label='BGI_12 (median CS)', color='#810f7c')
if x50.size:
    ax2.scatter(x50 + 0.15, y50, s=28, marker='o', label='BGI_50 (median CS)', color='#253494')
ax2.set_ylabel("Median CS of \n BGI_12 and BGI_50")
ax2.set_ylim(0.8, 5.2)
ax2.set_yticks([1,2,3,4,5]); ax2.set_yticklabels(['CS1','CS2','CS3','CS4','CS5'])

# shared x tick labels = years
ax[2].set_xticks(x)
ax[2].set_xticklabels(years, rotation=45, ha='right')

# combined legend for panel 3
h1, l1 = ax[2].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax[2].legend(h1 + h2, l1 + l2, loc='lower right', frameon=True, framealpha=0.9, ncol=2)

plt.show()

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

# ------------ parameters ------------
ALPHA = 0.01          # per-year significance level for t-test
CONFIRM_K = 3         # total number of significant years required (non-consecutive)
K_THRESHOLD = 3.0     # CUSUM threshold multiplier (× std of annual differences)
YEAR_MIN, YEAR_MAX = 1987, 2012

# ------------ helper ------------
def ensure_year_col(df, time_col="Time"):
    if np.issubdtype(df[time_col].dtype, np.number):
        out = df.rename(columns={time_col: "Year"})
        return out
    out = df.copy()
    out["Year"] = pd.to_datetime(out[time_col], errors="coerce").dt.year
    return out

# ------------ prep data ------------
baseline = ensure_year_col(baseline_df, "Time")
dynamic  = ensure_year_col(others_df,   "Time")

baseline = baseline[(baseline["Year"] >= YEAR_MIN) & (baseline["Year"] <= YEAR_MAX)]
dynamic  = dynamic[(dynamic["Year"]  >= YEAR_MIN) & (dynamic["Year"]  <= YEAR_MAX)]

# lists of values by year (each list = iterations/runs that year)
ref_all    = baseline.groupby("Year")["Outfall_Volume"].apply(list)
others_all = dynamic.groupby("Year")["Outfall_Volume"].apply(list)

# align years present in both
years = np.array(sorted(set(ref_all.index).intersection(others_all.index)))
if years.size == 0:
    raise ValueError("No overlapping years between baseline and dynamic.")

# ------------ t-tests per year (dynamic vs mean(static)) ------------
rows = []
for y in years:
    dyn_vals = np.asarray(others_all.loc[y], dtype=float)
    stat_mean = float(np.mean(ref_all.loc[y]))
    t_stat, p_val = ttest_1samp(dyn_vals, popmean=stat_mean, alternative="greater")
    rows.append({"Year": y, "t": t_stat, "p": p_val, "significant": p_val < ALPHA})

ttab = pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)
ttab["cum_sig"] = ttab["significant"].cumsum()

# confirmation year = first year cumulative significant years ≥ CONFIRM_K (non-consecutive)
hit = ttab.loc[ttab["cum_sig"] >= CONFIRM_K]
confirmation_year = int(hit.iloc[0]["Year"]) if not hit.empty else None

# ------------ CUSUM on annual mean differences ------------
# annual means per year
base_mean = np.array([np.mean(ref_all.loc[y])    for y in years])
dyn_mean  = np.array([np.mean(others_all.loc[y]) for y in years])
diff = dyn_mean - base_mean

# one-sided (upper) CUSUM with target = 0
cusum = np.zeros_like(diff, dtype=float)
for i in range(1, len(diff)):
    cusum[i] = max(0.0, cusum[i-1] + diff[i])

# threshold = K_THRESHOLD × std(diff)
diff_std = diff.std(ddof=1)
threshold = (K_THRESHOLD * diff_std) if diff_std > 0 else np.inf

tipping_year = None
for i, v in enumerate(cusum):
    if v >= threshold:
        tipping_year = int(years[i])
        break

# ------------ print summary ------------
for _, r in ttab.iterrows():
    print(f"year: {int(r['Year'])}, t-statistic = {r['t']:.3f}, p-value = {r['p']:.4g} -> {'sig' if r['significant'] else 'ns'}")

print("\nCUSUM candidate (threshold crossing):", tipping_year)
print(f"Confirmation year (≥{CONFIRM_K} total significant years, α={ALPHA}):", confirmation_year)

# ------------ plots ------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

# Panel A: CUSUM
ax[0].plot(years, cusum, lw=1.5, label="CUSUM (dyn − static mean)", color = '#7a0177')
ax[0].axhline(threshold, ls="--", color="red", label=f"Threshold = {threshold:.2f}")
if tipping_year is not None:
    ax[0].axvline(tipping_year, ls=":", color="green", label=f"Tipping year = {tipping_year}")
ax[0].set_title("CUSUM (one-sided, upward)")
ax[0].set_xlabel("Year"); ax[0].set_ylabel("CUSUM")
ax[0].grid(True, alpha=0.3); ax[0].legend(loc = 'lower right')

# Panel B: p-values
ax[1].plot(ttab["Year"], ttab["p"], marker="o", lw=1.5, label="p-value (one-sample t-test)", color = '#c51b8a')
ax[1].axhline(ALPHA, ls="--", color="red", label=f"α = {ALPHA}")
if confirmation_year is not None:
    ax[1].axvline(confirmation_year, ls=":", color="green",
                  label=f"Confirmation year (total {CONFIRM_K})")
ax[1].invert_yaxis()  # smaller p = stronger evidence
ax[1].set_title("Per-year significance (dynamic > static mean)")
ax[1].set_xlabel("Year"); ax[1].set_ylabel("p-value")
ax[1].grid(True, alpha=0.3); ax[1].legend(loc = 'lower right')

plt.show()
