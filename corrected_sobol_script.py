
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 16:29:38 2025

@author: joshipra
"""

# Import required libraries
import os
import shutil
import numpy as np
import pandas as pd
from pyswmm import Simulation
from swmm_api import read_rpt_file
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.ensemble import RandomForestRegressor

# Define the sensitivity problem
num_lids = 53
problem = {
    'num_vars': num_lids,
    'names': [f'BGI_{i+1}' for i in range(num_lids)],
    'bounds': [[1, 5]] * num_lids  # Continuous range for Sobol
}

# Generate samples
N = 500
sample = saltelli.sample(problem, N=N, calc_second_order=False)  # Keep as continuous

# Function to discretise state values inside the model
def discretise_state(x):
    return int(np.clip(np.floor(x), 1, 5))

# Function to edit the SWMM .inp file
def edit_inpfile(model_name, start_date, end_date, lid_state_row, start_line=842, end_line=894):
    with open(model_name, 'r') as file:
        lines = file.readlines()

    expected_rows = end_line - start_line + 1
    if len(lid_state_row) != expected_rows:
        raise ValueError("Mismatch in number of BGI states")

    with open(model_name, 'w') as file:
        for i, line in enumerate(lines):
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
            elif '[FILES]' in line:
                file.write(line)
            elif start_line - 1 <= i <= end_line - 1:
                tokens = line.split()
                if len(tokens) >= 9:
                    lid_index = i - (start_line - 1)
                    subcatchment = tokens[0]
                    state = discretise_state(lid_state_row[lid_index])
                    tokens[1] = f"BR_CS{state}"
                    tokens[8] = f'"LID_{subcatchment}_Tr1.txt"'
                    file.write(' '.join(tokens) + '\n')
                else:
                    file.write(line)
            else:
                file.write(line)

# Function to run simulations and collect CSO
def run_sobol_simulations(base_folder, base_model, start_date, end_date):
    baseline_state = [1] * num_lids
    baseline_folder = os.path.join(base_folder, "Simulation_0")
    baseline_inp = os.path.join(baseline_folder, base_model)
    baseline_rpt = baseline_inp.replace(".inp", ".rpt")
    baseline_out = baseline_inp.replace(".inp", ".out")

    if not os.path.exists(baseline_folder):
        os.makedirs(baseline_folder)
        shutil.copy(os.path.join(base_folder, base_model), baseline_inp)
        edit_inpfile(baseline_inp, start_date, end_date, baseline_state)

    with Simulation(baseline_inp, reportfile=baseline_rpt) as sim:
        for _ in sim:
            pass

    if os.path.exists(baseline_out):
        os.remove(baseline_out)

    baseline_cso = read_rpt_file(baseline_rpt).outfall_loading_summary['Total_Volume_10^6 ltr'].sum() - read_rpt_file(baseline_rpt).outfall_loading_summary['Total_Volume_10^6 ltr'].iloc[1]

    Y = []
    for i, lid_state in enumerate(sample):
        print(f"Running simulation no.:  {i + 1}/{len(sample)}")
        sim_folder = os.path.join(base_folder, f"Sim_1")  
        os.makedirs(sim_folder, exist_ok=True)
        inp_file = os.path.join(sim_folder, base_model)
        rpt_file = inp_file.replace(".inp", ".rpt")
        out_file = inp_file.replace(".inp", ".out")

        shutil.copy(baseline_inp, inp_file)
        edit_inpfile(inp_file, start_date, end_date, lid_state)

        with Simulation(inp_file, reportfile=rpt_file, outputfile=out_file) as sim:
            for _ in sim:
                pass

        if os.path.exists(out_file):
            os.remove(out_file)

        cso = abs(baseline_cso - (read_rpt_file(rpt_file).outfall_loading_summary['Total_Volume_10^6 ltr'].sum() - read_rpt_file(rpt_file).outfall_loading_summary['Total_Volume_10^6 ltr'].iloc[1]))
        Y.append(cso)

    return np.array(Y, dtype=float), baseline_cso

# Run everything
base_folder = r"Q:\Abteilungsprojekte\eng\SWWData\Prabhat_Joshi\PhD\BGI_simulator\04_workspace\Spatial_analysis\Fehraltorf"
base_model = "basicModel_catchment.inp"
start_date = "05/09/1988"
end_date = "05/12/1988"

Y, baseline = run_sobol_simulations(base_folder, base_model, start_date, end_date)

# Run Sobol analysis
Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=True)

# Output results
df = pd.DataFrame({
    "BGI": problem["names"],
    "S1": Si["S1"],    
    "ST": Si["ST"],
    "S1_conf": Si["S1_conf"],
    "ST_conf": Si["ST_conf"]
})
print(df.sort_values(by="S1", ascending=False))

#%% Plot first-order vs total-order Sobol indices
plt.figure(figsize=(8, 6))
plt.scatter(df["S1"], df["ST"], s=100, alpha=0.7, edgecolor='k')
for i, label in enumerate(df["BGI"]):
    plt.text(df["S1"][i] + 0.01, df["ST"][i], label, fontsize=9)
plt.xlabel("First-order Sobol index (S1)", fontsize=12)
plt.ylabel("Total-order Sobol index (ST)", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Plot with error bars
x = np.arange(len(df["BGI"]))
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(x, df["S1"], yerr=df["S1_conf"], fmt='o', capsize=5, label="First-order (S1)")
ax.errorbar(x, df["ST"], yerr=df["ST_conf"], fmt='s', capsize=5, label="Total-order (ST)")
ax.set_xticks(x)
ax.set_xticklabels(df["BGI"], rotation=90)
ax.set_ylabel("Sobol Index")
ax.set_title("Sobol Sensitivity Indices with Confidence Intervals")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

#%% Save results
df.to_csv("sobol_sensitivity.csv", index=False)
