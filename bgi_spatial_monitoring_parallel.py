# -*- coding: utf-8 -*-
"""
Created on Wed May 21 14:43:46 2025

@author: joshipra
"""

import os
import shutil
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from pyswmm import Simulation
from swmm_api import read_rpt_file
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt

# ----------------------------- Setup Problem --------------------------------

num_lids = 14
problem = {
    'num_vars': num_lids,
    'names': [f'BGI_{i+1}' for i in range(num_lids)],
    'bounds': [[1, 5]] * num_lids
}

N = 100  # Number of base samples
sample = saltelli.sample(problem, N=N)
sample = np.round(sample).astype(int)

# -------------------------- Simulation Function -----------------------------

def simulate_case(args):
    i, lid_state, base_folder, base_model, baseline_inp, start_date, end_date = args

    sim_folder = os.path.join(base_folder, f"Sim_{i}")
    os.makedirs(sim_folder, exist_ok=True)
    inp_file = os.path.join(sim_folder, base_model)
    rpt_file = inp_file.replace(".inp", ".rpt")
    out_file = inp_file.replace(".inp", ".out")

    shutil.copy(baseline_inp, inp_file)

    with open(inp_file, 'r') as file:
        lines = file.readlines()

    start_line, end_line = 842, 855
    with open(inp_file, 'w') as file:
        for j, line in enumerate(lines):
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
            elif start_line - 1 <= j <= end_line - 1:
                tokens = line.split()
                if len(tokens) >= 9:
                    lid_index = j - (start_line - 1)
                    subcatchment = tokens[0]
                    tokens[1] = f"BR_CS{lid_state[lid_index]}"
                    tokens[8] = f'"LID_{subcatchment}_Tr1.txt"'
                    file.write(' '.join(tokens) + '\n')
                else:
                    file.write(line)
            else:
                file.write(line)

    with Simulation(inp_file, reportfile=rpt_file, outputfile=out_file) as sim:
        for _ in sim:
            pass

    if os.path.exists(out_file):
        os.remove(out_file)

    cso = read_rpt_file(rpt_file).outfall_loading_summary['Total_Volume_10^6 ltr'][1]
    return cso

# ----------------------------- Parallel Runner ------------------------------

def run_parallel_simulations(sample, base_folder, base_model, start_date, end_date):
    baseline_inp = os.path.join(base_folder, "baseline.inp")
    if not os.path.exists(baseline_inp):
        shutil.copy(os.path.join(base_folder, base_model), baseline_inp)

    args_list = [(i, lid_state, base_folder, base_model, baseline_inp, start_date, end_date)
                 for i, lid_state in enumerate(sample)]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(simulate_case, args_list)

    return np.array(results, dtype=float)

# ----------------------------- Run Everything ------------------------------

if __name__ == "__main__":
    base_folder = r"Q:\Abteilungsprojekte\eng\SWWData\Prabhat_Joshi\PhD\BGI_simulator\04_workspace\Spatial_analysis\Fehraltorf"
    base_model = "basicModel_catchment.inp"
    start_date = "01/01/2009"
    end_date = "21/31/2009"

    Y = run_parallel_simulations(sample, base_folder, base_model, start_date, end_date)

    # Perform Sobol analysis
    Si = sobol.analyze(problem, Y, print_to_console=True)

    df = pd.DataFrame({
        "BGI": problem["names"],
        "S1": Si["S1"],
        "ST": Si["ST"],
        "S1_conf": Si["S1_conf"],
        "ST_conf": Si["ST_conf"]
    })

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df["S1"], df["ST"], s=100, alpha=0.7, edgecolor='k')
    for i, label in enumerate(df["BGI"]):
        plt.text(df["S1"][i] + 0.01, df["ST"][i], label, fontsize=9)

    plt.xlabel("First-order Sobol index (S1)\nMain effect")
    plt.ylabel("Total-order Sobol index (ST)\nIncludes interactions")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
