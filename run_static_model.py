# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:00:31 2025

@author: joshipra
"""

import os
import sys
import pandas as pd
from pyswmm import Simulation, Output, SystemSeries, NodeSeries


sys.path.append(r'C:\Users\joshipra\switchdrive\BGI_simulator\08_codes_and_functions')


base_dir = r"C:\Users\joshipra\switchdrive\BGI_simulator\04_workspace\SWMM_model"
folder_name = "Simulation_0"
folder_path = os.path.join(base_dir, folder_name)
os.chdir(folder_path)


base_name = "basicModel_catchment"
report_name = "rpt_basicModel_catchment_static"

inp_file = f"{base_name}.inp"
rpt_file = f"{report_name}.rpt"
out_file = f"{base_name}_catchment.out"

if os.path.exists(inp_file):
    with Simulation(inputfile=inp_file, reportfile=rpt_file, outputfile=out_file) as sim:
        print(f"Running simulation for {inp_file}...")        
               
        for step in sim:
            pass
        
    with Output(out_file) as out:
        ts11 = SystemSeries(out).flood_losses
        ts11_df = pd.DataFrame(list(ts11.items()), columns=["Time", "Flood_Volume"])
        ts11_df.to_csv("catchment_flood_static.csv")
        #print(ts11)        
        
        ts12 = NodeSeries(out)['CSO'].total_inflow
        ts12_df = pd.DataFrame(list(ts12.items()), columns=["Time", "Outfall_Volume"])
        ts12_df.to_csv("catchment_outfall_static.csv")
        
    print(f"Simulation for {inp_file} completed.")
    
    # Check and remove the .out file if it exists
    if os.path.exists(out_file):
        os.remove(out_file)
        print(f"Removed .out file: {out_file}")
else:
    print(f"File {inp_file} not found.")