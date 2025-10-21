# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 14:03:22 2025

@author: joshipra
"""

#%%
import os
import sys
import shutil
import pandas as pd
from pyswmm import Simulation, Output, SystemSeries, NodeSeries
from swmm_api import read_rpt_file
import matplotlib.pyplot as plt
 

sys.path.append(r'Q:\Abteilungsprojekte\eng\SWWData\Prabhat_Joshi\PhD\BGI_simulator\08_codes_and_functions')


#%%
main_folder = r"Q:\Abteilungsprojekte\eng\SWWData\Prabhat_Joshi\PhD\BGI_simulator\04_workspace\SWMM_model\Fehraltorf"
#rain_data = r"Q:\Abteilungsprojekte\eng\SWWData\Prabhat_Joshi\PhD\BGI_simulator\03_data\Kloten_B1_rainfall.txt"


# rain_df = pd.read_csv(rain_data)
# rain_df["Time"] = pd.to_datetime(rain_df["Time"], errors = "coerce")
# rain_dt = rain_df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

files_to_copy = ["basicModel_catchment.inp", "basicModel_catchment_Tr0.hsf"]

folder_name = "Simulation_0"

folder_path = os.path.join(main_folder, folder_name)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
os.chdir(folder_path)

for file_name in files_to_copy:
    src_file = os.path.join(main_folder, file_name)
    dest_file = os.path.join(folder_path, file_name)
    if os.path.exists(src_file):
        shutil.copy(src_file, dest_file)    

     
inp_file = "basicModel_catchment.inp"
rpt_file = "basicModel_catchment.rpt"
out_file = "basicModel_catchment.out"

# sim = Simulation(inp_file)
# sim.execute()

with Simulation(inp_file, reportfile=rpt_file, outputfile=None) as sim:        
    for step in sim:
        pass 

#%%
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
flood_annual_sum.to_csv("catchment_flood_ref.csv", index = False)
  
   
ts_outfall_df = pd.DataFrame(list(ts_outfall.items()), columns=["Time", "Outfall_Volume"])
#ts_outfall_df.to_csv("catchment_outfall_ref.csv", index = False)
ts_outfall_df["Time"] = pd.to_datetime(ts_outfall_df["Time"], errors = "coerce")
cso_dt = ts_outfall_df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
cso_dt["Outfall_Volume"] = cso_dt["Outfall_Volume"] * 0.6 # Convert to m3
cso_dt["Time"] = pd.to_datetime(cso_dt["Time"])
cso_dt.set_index("Time", inplace=True)
cso_annual_sum = cso_dt.resample('YE').sum()
cso_annual_sum["Time"] = cso_annual_sum.index.year
cso_annual_sum.to_csv("catchment_outfall_ref.csv", index = False) 

ts_ara_df = pd.DataFrame(list(ts_ara.items()), columns=["Time", "ARA"])
#ts_ara_df.to_csv("ara_outfall_ref.csv", index = False)
ts_ara_df["Time"] = pd.to_datetime(ts_ara_df["Time"], errors = "coerce")
ts_ara_dt = ts_ara_df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
ts_ara_dt["ARA"] = ts_ara_dt["ARA"] * 0.6 # Convert to m3
ts_ara_dt["Time"] = pd.to_datetime(ts_ara_dt["Time"])
ts_ara_dt.set_index("Time", inplace=True)
ara_annual_sum = ts_ara_dt.resample('YE').sum()
ara_annual_sum["Time"] = ara_annual_sum.index.year
ara_annual_sum.to_csv("ara_outfall_ref.csv", index = False) 

net_outfall = cso_annual_sum['Outfall_Volume'] - ara_annual_sum['ARA']
net_outfall.name = "Outfall_Volume"
net_outfall.index = net_outfall.index.year
net_outfall.to_csv("net_annual_outfall.csv", index = True)
