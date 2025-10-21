# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 18:30:31 2025

@author: joshipra
"""

import pandas as pd
import glob
import os

os.chdir(r'C:\Users\joshipra\switchdrive\BGI_simulator\04_workspace\SWMM_model\Simulation_1')

# Read the file. Adjust skiprows if your header spans a different number of lines.

files = glob.glob("LID_S4_Tr*.txt")
dfs = []
for f in files:
    df = pd.read_csv(f, delim_whitespace=True, skiprows=9, header=None)
    df.columns = ["Date", "Time", "Elapsed", "Inflow", "Evaporation", "Infiltration",
                  "PavementPerc", "SoilPerc", "Exfiltration", "SurfaceRunoff", "DrainOutflow",
                  "SurfaceLevel", "PavementLevel", "SoilContent", "StorageLevel"]
    df.fillna(0, inplace=True)
    df["Date"] = df["Date"].replace("", "0")
    df["Time"] = df["Time"].replace("", "0")
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], 
                                      format="%m/%d/%Y %H:%M:%S", errors="coerce").fillna("0")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
result = combined[["Datetime", "Inflow", "Evaporation", "SurfaceRunoff", "DrainOutflow"]]
print(result.head())
