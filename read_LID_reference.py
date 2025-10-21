# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 18:18:19 2025

@author: joshipra
"""

import os
import pandas as pd



os.chdir(r'C:\Users\joshipra\switchdrive\BGI_simulator\04_workspace\SWMM_model\Simulation_0')

# Read the file. Adjust skiprows if your header spans a different number of lines.
df = pd.read_csv("LID_S4_Tr1.txt", delim_whitespace=True, skiprows=9, header=None)

# Assign names to all 14 columns based on the file's header layout.
df.columns = [
    "Date", "Time", "Elapsed", "Inflow", "Evaporation", "Infiltration", 
    "PavementPerc", "SoilPerc", "Exfiltration", "SurfaceRunoff", "DrainOutflow", 
    "SurfaceLevel", "PavementLevel", "SoilContent", "StorageLevel"
]

# Replace any missing values in the entire dataframe with 0.
df.fillna(0, inplace=True)

# For rows with missing date or time (i.e. blank strings), replace them with "0"
# (so that the combination results in a "0" if conversion fails).
df['Date'] = df['Date'].replace("", "0")
df['Time'] = df['Time'].replace("", "0")

# Create a new datetime column by combining date and time.
# errors='coerce' will yield NaT (Not a Time) for any bad conversion.
df['Datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'], format="%m/%d/%Y %H:%M:%S", errors='coerce')

# As requested, for any rows where the date (and hence datetime) is missing, fill with 0.
# Note: this converts the datetime column to an object type.
df['Datetime'] = df['Datetime'].fillna("0")

# Extract the desired columns:
# We want the combined datetime, the Inflow (column 4, index 3), 
# Evaporation (column 5, index 4), Surface Runoff (column 10, index 9)
# and Drain Outflow (column 11, index 10).
result = df[['Datetime', 'Inflow', 'Evaporation', 'SurfaceRunoff', 'DrainOutflow']]

# Display the first few rows
print(result.head())
