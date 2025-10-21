# -*- coding: utf-8 -*-
"""
Created on Fri May 16 20:08:55 2025

@author: joshipra
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

#%%
basedir = r"Q:\Abteilungsprojekte\eng\SWWData\Prabhat_Joshi\PhD\BGI_simulator\04_workspace\Spatial_analysis\Fehraltorf"
os.chdir(basedir)
sobol = pd.read_csv("sobol_sensitivity.csv")
spatial_summary = pd.read_csv("spatial_analysis.csv")

merged_df = pd.merge(spatial_summary, sobol, on = "name")

# # Step 1: Manually enter the data
# data = {
#     "BGI area": [6655, 25000, 18000, 20000],
#     "Slope": [0.5, 0.5, 0.5, 0.5],
#     "Capture Ratio": [0.751314801, 0.9, 1.388888889, 1.5],
#     "Distance to CSO": [1200, 1200, 800, 800],
#     "Elevation": [1.2, 1.2, 1.1, 1.1],
#     "ST": [0.03, 0.43, 0.24, 0.3]
# }

custom_index = [f"S{i+1}" for i in range(53)]

data = pd.DataFrame({
    "BGI area": merged_df["bgi_area_m2"].tolist(),
    #"BGI width": merged_df["width_m"].tolist(),
    "Slope": merged_df["slope"].tolist(),
    "Capture Ratio": merged_df["capture_ratio"].tolist(),
    "Distance to nearest CSO": merged_df["distance_to_nearest_cso_m"].tolist(),
    "Elevation": merged_df["elevation_m"].tolist(),
    "ST": merged_df["ST"].tolist()
}, index = custom_index)


# Step 2: Split into features and target
X = data.drop(columns=["ST"])
y = data["ST"]

# Step 3: Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=1000, random_state=45)
model.fit(X, y)

# Step 4: Extract feature importances
importances = model.feature_importances_
features = X.columns

# Step 5: Create a summary DataFrame
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Step 6: Plot feature importances
plt.figure(figsize=(8, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"], color='#99d8c9')
plt.xlabel("Feature Importance", fontname='Calibri',  fontsize=14)
#plt.title("Random Forest Feature Importance for Predicting ST", fontname='Calibri')
plt.grid(axis='x', linestyle='--', linewidth=0.7, alpha=0.7)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Optional: Print the importance values
print(importance_df)

#%%
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.utils import resample

def stability_selection(X, y, n_boot=200, subsample=0.8, random_state=45):
    rng = np.random.RandomState(random_state)
    sel_counts = np.zeros(X.shape[1], dtype=int)
    for b in range(n_boot):
        idx = rng.choice(len(y), size=int(subsample*len(y)), replace=False)
        en = ElasticNetCV(l1_ratio=[0.2,0.5,0.8,1.0], cv=5, random_state=rng, max_iter=10000)
        en.fit(X[idx], y[idx])
        sel_counts += (np.abs(en.coef_) > 1e-8)
    return sel_counts / n_boot  # selection frequency per feature
