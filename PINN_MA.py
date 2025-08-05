import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ==== 1. Load & Preprocess Data ====

df = pd.read_csv("Input/in_silico_dataset.csv")

input_list = []
output_list = []

N_x = 201
N_D = 20

for idx, row in df.iterrows():
    # Inputs
    dV_ges = float(row["dV_ges"])
    eps_0 = float(row["eps_0"])
    phi_0 = float(row["phi_0"])
    x_arr = np.array([float(v) for v in row["x"].split(",")])                # shape (N_x,)
    # Outputs
    V_dis_arr = np.array([float(v) for v in row["V_dis"].split(",")])        # (N_x,)
    V_c_arr   = np.array([float(v) for v in row["V_c"].split(",")])          # (N_x,)
    phi32_arr = np.array([float(v) for v in row["phi_32"].split(",")])       # (N_x,)
    N_arrs    = [np.array([float(v) for v in row[f"N_{j}"].split(",")]) for j in range(N_D)]  # list of (N_x,)

    for i in range(N_x):
        inp = [dV_ges, eps_0, phi_0, x_arr[i]]
        out = [V_dis_arr[i], V_c_arr[i], phi32_arr[i]] + [N_arrs[j][i] for j in range(N_D)]  # total 23 outputs
        input_list.append(inp)
        output_list.append(out)

X = np.array(input_list, dtype=np.float32)
Y = np.array(output_list, dtype=np.float32)
Y = np.maximum(Y, 0)  # Set negative values to 0

print("X shape:", X.shape)  # (100*N_x, 4)
print("Y shape:", Y.shape)  # (100*N_x, 23)