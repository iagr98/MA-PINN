import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os

# ==== 1. Load & Preprocess in-silico data for data-driven model ====

def create_normalized_data(filename):

    df = pd.read_csv(os.path.join("Input", filename))
    input_list = []
    output_list = []
    N_x = 201
    N_D = 20
    for idx, row in df.iterrows():
        # Inputs
        dV_ges = float(row["dV_ges"]) / 3.6 * 1e-6
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

    # --- Remove all-zero output columns automatically ---
    col_nonzero = np.any(Y != 0, axis=0)
    Y = Y[:, col_nonzero]
    print("Kept output columns:", np.where(col_nonzero)[0])  # Save mapping for later

    # Min-max normalization for inputs
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)

    # Min-max normalization for outputs
    Y_min, Y_max = Y.min(axis=0), Y.max(axis=0)
    Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-8)

    print("X_norm shape:", X_norm.shape)
    print("Y_norm shape:", Y_norm.shape)
    return X_norm, Y_norm


# ==== 2. Create Torch Dataset and NN model ====


class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
class DNN(nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"layer{i}", nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.net.add_module(f"tanh{i}", nn.Tanh())
            else:
                self.net.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        return self.net(x)
    

# ==== 3. Initiallize and train the model ====

if __name__ == "__main__":

    filename = "in_silico_dataset.csv"
    X_norm, Y_norm = create_normalized_data(filename)

    dataset = MyDataset(X_norm, Y_norm)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    layers = [X_norm.shape[1], 128, 128, 128, Y_norm.shape[1]]
    model = DNN(layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_data = nn.MSELoss()

    # optimizer.param_groups[0]['lr'] = 1e-5

    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_data(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1:05d}/{epochs:05d}, Loss: {total_loss/len(dataset):.6e}")