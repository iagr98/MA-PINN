import numpy as np
from scipy.optimize import newton
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import torch

# Funktion berechnet Höhe eines Kreissegments auf Basis des Kreisradius r und der Fläche A
def getHeight(A, r):
    eq = lambda h: A - r**2 * np.arccos(1 - h / r) + (r - h) * np.sqrt(2 * r * h - h**2)
    h0 = r / 2
    if A < 0:
        #print('Querschnitt kleiner Null: ' + str(A))
        return 0
    elif A > np.pi * r**2:
        #print('Querschnitt größer als zulässig: ' + str(A))
        return 2*r
    return newton(eq, h0)

def getHeightArray(A, r):
    h = np.zeros_like(A)
    for i in range(len(h)):
        h[i] = getHeight(A[i], r)
    return h

# Funktion berechnet die Fläche eines Kreissegments auf Basis des Kreisradiuses r und der Höhe h des Segments
def getArea(h, r):
    return r**2 * np.arccos(1 - h / r) - (r - h) * np.sqrt(2 * r * h - h**2)

def model_prediction(model, dV_ges, eps_0, phi_0, x, X_min, X_max, Y_min, Y_max, col_nonzero):
    # Build input array for all x values:
    if isinstance(x, (int, float)):
        X_input_dim = np.array([
            dV_ges / 3.6 * 1e-6,
            eps_0,
            phi_0,
            x
        ], dtype=np.float32)
        # Min-max normalization for input
        X_input_norm = (X_input_dim - X_min) / (X_max - X_min + 1e-8)

        # Predict
        model.eval()
        with torch.no_grad():
            input_tensor = torch.from_numpy(X_input_norm).unsqueeze(0).float()
            y_pred_norm = model(input_tensor).numpy().squeeze()
            y_pred_dim = y_pred_norm * (Y_max - Y_min) + Y_min

        # Mapping output names for user
        base_outputs = ["V_dis", "V_c", "phi_32"] + [f"N_{j}" for j in range(20)]
        kept_outputs = [name for i, name in enumerate(base_outputs) if col_nonzero[i]]
        result = {name: val for name, val in zip(kept_outputs, y_pred_dim)}
    else:
        X_input_dim = np.column_stack([
            np.full_like(x, dV_ges / 3.6 * 1e-6),
            np.full_like(x, eps_0),
            np.full_like(x, phi_0),
            x
        ])
        X_input_norm = (X_input_dim - X_min) / (X_max - X_min + 1e-8)

        model.eval()  # set model to eval mode
        
        with torch.no_grad():
            input_tensor = torch.from_numpy(X_input_norm).unsqueeze(0).float()  # shape (1,4)
            y_pred_norm = model(input_tensor).numpy().squeeze()   # shape (n_outputs,)
            y_pred_dim = y_pred_norm * (Y_max - Y_min) + Y_min
         # ---- Build output as dict (for user clarity) ----
        # Reconstruct the kept output names
        base_outputs = ["V_dis", "V_c", "phi_32"] + [f"N_{j}" for j in range(20)]
        kept_outputs = [name for i, name in enumerate(base_outputs) if col_nonzero[i]]

        result = {name: y_pred_dim[:,i] for i, name in enumerate(kept_outputs)}     

    return result