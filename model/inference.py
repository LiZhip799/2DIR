import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import refactored modules
from config import CONFIG
from dataset import MatrixDataset, PropertyDataset
from model import MaxViTFullModel

@torch.no_grad()
def run_matrix_inference(model, loader, device, output_dir):
    """
    Generate C-alpha distance maps for 3D structure reconstruction.
    Outputs are saved as .npy files for each protein sample.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    for imgs, _, ids in tqdm(loader, desc="Inference: Matrix"):
        imgs = imgs.to(device)
        preds = model(imgs)
        preds = torch.abs((preds + preds.transpose(-1, -2)) / 2)
        
        preds_np = preds.squeeze(1).cpu().numpy()
        for i, sample_id in enumerate(ids):
            np.save(os.path.join(output_dir, f"{sample_id}.npy"), preds_np[i])

@torch.no_grad()
def run_property_inference(model, loader, device, prop_stats, output_path):
    """
    Infer physicochemical descriptors including secondary structure, Rg, and Hbonds[cite: 11, 29].
    Results are aggregated into a single CSV file.
    """
    model.eval()
    results = []

    for batch in tqdm(loader, desc="Inference: Properties"):
        x = batch["pixel_values"].to(device)
        ids = batch["id"]
        
        pred_ss, pred_props = model(x)
        pred_ss = pred_ss.cpu().numpy()
        
        for i, sample_id in enumerate(ids):
            item = {"ID": sample_id}

            for j, name in enumerate(["Helix", "Strand", "Coil"]):
                item[f"Pred_{name}"] = pred_ss[i, j]

            for task in CONFIG["PROPS"]["PROP_TASKS"]:
                mean_v, std_v = prop_stats.get(task, (0, 1))
                # Denormalize predictions using reference statistics [cite: 163]
                item[f"Pred_{task}"] = pred_props[task][i].item() * std_v + mean_v
            
            results.append(item)
            
    pd.DataFrame(results).to_csv(output_path, index=False)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Structure (Matrix) Inference ---
    matrix_model = MaxViTFullModel(task_type='matrix').to(device)
    matrix_model.load_state_dict(torch.load(f"{CONFIG['MATRIX']['OUTPUT_DIR']}/best_model.pth"))
    
    matrix_ds = MatrixDataset(CONFIG["MATRIX"]["TEST_IMG"], CONFIG["MATRIX"]["TEST_CSV"], is_train=False)
    matrix_loader = DataLoader(matrix_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=CONFIG["NUM_WORKERS"])
    
    matrix_save_dir = os.path.join(CONFIG["MATRIX"]["OUTPUT_DIR"], "npy_predictions")
    run_matrix_inference(matrix_model, matrix_loader, device, matrix_save_dir)

    # --- 2. Physicochemical Property Inference ---
    with open(CONFIG["PROPS"]["STATS_FILE"], 'r') as f:
        prop_stats = json.load(f)
        
    prop_model = MaxViTFullModel(task_type='props').to(device)
    prop_model.load_state_dict(torch.load(f"{CONFIG['PROPS']['OUTPUT_DIR']}/best_model.pth"))
    
    prop_ds = PropertyDataset(CONFIG["PROPS"]["TEST_IMG"], CONFIG["PROPS"]["TEST_CSV"], prop_stats, is_train=False)
    prop_loader = DataLoader(prop_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=CONFIG["NUM_WORKERS"])
    
    prop_csv_path = os.path.join(CONFIG["PROPS"]["OUTPUT_DIR"], "property_predictions.csv")
    run_property_inference(prop_model, prop_loader, device, prop_stats, prop_csv_path)

    print(f"Inference Complete. Matrix saved to {matrix_save_dir}, Properties saved to {prop_csv_path}")

if __name__ == "__main__":
    main()