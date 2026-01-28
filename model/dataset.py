import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
from pathlib import Path

class BaseDataset(Dataset):
    def __init__(self, is_train=False):
        self.is_train = is_train
        self.mean = torch.tensor(CONFIG["IMAGENET_MEAN"]).view(3, 1, 1)
        self.std = torch.tensor(CONFIG["IMAGENET_STD"]).view(3, 1, 1)

    def _process_image(self, img):
        if self.is_train:
            if np.random.random() < 0.5:
                img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.9, 1.1))
        
        img_np = np.array(img)
        if len(img_np.shape) == 2: img_np = np.stack([img_np]*3, axis=-1)
        elif img_np.shape[2] == 4: img_np = img_np[:, :, :3]
        
        tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
        return (tensor - self.mean) / self.std

class MatrixDataset(BaseDataset):
    """Dataset for C-alpha distance map prediction"""
    def __init__(self, img_folder, csv_folder, is_train=False):
        super().__init__(is_train)
        self.img_folder = img_folder
        self.csv_folder = csv_folder
        self.file_list = [f for f in os.listdir(img_folder) if f.lower().endswith('.png')]

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        prefix = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.img_folder, img_name)
        csv_path = os.path.join(self.csv_folder, f"{prefix}.csv")

        # Process Image
        with Image.open(img_path) as im:
            img = im.convert("RGB").resize((224, 224), Image.BILINEAR)
        img_tensor = self._process_image(img)

        # Process Matrix [cite: 158]
        try:
            matrix = pd.read_csv(csv_path, skiprows=1, header=None).values.astype(np.float32)
            # Resize logic for sequence lengths up to 100 [cite: 44]
            target = np.zeros((100, 100), dtype=np.float32)
            h, w = min(matrix.shape[0], 100), min(matrix.shape[1], 100)
            target[:h, :w] = matrix[:h, :w]
            matrix_tensor = torch.tensor(target).unsqueeze(0)
        except:
            matrix_tensor = torch.zeros((1, 100, 100))

        return img_tensor, matrix_tensor, prefix

class PropertyDataset(BaseDataset):
    """Dataset for Secondary Structure and Physicochemical properties"""
    def __init__(self, img_dir, csv_path, prop_stats, is_train=False):
        super().__init__(is_train)
        self.img_dir = Path(img_dir)
        self.df = pd.read_csv(csv_path)
        self.prop_stats = prop_stats

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row["id"])
        img_path = self.img_dir / f"{img_id}.png"

        with Image.open(img_path) as im:
            img = im.convert("RGB").resize((224, 224), Image.BILINEAR)
        pixel_values = self._process_image(img)

        # Secondary Structure [cite: 160]
        ss_vals = row[CONFIG["PROPS"]["SS_COLS"]].values.astype(np.float32) / 100.0

        # Physicochemical Properties [cite: 161, 162]
        prop_targets = {}
        for task in CONFIG["PROPS"]["PROP_TASKS"]:
            val = float(row[task])
            mean_v, std_v = self.prop_stats.get(task, (0, 1))
            prop_targets[task] = torch.tensor((val - mean_v) / (std_v + 1e-8), dtype=torch.float32)

        return {"pixel_values": pixel_values, "target_ss": torch.from_numpy(ss_vals), "target_props": prop_targets, "id": img_id}