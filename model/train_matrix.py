import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from config import CONFIG
from dataset import MatrixDataset
from model import MaxViTFullModel
from losses import MaskedL1Loss

def main():
    # Initialize DDP for distributed training [cite: 165, 178]
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device(f"cuda:{local_rank}")

    # Prepare datasets (188,967 pairs for structure task) [cite: 63, 157]
    train_ds = MatrixDataset(CONFIG["MATRIX"]["TRAIN_IMG"], CONFIG["MATRIX"]["TRAIN_CSV"], is_train=True)
    test_ds = MatrixDataset(CONFIG["MATRIX"]["TEST_IMG"], CONFIG["MATRIX"]["TEST_CSV"], is_train=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], sampler=train_sampler, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=CONFIG["NUM_WORKERS"])


    model = MaxViTFullModel(task_type='matrix').to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])


    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR_BACKBONE"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=CONFIG["WARMUP_EPOCHS"] * len(train_loader), 
        num_training_steps=CONFIG["EPOCHS"] * len(train_loader)
    )
    scaler = GradScaler()
    criterion = MaskedL1Loss() 

    best_loss = float('inf')
    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        
        # Training loop
        for imgs, targets, _ in tqdm(train_loader, disable=(rank != 0)):
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with autocast():
                # Inference of C-alpha distance maps [cite: 44, 175]
                preds = torch.abs(model(imgs))
                loss = criterion(preds, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()


        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets, _ in test_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                # Ensure symmetry for protein distance matrices [cite: 58, 64, 158]
                preds = model(imgs)
                preds = (preds + preds.transpose(-1, -2)) / 2 
                val_loss += criterion(preds, targets).item()
        
        if rank == 0:
            avg_val_loss = val_loss / len(test_loader)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.module.state_dict(), f"{CONFIG['MATRIX']['OUTPUT_DIR']}/best_model.pth")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()