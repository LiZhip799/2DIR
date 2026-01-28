import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# Import refactored modules
from config import CONFIG
from dataset import PropertyDataset
from model import MaxViTFullModel
from losses import get_prop_loss

def main():
    # Initialize DDP
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device(f"cuda:{local_rank}")


    with open(CONFIG["PROPS"]["STATS_FILE"], 'r') as f:
        prop_stats = json.load(f)


    train_ds = PropertyDataset(CONFIG["PROPS"]["TRAIN_IMG"], CONFIG["PROPS"]["TRAIN_CSV"], prop_stats, is_train=True)
    test_ds = PropertyDataset(CONFIG["PROPS"]["TEST_IMG"], CONFIG["PROPS"]["TEST_CSV"], prop_stats, is_train=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], sampler=train_sampler, num_workers=CONFIG["NUM_WORKERS"])
    test_loader = DataLoader(test_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=CONFIG["NUM_WORKERS"])


    model = MaxViTFullModel(task_type='props').to(device)
    model = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR_BACKBONE"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=CONFIG["WARMUP_EPOCHS"] * len(train_loader), 
        num_training_steps=CONFIG["EPOCHS"] * len(train_loader)
    )
    scaler = GradScaler()

    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        for batch in tqdm(train_loader, disable=(rank != 0)):
            x = batch["pixel_values"].to(device)
            target_ss = batch["target_ss"].to(device) # Helix, strand, coil [cite: 70, 160]
            target_props = {t: batch["target_props"][t].to(device) for t in CONFIG["PROPS"]["PROP_TASKS"]}

            with autocast():
                pred_ss, pred_props = model(x)
                loss = get_prop_loss(pred_ss, target_ss, pred_props, target_props, CONFIG["PROPS"]["PROP_TASKS"])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()


        if rank == 0:
            torch.save(model.module.state_dict(), f"{CONFIG['PROPS']['OUTPUT_DIR']}/best_model.pth")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()