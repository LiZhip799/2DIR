import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MatrixHead(nn.Module):
    """Decoder for 3D structure reconstruction [cite: 48, 175]"""
    def __init__(self, in_dim, out_size=100):
        super().__init__()
        self.fc = nn.Linear(in_dim, 128 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(size=(out_size, out_size), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.fc(x).view(-1, 128, 7, 7)
        return self.decoder(x)

class PropertyHead(nn.Module):
    """Heads for SS% and descriptors like Rg or Hbonds [cite: 176]"""
    def __init__(self, feat_dim, prop_tasks):
        super().__init__()
        self.ss_head = nn.Linear(feat_dim, 3)
        self.prop_heads = nn.ModuleDict({
            task: nn.Linear(feat_dim, 1) for task in prop_tasks
        })

    def forward(self, feat):
        ss_probs = F.softmax(self.ss_head(feat), dim=-1)
        prop_preds = {task: head(feat).squeeze(-1) for task, head in self.prop_heads.items()}
        return ss_probs, prop_preds

class MaxViTFullModel(nn.Module):
    def __init__(self, task_type='matrix'):
        super().__init__()
        self.backbone = timm.create_model(CONFIG["MODEL_NAME"], pretrained=True, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        
        if task_type == 'matrix':
            self.head = MatrixHead(feat_dim)
        else:
            self.head = PropertyHead(feat_dim, CONFIG["PROPS"]["PROP_TASKS"])

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)