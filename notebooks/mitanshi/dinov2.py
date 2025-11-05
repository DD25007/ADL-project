import os
import cv2
import torch
import random
import requests
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import timm
from safetensors.torch import load_file

def stream_video(url, max_frames=200):
    """Stream a video from a remote Flask server and extract frames."""
    cap = cv2.VideoCapture(url)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

""" video list from server"""

BASE_URL = "http://172.18.43.247:5001"
videos = requests.get(f"{BASE_URL}/list").json()["videos"]
print("Total videos:", len(videos))

random.seed(42)
random.shuffle(videos)
n = len(videos)
train_videos = videos[:int(0.7*n)]
val_videos = videos[int(0.7*n):int(0.85*n)]
test_videos = videos[int(0.85*n):]

"""Dataset class for remote video"""

class RemoteVideoDataset(Dataset):
    def __init__(self, base_url, video_list, frames_per_video=100, transform=None):
        self.base_url = base_url
        self.video_list = video_list
        self.frames_per_video = frames_per_video
        self.transform = transform or T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(518, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name = self.video_list[idx]
        url = f"{self.base_url}/video/{video_name}"
        cap = cv2.VideoCapture(url)
        frames = []
        while cap.isOpened() and len(frames) < self.frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            raise ValueError(f"Empty video: {video_name}")
        frame = random.choice(frames)
        return self.transform(frame)


train_loader = DataLoader(RemoteVideoDataset(BASE_URL, train_videos), batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(RemoteVideoDataset(BASE_URL, val_videos), batch_size=8)

"""Load DINOv2 model"""

device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False)
weight_path = "/home/teaching/ADL-project/notebooks/mitanshi/model.safetensors"
state_dict = load_file(weight_path)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
print("DINOv2 model loaded successfully!")

"""proj+predictor heads"""

proj_head = nn.Sequential(
    nn.Linear(model.num_features, 512),
    nn.ReLU(),
    nn.Linear(512, 128)
).to(device)

predictor = nn.Sequential(
    nn.Linear(128, 512),
    nn.ReLU(),
    nn.Linear(512, 128)
).to(device)

criterion = nn.CosineSimilarity(dim=-1)
optimizer = torch.optim.Adam(
    list(proj_head.parameters()) + list(predictor.parameters()),
    lr=1e-4
)

"""training"""
from sklearn.metrics import accuracy_score, roc_auc_score
train_losses, val_losses, val_accs, val_aucs = [], [], [], []

def get_fake_labels(batch_size):
    return torch.randint(0, 2, (batch_size,), device=device)
clf_head = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
).to(device)
clf_opt = torch.optim.Adam(clf_head.parameters(), lr=1e-4,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(clf_opt, step_size=3, gamma=0.5)

for epoch in range(10):
    clf_head.train()
    train_loss = 0

    for batch in train_loader:
        videos, labels = batch[0]
        videos, labels = videos.to(device), labels.to(device)

        with torch.no_grad():  # Freeze DINO backbone
            feats = model(videos)
        
        preds = clf_head(feats)
        loss = criterion(preds, labels)

        clf_opt.zero_grad()
        loss.backward()
        clf_opt.step()
        train_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1} | TrainLoss={train_loss/len(train_loader):.4f}")

    #validation
    model.eval(); proj_head.eval(); clf_head.eval()
    with torch.no_grad():
        all_preds,all_labels=[],[]
        val_loss=0
        for imgs in val_loader:
            imgs=imgs.to(device)
            feats=model.forward_features(imgs)
            z=proj_head(feats.mean(dim=1))
            logits=clf_head(z)
            y=get_fake_labels(imgs.size(0))
            loss=F.cross_entropy(logits,y)
            val_loss+=loss.item()
            preds=torch.softmax(logits,dim=1)[:,1].detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
        val_losses.append(val_loss/len(val_loader))
        acc=accuracy_score(all_labels,np.array(all_preds)>0.5)
        auc=roc_auc_score(all_labels,all_preds)
        val_accs.append(acc)
        val_aucs.append(auc)
    print(f"Epoch {epoch+1} | Loss={train_losses[-1]: .3f} | ValAcc={acc:.3f}")

epochs = range(1, len(train_losses)+1)

plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.legend(); plt.title("Loss Curve")

plt.subplot(1,3,2)
plt.plot(epochs, val_accs, label="Accuracy", color="g")
plt.legend(); plt.title("Validation Accuracy")

plt.subplot(1,3,3)
plt.plot(epochs, val_aucs, label="AUC", color="m")
plt.legend(); plt.title("Validation AUC")

plt.tight_layout()
plt.savefig("stage1_metrics.png")
print("Saved: stage1_metrics.png")

"""Attention"""

def get_attention_map(model, frame):
    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
    ])
    model.eval()
    img = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
    attn_maps = []

    def hook_fn(module, input, output):
        B, N, C = input[0].shape
        qkv = module.qkv(input[0])
        qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (q.shape[-1] ** 0.5))
        attn = attn.softmax(dim=-1)
        attn_maps.append(attn.detach().cpu())

    handle = model.blocks[-1].attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(img)
    handle.remove()

    attn = attn_maps[0].mean(1)[0]
    size = int((attn.shape[-1] - 1) ** 0.5)
    attn_cls = attn[0, 1:].reshape(size, size)
    attn_cls = (attn_cls - attn_cls.min()) / (attn_cls.max() - attn_cls.min())
    attn_cls = cv2.resize(attn_cls.numpy(), frame.shape[:2][::-1])
    return attn_cls

"""visualization"""
sample_video = f"{BASE_URL}/video/{test_videos[2]}"
frames = stream_video(sample_video, max_frames=100)
print("Loaded", len(frames), "frames")

attn_map = get_attention_map(model, frames[50])
print("Attention map stats:")
print(f"min={attn_map.min():.6f}, max={attn_map.max():.3f}, mean={attn_map.mean():.6f}")

overlay = cv2.addWeighted(
    cv2.cvtColor(frames[50], cv2.COLOR_RGB2BGR),
    0.6,
    cv2.applyColorMap((attn_map * 255).astype(np.uint8), cv2.COLORMAP_JET),
    0.4,
    0
)

cv2.imwrite("attention_overlay.png", overlay)
print("Saved overlay -> attention_overlay.png")

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("DINOv2 Attention Overlay")
plt.axis("off")
plt.show()



