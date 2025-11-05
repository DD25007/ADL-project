# import os
# import cv2
# import torch
# import random
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as T
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import matplotlib.pyplot as plt
# import timm
# from safetensors.torch import load_file
# from sklearn.metrics import accuracy_score, roc_auc_score

# import os, torch, random
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import torchvision.transforms as T

# class BUSIDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.samples = []
#         for label, cls in enumerate(["benign", "malignant"]):
#             folder = os.path.join(root_dir, cls)
#             for fname in os.listdir(folder):
#                 if fname.endswith(".png") and "mask" not in fname.lower():
#                     self.samples.append((os.path.join(folder, fname), label))
#         self.transform = transform or T.Compose([
#             T.Resize((518,518)),
#             T.ToTensor(),
#             T.Normalize(mean=[0.5]*3, std=[0.5]*3),
#         ])
#     def __len__(self):
#         return len(self.samples)
#     def __getitem__(self, idx):
#         img_path, label = self.samples[idx]
#         img = Image.open(img_path).convert("RGB")
#         return self.transform(img), torch.tensor(label)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# root_dir = "/home/teaching/ADL-project/data/Dataset_BUSI_with_GT"

# dataset = BUSIDataset(root_dir)
# train_len = int(0.7*len(dataset))
# val_len   = int(0.2*len(dataset))
# test_len  = len(dataset) - train_len - val_len
# train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset,[train_len,val_len,test_len])
# train_loader = DataLoader(train_ds,batch_size=8,shuffle=True,num_workers=4)
# val_loader   = DataLoader(val_ds,batch_size=8)
# model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False)
# model.to(device)

# clf_head = nn.Sequential(
#     nn.Linear(model.num_features, 256),
#     nn.ReLU(),
#     nn.Dropout(0.4),
#     nn.Linear(256, 2)
# ).to(device)

# opt = torch.optim.Adam(clf_head.parameters(), lr=1e-4, weight_decay=1e-4)


# checkpoint = torch.load("/home/teaching/ADL-project/notebooks/mitanshi/busi_full_model.pth")
# model.load_state_dict(checkpoint['backbone'])
# clf_head.load_state_dict(checkpoint['classifier'])
# opt.load_state_dict(checkpoint['optimizer'])

# prompt_vectors = {
#     "benign": torch.tensor([1.0, 0.0]).to(device),
#     "malignant": torch.tensor([0.0, 1.0]).to(device)
# }


# def predict_with_prompt(img):
#     model.eval()
#     with torch.no_grad():
#         feat = model.forward_features(img.unsqueeze(0).to(device)).mean(dim=1)
#         logits = clf_head(feat)
#         probs = torch.softmax(logits, dim=1).squeeze()
#         pred = "malignant" if probs[1] > probs[0] else "benign"
#     return pred, probs.cpu().numpy()
# import random
# plt.figure(figsize=(10,5))
# for i in range(4):
#     img, label = random.choice(test_ds)
#     pred, probs = predict_with_prompt(img)
#     plt.subplot(1,4,i+1)
#     plt.imshow((img.permute(1,2,0).cpu().numpy()*0.5+0.5))
#     plt.title(f"Pred: {pred}\nConf: {probs.max():.2f}")
#     plt.axis("off")
# plt.tight_layout()
# plt.savefig("busi_predictions.png")

# correct = 0
# total = 0
# clf_head.eval()
# with torch.no_grad():
#     for imgs, labels in DataLoader(test_ds, batch_size=8):
#         feats = model.forward_features(imgs.to(device)).mean(dim=1)
#         logits = clf_head(feats)
#         preds = torch.argmax(logits, dim=1)
#         correct += (preds.cpu() == labels).sum().item()
#         total += len(labels)

# accuracy = correct / total
# print(accuracy)

# from sklearn.metrics import classification_report
# print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))

# def get_bbox_from_mask(mask):
#     ys, xs = np.where(mask > 0)
#     if len(xs) == 0:  # empty mask fallback
#         return None
#     return (xs.min(), ys.min(), xs.max(), ys.max())

# import cv2

# def draw_bbox(img, bbox, color=(0,255,0), label=None):
#     img = np.uint8((img.permute(1,2,0).cpu().numpy()*127.5 + 127.5))
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     if bbox:
#         x1,y1,x2,y2 = bbox
#         cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
#         if label:
#             cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# mask = np.array(Image.open(mask_path).convert("L"))
# bbox = get_bbox_from_mask(mask)
# img_with_box = draw_bbox(img, bbox, label=f"Pred: {pred}\nConf: {probs.max():.2f}")
# prompt_vectors = {
#     "benign": torch.tensor([1.0, 0.0]).to(device),
#     "malignant": torch.tensor([0.0, 1.0]).to(device)
# }

# feat = model.forward_features(img.unsqueeze(0).to(device)).mean(dim=1)
# similarity = torch.softmax(torch.matmul(feat, torch.stack(list(prompt_vectors.values())).T), dim=1)

# feats = model.forward_features(img.unsqueeze(0).to(device))
# attn_map = feats[0].mean(0).reshape(14,14).cpu().numpy()
# attn_map = cv2.resize(attn_map, (img.shape[2], img.shape[1]))
# plt.imshow((img.permute(1,2,0).cpu().numpy()*0.5+0.5))
# plt.imshow(attn_map, cmap='jet', alpha=0.4)
# plt.title(f"Pred: {pred}")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import timm
from safetensors.torch import load_file
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import random
import cv2

# -------------------------
# Dataset Definition
# -------------------------
class BUSIDataset(Dataset):
    def __init__(self, root_dir, transform=None, include_mask=True):
        self.samples = []
        for label, cls in enumerate(["benign", "malignant"]):
            folder = os.path.join(root_dir, cls)
            for fname in os.listdir(folder):
                if fname.endswith(".png") and "mask" not in fname.lower():
                    mask_path = os.path.join(folder, fname.replace(".png", "_mask.png"))
                    mask_path = mask_path if os.path.exists(mask_path) else None
                    self.samples.append((os.path.join(folder, fname), mask_path, label))
        self.transform = transform or T.Compose([
            T.Resize((518,518)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
        self.include_mask = include_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = None
        if self.include_mask and mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (518, 518))
        return self.transform(img), torch.tensor(label), mask, os.path.basename(img_path)

# -------------------------
# Load Data
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
root_dir = "/home/teaching/ADL-project/data/Dataset_BUSI_with_GT"
dataset = BUSIDataset(root_dir)
train_len = int(0.7*len(dataset))
val_len   = int(0.2*len(dataset))
test_len  = len(dataset) - train_len - val_len
train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset,[train_len,val_len,test_len])
test_loader = DataLoader(test_ds,batch_size=1)

# -------------------------
# Load Backbone + Classifier
# -------------------------
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False)
state_dict = load_file("/home/teaching/ADL-project/notebooks/mitanshi/model.safetensors")
model.load_state_dict(state_dict, strict=False)
model.to(device)
for p in model.parameters(): p.requires_grad = False

clf_head = nn.Sequential(
    nn.Linear(model.num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
).to(device)

checkpoint = torch.load("/home/teaching/ADL-project/notebooks/mitanshi/busi_full_model.pth", map_location=device)
model.load_state_dict(checkpoint['backbone'])
clf_head.load_state_dict(checkpoint['classifier'])

# -------------------------
# Prompt-based Detection Setup
# -------------------------
# Fake “text embeddings” for prompts
feat_dim = model.num_features  # 768 for DINOv2

# Fake “text embeddings” expanded to match image feature size
prompt_embeddings = {
    "benign tumor": F.normalize(torch.randn(feat_dim), dim=0).to(device),
    "malignant tumor": F.normalize(torch.randn(feat_dim), dim=0).to(device)
}

def prompt_based_detect(img, prompt="malignant tumor"):
    model.eval(); clf_head.eval()
    with torch.no_grad():
        feat = model.forward_features(img.unsqueeze(0).to(device)).mean(dim=1)
        feat = F.normalize(feat, dim=-1)
        text_vec = prompt_embeddings[prompt].unsqueeze(0)
        sim = torch.matmul(feat, text_vec.T).item()
        pred_logits = clf_head(feat)
        probs = torch.softmax(pred_logits, dim=1).squeeze()
        pred_class = "malignant" if probs[1] > probs[0] else "benign"
    return pred_class, probs.cpu().numpy(), sim
# -------------------------
# Visualize & Save
# -------------------------
os.makedirs("busi_postproc_outputs", exist_ok=True)
plt.figure(figsize=(12,6))

for i in range(4):
    img_tensor, label, mask, name = random.choice(test_ds)
    pred, probs, sim = prompt_based_detect(img_tensor, prompt="malignant tumor")

    # Convert tensor to image
    img_np = (img_tensor.permute(1,2,0).cpu().numpy()*0.5+0.5)
    img_vis = (img_np * 255).astype(np.uint8)
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

    # Draw bbox if mask available
    if mask is not None:
        contours, _ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            color = (0,255,0) if pred=="benign" else (0,0,255)
            cv2.rectangle(img_vis, (x,y), (x+w,y+h), color, 2)
    
    # Save single output image
    cv2.putText(img_vis, f"{pred.upper()} ({probs.max():.2f})", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    out_path = f"busi_postproc_outputs/{name.replace('.png','_pred.png')}"
    cv2.imwrite(out_path, img_vis)
    print(f"✅ Saved: {out_path}")

print("\n🎯 All prompt-based detections with bounding boxes saved in 'busi_postproc_outputs/'")
