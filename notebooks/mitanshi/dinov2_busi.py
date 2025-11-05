import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import timm
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, roc_auc_score

import os, torch, random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

class BUSIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        for label, cls in enumerate(["benign", "malignant"]):
            folder = os.path.join(root_dir, cls)
            for fname in os.listdir(folder):
                if fname.endswith(".png") and "mask" not in fname.lower():
                    self.samples.append((os.path.join(folder, fname), label))
        self.transform = transform or T.Compose([
            T.Resize((518,518)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), torch.tensor(label)

root_dir = "/home/teaching/ADL-project/data/Dataset_BUSI_with_GT"
dataset = BUSIDataset(root_dir)
train_len = int(0.7*len(dataset))
val_len   = int(0.2*len(dataset))
test_len  = len(dataset) - train_len - val_len
train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset,[train_len,val_len,test_len])
train_loader = DataLoader(train_ds,batch_size=8,shuffle=True,num_workers=4)
val_loader   = DataLoader(val_ds,batch_size=8)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False)
state_dict = load_file("/home/teaching/ADL-project/notebooks/mitanshi/model.safetensors")
model.load_state_dict(state_dict, strict=False)
model.to(device)

# freeze backbone, train small classifier
for p in model.parameters():
    p.requires_grad = False

clf_head = nn.Sequential(
    nn.Linear(model.num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
).to(device)

opt = torch.optim.Adam(clf_head.parameters(), lr=1e-4, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()

from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

train_losses, val_losses, val_accs, val_aucs = [], [], [], []

for epoch in range(20):
    clf_head.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            feats = model.forward_features(imgs).mean(dim=1)
        logits = clf_head(feats)
        loss = crit(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
        # if auc > best_auc:
        #     best_auc = auc
        #     torch.save(clf_head.state_dict(), "busi_best_head.pth")

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1} | TrainLoss={avg_train_loss:.3f}")

    # validation
    clf_head.eval()
    preds_all, labels_all = [], []
    val_loss = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = model.forward_features(imgs).mean(dim=1)
            logits = clf_head(feats)
            loss = crit(logits, labels)
            val_loss += loss.item()
            preds = torch.softmax(logits,dim=1)[:,1].cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    acc = accuracy_score(labels_all, np.array(preds_all)>0.5)
    auc = roc_auc_score(labels_all, preds_all)
    val_accs.append(acc)
    val_aucs.append(auc)
    print(f"ValLoss={avg_val_loss:.3f} | ValAcc={acc:.3f} | AUC={auc:.3f}")


epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.legend(); plt.title("Loss Curve")

plt.subplot(1,3,2)
plt.plot(epochs, val_accs, 'g', label="Val Accuracy")
plt.legend(); plt.title("Accuracy")

plt.subplot(1,3,3)
plt.plot(epochs, val_aucs, 'm', label="Val AUC")
plt.legend(); plt.title("AUC")

plt.tight_layout()
plt.savefig("busi_metrics.png")
print(" Saved plot -> busi_metrics.png")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
clf_head.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for imgs, labels in DataLoader(test_ds, batch_size=8):
        imgs = imgs.to(device)
        feats = model.forward_features(imgs).mean(dim=1)
        logits = clf_head(feats)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_labels.extend(labels.numpy())
cm = confusion_matrix(test_labels, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - BUSI Test Set")
plt.savefig("busi_confusion_matrix.png")
plt.close()
report = classification_report(test_labels, test_preds, target_names=["Benign", "Malignant"])
print("\nClassification Report:\n", report)

with open("busi_classification_report.txt", "w") as f:
    f.write(report)
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.legend(); plt.title("Loss Curve")

plt.subplot(1,3,2)
plt.plot(epochs, val_accs, 'g', label="Val Accuracy")
plt.legend(); plt.title("Accuracy")

plt.subplot(1,3,3)
plt.plot(epochs, val_aucs, 'm', label="Val AUC")
plt.legend(); plt.title("AUC")

plt.tight_layout()
plt.savefig("busi_metrics.png")
 
save_path = "busi_full_model.pth"
torch.save({
    'backbone': model.state_dict(),
    'classifier': clf_head.state_dict(),
    'optimizer': opt.state_dict()
}, save_path)
print(f" Full model saved at: {save_path}")
