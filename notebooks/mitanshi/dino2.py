# import cv2
# import matplotlib.pyplot as plt

# fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=True)

# video_path = "/home/teaching/ADL-project/video_benign_Pt-num-000_vid-num-000.mp4"
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     raise Exception("Could not open video file!")

# width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0

# out_path = "/home/teaching/ADL-project/output_motion_tracking.mp4"
# out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# print(f" Processing video... (saving to {out_path})")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     fgmask = fgbg.apply(frame)
#     fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
#     _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for c in contours:
#         if cv2.contourArea(c) > 400:
#             x, y, w, h = cv2.boundingRect(c)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
#             cv2.putText(frame, "tumour", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     out.write(frame)

# cap.release()
# out.release()

# print(f" Motion detection video saved at: {out_path}")

# plt.figure(figsize=(10, 6))
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title("Last Frame (with Detected Motion)")
# plt.show()

###########################

# import cv2
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from groundingdino.util.inference import Model
# import supervision as sv
# from PIL import Image

# model = Model(
#     model_config_path="/home/teaching/ADL-project/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
#     model_checkpoint_path="/home/teaching/ADL-project/GroundingDINO/weights/groundingdino_swint_ogc.pth",
#     device="cuda" if torch.cuda.is_available() else "cpu"
# )

# video_path = "/home/teaching/ADL-project/video_benign_Pt-num-000_vid-num-000.mp4"
# output_path = "/home/teaching/ADL-project/output_dino_motion.mp4"

# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     raise Exception("Could not open video file!")

# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

# out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=True)

# prompts = "malignant tissue, benign tissue, needle, hand, instrument"

# frame_count = 0
# print("Starting motion-triggered GroundingDINO detection...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_count += 1

#     fgmask = fgbg.apply(frame)
#     fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
#     _, thresh = cv2.threshold(fgmask, 110, 255, cv2.THRESH_BINARY)

#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     motion_detected = False

#     for c in contours:
#         if cv2.contourArea(c) > 800:  # Adjust for sensitivity
#             motion_detected = True
#             x, y, w, h = cv2.boundingRect(c)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
#             cv2.putText(frame, "lession", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

#     # Run DINO only when there's motion
#     if motion_detected and frame_count % 5 == 0:
#         boxes, phrases = model.predict_with_caption(
#             frame,
#             prompts,
#             box_threshold=0.35,
#             text_threshold=0.25
#         )

#         if boxes is not None and isinstance(boxes, np.ndarray) and boxes.shape[0] > 0:
#             detections = sv.Detections(xyxy=boxes)
#             box_annotator = sv.BoxAnnotator(thickness=1)
#             frame = box_annotator.annotate(scene=frame, detections=detections)

#             for box, label in zip(boxes, phrases):
#                 x1, y1, x2, y2 = map(int, box)
#                 cv2.putText(frame, label, (x1, max(20, y1 - 10)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         else:
#             pass

#     out.write(frame)

# cap.release()
# out.release()

############################
import cv2
import requests
import numpy as np
import random

def stream_video(url,max_frames=200):
    cap=cv2.VideoCapture(url)
    frames=[]
    count=0
    while cap.isOpened() and count<max_frames:
        ret,frame=cap.read()
        if not ret:
            break
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count+=1
    cap.release()
    return frames

# video_url = "http://172.18.43.247:5001/video/video_benign_Pt-num-000_vid-num-000.mp4"
# frames = stream_video(video_url)
# print("Loaded", len(frames), "frames from remote server.")

videos =requests.get("http://172.18.43.247:5001/list").json()["videos"]
print("Total videos: ", len(videos)) 

random.seed(42)
random.shuffle(videos)
#split
n=len(videos)
train_videos=videos[:int(0.7*n)]
val_videos=videos[int(0.7*n):int(0.85*n)]
test_videos=videos[int(0.85*n):]

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T
import timm
import torch.nn.functional as F

class remote_video_data(Dataset):
    def __init__(self,base_url,video_list,frames_per_video=100,transform=None):
        self.base_url = base_url
        self.video_list = video_list
        self.frames_per_video = frames_per_video
        self.transform = transform or T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(518,scale=(0.5,1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
        ])
    def __len__(self):
        return len(self.video_list)
    def __getitem__(self, idx):
        video_name=self.video_list[idx]
        url=f"{self.base_url}/video/{video_name}"
        cap=cv2.VideoCapture(url)
        frames=[]
        while cap.isOpened() and len(frames)<self.frames_per_video:
            ret,frame=cap.read()
            if not ret:
                break
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        frame=random.choice(frames)
        return self.transform(frame)
    
base_url = "http://172.18.43.247:5001"
train_dataset = remote_video_data(base_url, train_videos)
val_dataset = remote_video_data(base_url, val_videos)
test_dataset = remote_video_data(base_url, test_videos)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False)
state_dict=load_file("/home/teaching/ADL-project/notebooks/mitanshi/model.safetensors")
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion=nn.MSELoss()

transform = T.Compose([
    T.Resize((518, 518)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
])

from PIL import Image

def get_frame_emb(frames):
    with torch.no_grad():
        batch=torch.stack([transform(Image.fromarray(f)) for f in frames]).to(device)
        features=model.forward_features(batch)
        emb=F.normalize(features.mean(dim=1),dim=-1)
    return emb

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
optimizer = torch.optim.Adam(list(proj_head.parameters()) + list(predictor.parameters()), lr=1e-4)


for epoch in range(2):
    model.eval()  # DINO backbone frozen
    proj_head.train()
    predictor.train()
    
    for imgs in train_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            feats = model.forward_features(imgs)
            z = proj_head(feats.mean(dim=1).detach())
        
        p = predictor(z)
        loss = -criterion(z.detach(), p).mean()  # negative cosine similarity
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} | loss={loss.item():.4f}")

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_attention_map(model, frame):
    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
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

    if not attn_maps:
        raise RuntimeError(" No attention maps captured!")

    attn = attn_maps[0].mean(1)[0]
    num_patches = attn.shape[-1] - 1
    size = int(num_patches ** 0.5)
    attn_cls = attn[0, 1:].reshape(size, size)
    attn_cls = (attn_cls - attn_cls.min()) / (attn_cls.max() - attn_cls.min())
    attn_cls = cv2.resize(attn_cls.numpy(), frame.shape[:2][::-1])
    return attn_cls


sample_video = f"{base_url}/video/{test_videos[5]}"
frames = stream_video(sample_video, max_frames=100)

attn_map = get_attention_map(model, frames[50])
print("Attention map stats:")
print("min:", attn_map.min(), "max:", attn_map.max(), "mean:", attn_map.mean())

overlay = cv2.addWeighted(
    cv2.cvtColor(frames[50], cv2.COLOR_RGB2BGR),
    0.6,
    cv2.applyColorMap((attn_map * 255).astype(np.uint8), cv2.COLORMAP_JET),
    0.4,
    0
)

# Save to file
out_path = "attention_overlay.png"
cv2.imwrite(out_path, overlay)
print(f"Saved overlay at: {out_path}")
