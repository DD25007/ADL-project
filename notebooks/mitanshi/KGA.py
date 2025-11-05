import os
import argparse
import random
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
from torchvision import models, transforms

class encoder(nn.Module):
    def __init__(self,out_dim=512):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,32,3,stride=2,padding=1),nn.ReLU(),nn.BatchNorm2d(32),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.ReLU(),nn.BatchNorm2d(64),
            nn.Conv2d(64,128,3,stride=2,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128,out_dim)
        )
    def forward(self,x):
        return self.net(x)
    
class temporal_attn_enc(nn.Module):
    def __init__(self, frame_emb_dim=512, proj_dim=128, nhead=8):
        super().__init__()
        self.attn=nn.MultiheadAttention(embed_dim=frame_emb_dim,num_heads=nhead,batch_first=True)
        self.proj=nn.Sequential(
            nn.Linear(frame_emb_dim,proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim,proj_dim)
        )
    def forward(self,frame_feats):
        attn_out,_=self.attn(frame_feats,frame_feats,frame_feats)
        vid_emb=attn_out.mean(dim=1)
        projected=self.proj(vid_emb)
        return F.normalize(projected,dim=-1)
    
class KGA(nn.Module):
    def __init__(self, frame_emb_dim=512, proj_dim=128, use_torchvision_backbone=True):
        super().__init__()
        self.frame_encoder=encoder(out_dim=frame_emb_dim)
        self.temporal=temporal_attn_enc(frame_emb_dim=frame_emb_dim,proj_dim=proj_dim)
    def forward(self,clip):
        B,T,C,H,W=clip.shape
        frames=clip.view(B*T,C,H,W)
        feats=self.frame_encoder(frames)
        vid_emb=self.temporal(feats)
        return vid_emb,feats
    
def triplet_loss(anchor, positive, negative, margin=1.0):
    d_pos = F.pairwise_distance(anchor, positive, p=2)
    d_neg = F.pairwise_distance(anchor, negative, p=2)
    loss = F.relu(d_pos - d_neg + margin)
    return loss.mean()

class MP4vid(Dataset):
    def __init__(self,root_dir,frames_per_clip=8,resize=(224,224),transform=None,random_sample=True):
        self.root=Path(root_dir)
        self.videos=sorted([p for p in self.root.rglob("*mp4")])
        self.frames_per_clip=frames_per_clip
        self.resize=resize
        self.transform=transform
        self.random_sample=random_sample
    def __len__(self):
        return len(self.videos)
    def _read_frames(self,path):
        cap=cv2.VideoCapture(str(path))
        frames=[]
        ret=True
        while ret:
            ret,frame=cap.read()
            if not ret:
                break
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if self.resize is not None:
                frame=cv2.resize(frame,self.resize,interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        return frames
    def _to_tensor(self,frame_list):
        import numpy as np
        arr=np.stack(frame_list,axis=0).astype('float32')/255.0
        return torch.from_numpy(arr.transpose(0,3,1,2))
    def __getitem__(self,idx):
        vpath=self.videos[idx]
        frames=self._read_frames(vpath)
        T=len(frames)
        k=self.frames_per_clip
        def sample_clip():
            if T<=k:
                sel=list(range(T))
            else:
                if self.random_sample:
                    start=random.randint(0,T-k)
                    sel=list(range(start,start+k))
                else:
                    indices=[int(i*T/k) for i in range(k)]
                    sel=indices
            return [frames[i] for i in sel]
        clip1=sample_clip()
        clip2=sample_clip()
        clip1_t=self._to_tensor(clip1)
        clip2_t=self._to_tensor(clip2)
        from PIL import Image
        clip1_t = torch.stack([self.transform(Image.fromarray((f*255).astype('uint8').transpose(1,2,0))) for f in clip1_t])
        clip2_t = torch.stack([self.transform(Image.fromarray((f*255).astype('uint8').transpose(1,2,0))) for f in clip2_t])
        return clip1_t,clip2_t,str(vpath)

def get_negative(batch_videos_ids):
    B=len(batch_videos_ids)
    neg_idx=[]
    for i in range(B):
        choices=list(range(B))
        choices.remove(i)
        if len(choices)==0:
            neg_idx.append(i)
        else:
            neg_idx.append(random.choice(choices))
    return neg_idx
def train(model,loader,optimizer,device,margin=0.8):
    model.train()
    total_loss=0.0
    for batch in loader:
        clip_a,clip_p,vid_ids=batch
        clip_a=clip_a.to(device).float()
        clip_p=clip_p.to(device).float()
        B=clip_a.shape[0]
        negIdx=get_negative(vid_ids)
        clip_n=clip_a[negIdx].to(device)
        emb_a,_=model(clip_a)
        emb_p,_=model(clip_p)
        emb_n,_=model(clip_n)
        loss=triplet_loss(emb_a,emb_p,emb_n,margin=margin)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()*B
    return total_loss/len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help=" .mp4 videos")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--frames_per_clip", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu_backbone", action="store_true", help="conv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resize", type=int, nargs=2, default=[224,224], help="resize H W")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    transform=None
    transform=transforms.Compose([transforms.Resize(tuple(args.resize)),transforms.ToTensor()])
    ds=MP4vid(args.data_root,frames_per_clip=args.frames_per_clip,resize=tuple(args.resize),transform=transform)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=2,pin_memory=True,collate_fn=None)
    model=KGA(frame_emb_dim=512,proj_dim=128,use_torchvision_backbone=(not args.cpu_backbone))
    device=torch.device(args.device)
    model.to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    for epochs in range(1,args.epochs+1):
        loss=train(model,loader,opt,device)
        print(args.epochs,loss)
        ckpt={"epochs":epochs,"model":model.state_dict(),"optimizer":opt.state_dict()}
        torch.save(ckpt,os.path.join(args.save_dir,f'ckpt{epochs}.pt'))
if __name__=="__main__":
    main()