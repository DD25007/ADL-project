import gc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve
from tqdm import tqdm
import logging
import subprocess

# set random seed for reproducibility
torch.manual_seed(42)

option = 2

if option == 1:
    training_file = "training_8.log"
elif option == 2:
    training_file = "training_16.log"
elif option == 3:
    training_file = "training_32.log"
else:
    training_file = "training_reduced.log"


# Custom handler class that flushes after every write
class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logger(log_file="training.log", clear_existing=False):
    """Setup logger with file and console handlers."""
    if clear_existing and os.path.exists(log_file):
        os.remove(log_file)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = FlushingFileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ============================================================================
# GPU SELECTION UTILITY
# ============================================================================


def cleanup_gpu_memory():
    """Aggressively clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_least_used_gpu():
    """Find the GPU with the lowest memory utilization percentage."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        gpu_info = []
        for line in result.stdout.strip().split("\n"):
            gpu_id, memory_used, memory_total = line.split(",")
            gpu_id = int(gpu_id.strip())
            memory_used = float(memory_used.strip())
            memory_total = float(memory_total.strip())
            memory_free = memory_total - memory_used
            gpu_info.append((gpu_id, memory_used, memory_total, memory_free))

        best_gpu = max(gpu_info, key=lambda x: x[3])
        return best_gpu[0]

    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logger.warning(f"Failed to detect GPU usage: {e}")
        logger.warning("Falling back to GPU 0")
        return 0


def select_best_device():
    """Select the best available device (GPU with lowest memory or CPU)."""
    if not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU")
        return torch.device("cpu")

    gpu_id = get_least_used_gpu()
    device = torch.device(f"cuda:{gpu_id}")

    try:
        torch.cuda.set_device(device)
        _ = torch.zeros(1).to(device)
        logger.info(f"✓ Successfully initialized {device}")
        return device
    except Exception as e:
        logger.warning(f"Failed to initialize cuda:{gpu_id}: {e}")
        logger.warning("Falling back to cuda:0")
        return torch.device("cuda:0")


# ============================================================================
# DATASET CLASSES AND DATALOADERS
# ============================================================================


class BUSIImageDataset(Dataset):
    """Dataset for BUSI ultrasound images with COCO annotations."""

    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir

        with open(annotation_file, "r") as f:
            self.coco_data = json.load(f)

        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.image_ids = list(self.images.keys())

        self.image_labels = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_labels:
                self.image_labels[img_id] = ann["category_id"] - 1

        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        image_path = os.path.join(self.root_dir, image_info["file_name"])

        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        label = self.image_labels.get(image_id, 0)

        if self.transform:
            image = self.transform(image)

        return image, label


class UltrasoundVideoDataset(Dataset):
    """Dataset class for loading ultrasound videos stored as individual frames."""

    def __init__(
        self,
        root_dir,
        annotation_file,
        num_frames=16,
        frame_format="{:05d}.jpg",
        transform=None,
        use_train_frames=True,
    ):
        self.root_dir = root_dir
        self.rawframes_dir = os.path.join(root_dir, "rawframes")
        self.num_frames = num_frames
        self.frame_format = frame_format
        self.use_train_frames = use_train_frames

        annotation_path = os.path.join(root_dir, annotation_file)
        with open(annotation_path, "r") as f:
            data = json.load(f)

        self.categories = {cat["id"]: cat["name"] for cat in data["categories"]}
        self.videos = data["videos"]
        self.video_id_to_idx = {
            video["id"]: idx for idx, video in enumerate(self.videos)
        }

        self.video_labels = {}
        for ann in data["annotations"]:
            video_id = ann["video_id"]
            category_id = ann["category_id"]
            if video_id not in self.video_labels:
                self.video_labels[video_id] = category_id - 1

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.videos)

    def _get_all_frames_in_dir(self, video_path):
        """Get all frame files in the video directory."""
        frame_files = []
        if os.path.exists(video_path):
            files = sorted(os.listdir(video_path))
            for f in files:
                if f.endswith((".jpg", ".jpeg", ".png")):
                    frame_files.append(f)
        return frame_files

    def _load_frame(self, video_path, frame_idx):
        """Load a single frame from disk."""
        frame_name = self.frame_format.format(frame_idx)
        frame_path = os.path.join(video_path, frame_name)

        try:
            frame = Image.open(frame_path).convert("RGB")
        except FileNotFoundError:
            alternatives = [
                f"{frame_idx:05d}.jpg",
                f"{frame_idx:04d}.jpg",
                f"frame_{frame_idx:05d}.jpg",
                f"img_{frame_idx:05d}.jpg",
                f"{frame_idx:05d}.png",
            ]

            for alt_name in alternatives:
                alt_path = os.path.join(video_path, alt_name)
                if os.path.exists(alt_path):
                    frame = Image.open(alt_path).convert("RGB")
                    break
            else:
                all_frames = self._get_all_frames_in_dir(video_path)
                if len(all_frames) > 0:
                    fallback_path = os.path.join(video_path, all_frames[0])
                    frame = Image.open(fallback_path).convert("RGB")
                else:
                    raise FileNotFoundError(
                        f"Could not find any frames in {video_path}"
                    )

        return frame

    def _sample_frames(self, available_frames, num_frames):
        """Sample frame indices from available frames."""
        valid_frames = [f for f in available_frames if f > 0]

        if len(valid_frames) == 0:
            return list(range(1, num_frames + 1))

        if len(valid_frames) <= num_frames:
            sampled = valid_frames.copy()
            sampled += [valid_frames[-1]] * (num_frames - len(valid_frames))
        else:
            step = len(valid_frames) / num_frames
            sampled = [valid_frames[int(i * step)] for i in range(num_frames)]

        return sampled

    def __getitem__(self, idx):
        """Load a video and its annotations."""
        video = self.videos[idx]
        video_id = video["id"]
        video_name = video["name"]
        label = self.video_labels.get(video_id, 0)
        video_path = os.path.join(self.rawframes_dir, video_name)

        if self.use_train_frames and "vid_train_frames" in video:
            available_frames = video["vid_train_frames"]
            frame_indices = self._sample_frames(available_frames, self.num_frames)
        else:
            frame_indices = list(range(1, self.num_frames + 1))

        frames = []
        successfully_loaded = []

        for i, frame_idx in enumerate(frame_indices):
            try:
                frame = self._load_frame(video_path, frame_idx)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
                successfully_loaded.append(i)
            except Exception as e:
                logger.warning(
                    f"Error loading frame {frame_idx} from {video_name}: {e}"
                )
                frames.append(None)

        for i, frame in enumerate(frames):
            if frame is None:
                if successfully_loaded:
                    nearest_idx = min(successfully_loaded, key=lambda x: abs(x - i))
                    frames[i] = frames[nearest_idx].clone()
                else:
                    if self.transform:
                        frames[i] = self.transform(
                            Image.new("RGB", (224, 224), (0, 0, 0))
                        )
                    else:
                        frames[i] = torch.zeros(3, 224, 224)

        frames = torch.stack(frames)
        return frames, label


def video_dataloader(
    root_dir,
    train_annotation="imagenet_vid_train_15frames.json",
    val_annotation="imagenet_vid_val.json",
    batch_size=4,
    num_frames=15,
    num_workers=4,
):
    """Create training and validation dataloaders."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = UltrasoundVideoDataset(
        root_dir=root_dir,
        annotation_file=train_annotation,
        num_frames=num_frames,
        transform=train_transform,
    )

    val_dataset = UltrasoundVideoDataset(
        root_dir=root_dir,
        annotation_file=val_annotation,
        num_frames=128,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,
    )

    return train_loader, val_loader


# ============================================================================
# KGA-NET COMPONENTS
# ============================================================================


class FrameAttention(nn.Module):
    """Frame attention module that produces scalar weight per frame."""

    def __init__(self, feature_dim):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, frame_features):
        """
        Args:
            frame_features: (B, N, C, H, W)
        Returns:
            weights: (B, N, 1, 1, 1)
        """
        B, N, C, H, W = frame_features.shape
        frame_vectors = frame_features.view(B, N, C)
        weights = torch.sigmoid(self.fc(frame_vectors))
        return weights.view(B, N, 1, 1, 1)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================


class CenterLoss(nn.Module):
    """Center Loss for discriminative feature learning."""

    def __init__(self, num_classes, feature_dim=512, device="cuda"):
        super().__init__()
        self.centers = nn.Parameter(
            torch.randn(num_classes, feature_dim, device=device) * 0.01
        )
        self.feature_dim = feature_dim

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers[labels]
        loss = torch.sum((features - centers_batch) ** 2)
        loss = loss / (2.0 * batch_size * self.feature_dim)
        return loss


class CoherenceLoss(nn.Module):
    """Coherence Loss from paper Eq. 3"""

    def __init__(self, in_features=2048, feature_dim=512, device="cuda"):
        super().__init__()

    def forward(
        self,
        frame_features: torch.tensor,
        attention_weights: torch.tensor,
        class_centers: torch.tensor,
        labels: torch.tensor,
        eps: float = 1e-8,
    ):
        """
        Args:
            frame_features: (B, N, C, H, W)
            attention_weights: (B, N, 1, 1, 1)
            class_centers: (num_classes, C)
            labels: (B,)
        """
        B, N, C, H, W = frame_features.shape
        frame_vectors = frame_features.view(B, N, C)

        centers = class_centers[labels].unsqueeze(1)
        distances = torch.norm(frame_vectors - centers, dim=2)

        d_norm = distances / (distances.norm(dim=1, keepdim=True).clamp(min=eps))

        weights = attention_weights.view(B, N)
        w_inv = 1.0 - weights
        w_norm = w_inv / (w_inv.norm(dim=1, keepdim=True).clamp(min=eps))

        gram_d = d_norm.unsqueeze(2).bmm(d_norm.unsqueeze(1))
        gram_w = w_norm.unsqueeze(2).bmm(w_norm.unsqueeze(1))

        diff = gram_w - gram_d
        loss_per_sample = (diff * diff).sum(dim=(1, 2))
        loss = 0.5 * loss_per_sample.mean()

        return loss


class StandardTripletLoss(nn.Module):
    """Standard Triplet Loss with mining strategies."""

    def __init__(self, margin=1.0, mining="hard"):
        super(StandardTripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining

        if mining == "all":
            import warnings

            warnings.warn(
                "Mining='all' can be very slow with many samples. "
                "Consider using 'hard' or 'semi-hard' mining instead."
            )

    def forward(self, frame_features: torch.Tensor, labels: torch.Tensor):
        if len(frame_features.shape) == 5:
            batch_size, num_frames, channels, height, width = frame_features.shape
            frame_features = F.adaptive_avg_pool2d(
                frame_features.view(batch_size * num_frames, channels, height, width), 1
            ).view(batch_size * num_frames, channels)
            labels = labels.reshape(-1)

        num_samples, channels = frame_features.shape
        dist_matrix = torch.cdist(frame_features, frame_features, p=2)

        labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
        labels_not_equal = ~labels_equal

        mask_diag = torch.eye(
            num_samples, dtype=torch.bool, device=frame_features.device
        )
        labels_equal = labels_equal & ~mask_diag

        total_loss = 0.0
        count = 0

        for anchor_idx in range(num_samples):
            pos_mask = labels_equal[anchor_idx]
            neg_mask = labels_not_equal[anchor_idx]

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            pos_dists = dist_matrix[anchor_idx][pos_mask]
            neg_dists = dist_matrix[anchor_idx][neg_mask]

            if self.mining == "hard":
                hardest_pos = pos_dists.max()
                hardest_neg = neg_dists.min()
                loss = torch.clamp(hardest_pos - hardest_neg + self.margin, min=0.0)
                total_loss += loss
                count += 1

            elif self.mining == "semi-hard":
                for pos_d in pos_dists:
                    semi_hard_negs = neg_dists[
                        (neg_dists > pos_d) & (neg_dists < pos_d + self.margin)
                    ]
                    if len(semi_hard_negs) > 0:
                        for neg_d in semi_hard_negs:
                            loss = torch.clamp(pos_d - neg_d + self.margin, min=0.0)
                            total_loss += loss
                            count += 1

            elif self.mining == "all":
                max_triplets_per_anchor = 100
                triplet_count = 0
                for pos_d in pos_dists:
                    for neg_d in neg_dists:
                        if triplet_count >= max_triplets_per_anchor:
                            break
                        loss = torch.clamp(pos_d - neg_d + self.margin, min=0.0)
                        total_loss += loss
                        count += 1
                        triplet_count += 1
                    if triplet_count >= max_triplets_per_anchor:
                        break

        if count > 0:
            total_loss = total_loss / count
        else:
            total_loss = frame_features.sum() * 0.0

        return total_loss


# ============================================================================
# KGA-NET MODEL
# ============================================================================


class KGANet(nn.Module):
    """KGA-Net: Keyframe Guidance Attention Network."""

    def __init__(self, num_classes=2, feature_dim=2048, backbone=None):
        super().__init__()

        if backbone is not None:
            self.backbone = backbone
        else:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.frame_attention = FrameAttention(feature_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if feature_dim != 2048:
            self.projection = nn.Sequential(
                nn.Linear(2048, feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU()
            )
        else:
            self.projection = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(feature_dim, num_classes)
        )

    def forward(self, video_frames, return_features=False) -> torch.Tensor:
        """
        Args:
            video_frames: (B, N, 3, H, W)
            return_features: whether to return intermediate features
        """
        batch_size, num_frames, rgb_channels, height, width = video_frames.shape

        frames_flat = video_frames.view(
            batch_size * num_frames, rgb_channels, height, width
        )
        features_flat = self.backbone(frames_flat)

        pooled_flat = self.global_pool(features_flat).view(batch_size * num_frames, -1)
        projected_flat = self.projection(pooled_flat)

        frame_features = projected_flat.view(batch_size, num_frames, -1)

        frame_features_5d = frame_features.unsqueeze(-1).unsqueeze(-1)
        attention_weights = self.frame_attention(frame_features_5d)

        weighted_features = frame_features_5d * attention_weights
        aggregated = weighted_features.sum(dim=1).squeeze(-1).squeeze(-1)

        logits = self.classifier(aggregated)

        if return_features:
            return logits, {
                "frame_features": frame_features_5d,
                "attention_weights": attention_weights,
                "aggregated_features": aggregated,
            }
        return logits


class ImageClassificationNetwork(nn.Module):
    """Image classification network that shares the same 2D backbone as KGANet."""

    def __init__(self, num_classes=2, feature_dim=512, backbone=None):
        super(ImageClassificationNetwork, self).__init__()

        if backbone is not None:
            self.backbone = backbone
        else:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if feature_dim != 2048:
            self.projection = nn.Sequential(
                nn.Linear(2048, feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU()
            )
        else:
            self.projection = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(feature_dim, num_classes)
        )

    def forward(self, images: torch.Tensor, return_features: bool = False):
        feats = self.backbone(images)
        pooled = self.global_pool(feats).view(feats.size(0), -1)
        projected = self.projection(pooled)
        logits = self.classifier(projected)

        if return_features:
            return logits, projected
        return logits


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def calculate_youden_threshold(model, val_loader, device):
    """Calculate optimal classification threshold using Youden's index."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x in val_loader:
            inputs, labels = x
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)

    fpr, tpr, thresholds = roc_curve(labels, probs)
    idx = np.argmax(tpr - fpr)

    return thresholds[idx]


def train_joint_epoch(
    image_model,
    video_model,
    image_loader,
    video_loader,
    main_optimizer,
    center_optimizer,
    device,
    center_loss_fn,
    main_scheduler,
    alpha_center=0.5,
    video_loss_type="coherence",
    mining="hard",
    lambda_coh=1.0,
    start_iter=0,
    max_iters=8000,
    feature_dim=2048,
):
    """Train both image and video models jointly for one epoch."""
    image_model.train()
    video_model.train()

    total_image_loss = 0.0
    total_video_loss = 0.0
    image_correct = 0
    video_correct = 0
    image_total = 0
    video_total = 0

    cls_criterion = nn.CrossEntropyLoss()

    if video_loss_type == "coherence":
        video_aux_criterion = CoherenceLoss(feature_dim=feature_dim, device=device)
    elif video_loss_type == "triplet_standard":
        video_aux_criterion = StandardTripletLoss(margin=1.0, mining=mining)
    else:
        raise ValueError(f"Unknown video_loss_type: {video_loss_type}")

    image_iter = iter(image_loader)
    video_iter = iter(video_loader)

    max_iters_this_epoch = min(len(image_loader), len(video_loader))
    pbar = tqdm(range(max_iters_this_epoch), desc=f"Training (Iter {start_iter})")

    iters_done = 0

    for _ in pbar:
        if start_iter + iters_done >= max_iters:
            break

        # ========== IMAGE BATCH ==========
        try:
            images, img_labels = next(image_iter)
        except StopIteration:
            image_iter = iter(image_loader)
            images, img_labels = next(image_iter)

        images = images.to(device, non_blocking=True)
        img_labels = img_labels.to(device, non_blocking=True)

        main_optimizer.zero_grad()
        center_optimizer.zero_grad()

        img_logits, img_features = image_model(images, return_features=True)
        img_cls_loss = cls_criterion(img_logits, img_labels)
        img_center_loss = center_loss_fn(img_features, img_labels)
        img_loss = img_cls_loss + alpha_center * img_center_loss

        img_loss.backward()
        main_optimizer.step()
        center_optimizer.step()

        total_image_loss += img_loss.item()
        _, img_pred = img_logits.max(1)
        image_total += img_labels.size(0)
        image_correct += img_pred.eq(img_labels).sum().item()

        # ========== VIDEO BATCH ==========
        try:
            frames, vid_labels = next(video_iter)
        except StopIteration:
            video_iter = iter(video_loader)
            frames, vid_labels = next(video_iter)

        frames = frames.to(device, non_blocking=True)
        vid_labels = vid_labels.to(device, non_blocking=True)

        main_optimizer.zero_grad()

        vid_logits, vid_features = video_model(frames, return_features=True)

        # Video-level classification loss
        vid_cls_loss = cls_criterion(vid_logits, vid_labels)

        # Frame-level classification loss
        batch_size, num_frames = frames.shape[0], frames.shape[1]
        frame_features = vid_features["frame_features"]
        frame_vectors = frame_features.view(batch_size * num_frames, -1)
        frame_logits = video_model.classifier(frame_vectors)

        frame_labels = vid_labels.unsqueeze(1).expand(-1, num_frames).reshape(-1)
        frame_cls_loss = cls_criterion(frame_logits, frame_labels)

        # Auxiliary loss
        if video_loss_type == "coherence":
            vid_aux_loss = video_aux_criterion(
                vid_features["frame_features"],
                vid_features["attention_weights"],
                center_loss_fn.centers,
                vid_labels,
            )
        elif video_loss_type == "triplet_standard":
            frame_labels_expanded = vid_labels.unsqueeze(1).expand(-1, num_frames)
            vid_aux_loss = video_aux_criterion(
                vid_features["frame_features"], frame_labels_expanded
            )

        vid_loss = vid_cls_loss + frame_cls_loss + lambda_coh * vid_aux_loss

        if (iters_done + 1) % 50 == 0:
            with torch.no_grad():
                logger.info(
                    f"   Image CE: {img_cls_loss:.4f}, Center: {img_center_loss:.4f}, Current LR: {main_scheduler.get_last_lr()[0]:.6f}"
                )
                center_norm = center_loss_fn.centers.norm(dim=1)
                center_dist = torch.dist(
                    center_loss_fn.centers[0], center_loss_fn.centers[1]
                )
                logger.info(
                    f"   Center norms: [{center_norm[0]:.2f}, {center_norm[1]:.2f}], Distance: {center_dist:.2f}"
                )

        vid_loss.backward()
        main_optimizer.step()

        total_video_loss += vid_loss.item()
        _, vid_pred = vid_logits.max(1)
        video_total += vid_labels.size(0)
        video_correct += vid_pred.eq(vid_labels).sum().item()

        iters_done += 1
        main_scheduler.step()

        pbar.set_postfix(
            {
                "iter": f"{start_iter + iters_done}/{max_iters}",
                "img_acc": f"{100.*image_correct/image_total:.1f}%",
                "vid_acc": f"{100.*video_correct/video_total:.1f}%",
            }
        )

        # Clean up tensors
        del frames, vid_labels, vid_logits, vid_features, vid_loss
        del vid_cls_loss, frame_cls_loss, img_loss
        del img_logits, img_features, images, img_labels
        del frame_features, frame_logits, frame_labels

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return (
        total_image_loss / iters_done,
        total_video_loss / iters_done,
        100.0 * image_correct / image_total,
        100.0 * video_correct / video_total,
        iters_done,
    )


def train_joint_kga_net(
    image_root_dir,
    image_annotation_file,
    video_root_dir,
    video_train_annotation,
    video_val_annotation,
    total_iter=8000,
    image_batch_size=8,
    video_batch_size=8,
    num_frames=16,
    learning_rate=0.005,
    alpha_center=0.5,
    lambda_coh=1.0,
    video_loss_type="coherence",
    mining="hard",
    save_dir="checkpoints/joint_model",
    num_workers=4,
    early_stopping=False,
    patience=15,
    device="cpu",
    feature_dim=2048,
):
    """Train image and video models jointly."""
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Using device: {device}")

    # ==================== CREATE IMAGE DATALOADERS ====================
    logger.info("\nLoading image dataset...")

    train_img_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_img_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_image_dataset = BUSIImageDataset(
        image_root_dir, image_annotation_file, train_img_transform
    )

    val_image_dataset = BUSIImageDataset(
        image_root_dir, image_annotation_file, val_img_transform
    )

    image_train_loader = DataLoader(
        train_image_dataset,
        batch_size=image_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    image_val_loader = DataLoader(
        val_image_dataset,
        batch_size=image_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Image training samples: {len(train_image_dataset)}")
    logger.info(f"Image validation samples: {len(val_image_dataset)}")

    # ==================== CREATE VIDEO DATALOADERS ====================
    logger.info("\nLoading video dataset...")
    video_train_loader, video_val_loader = video_dataloader(
        root_dir=video_root_dir,
        train_annotation=video_train_annotation,
        val_annotation=video_val_annotation,
        batch_size=video_batch_size,
        num_frames=num_frames,
        num_workers=num_workers,
    )
    logger.info(f"Video training samples: {len(video_train_loader.dataset)}")
    logger.info(f"Video validation samples: {len(video_val_loader.dataset)}")

    # ==================== CREATE MODELS ====================
    logger.info("\nInitializing models...")
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    shared_backbone = nn.Sequential(*list(resnet.children())[:-2]).to(device)

    image_model = ImageClassificationNetwork(
        num_classes=2,
        feature_dim=feature_dim,
        backbone=shared_backbone,
    ).to(device)

    video_model = KGANet(
        num_classes=2,
        feature_dim=feature_dim,
        backbone=shared_backbone,
    ).to(device)

    center_loss_fn = CenterLoss(num_classes=2, feature_dim=feature_dim, device=device)

    # ==================== OPTIMIZERS ====================
    def _unique_params(param_iterables):
        """Deduplicate parameters"""
        seen = set()
        unique = []
        for p in param_iterables:
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            unique.append(p)
        return unique

    params_to_optimize = []
    params_to_optimize += list(shared_backbone.parameters())
    params_to_optimize += list(image_model.projection.parameters())
    params_to_optimize += list(video_model.projection.parameters())
    params_to_optimize += list(image_model.classifier.parameters())
    params_to_optimize += list(video_model.frame_attention.parameters())
    params_to_optimize += list(video_model.classifier.parameters())

    params_to_optimize = _unique_params(params_to_optimize)

    main_optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
    )

    center_optimizer = torch.optim.SGD(
        center_loss_fn.parameters(),
        lr=0.5,
        momentum=0.9,
    )

    # ==================== SCHEDULERS ====================
    iterations_per_epoch = min(len(image_train_loader), len(video_train_loader))
    num_epochs = math.ceil(total_iter / iterations_per_epoch)

    logger.info(f"Iterations per epoch: {iterations_per_epoch}")
    logger.info(f"Training for {num_epochs} epochs to reach {total_iter} iterations")

    def lr_lambda(current_iter):
        if current_iter < 1000:
            return float(current_iter) / 1000.0
        elif current_iter < 4000:
            return 1.0
        elif current_iter < 6000:
            return 0.1
        else:
            return 0.01

    main_scheduler = torch.optim.lr_scheduler.LambdaLR(main_optimizer, lr_lambda)

    # ==================== TRAINING LOOP ====================
    logger.info(f"\nStarting joint training...")

    best_img_acc = 0.0
    best_vid_acc = 0.0
    global_iter = 0

    if early_stopping:
        epochs_without_improvement = 0

    for epoch in range(num_epochs):
        if global_iter >= total_iter:
            break

        logger.info(f"\nEpoch {epoch+1}/{num_epochs} (Iter {global_iter}/{total_iter})")

        img_loss, vid_loss, img_acc, vid_acc, iters_done = train_joint_epoch(
            image_loader=image_train_loader,
            video_loader=video_train_loader,
            image_model=image_model,
            video_model=video_model,
            main_optimizer=main_optimizer,
            center_optimizer=center_optimizer,
            device=device,
            center_loss_fn=center_loss_fn,
            main_scheduler=main_scheduler,
            alpha_center=alpha_center,
            lambda_coh=lambda_coh,
            video_loss_type=video_loss_type,
            mining=mining,
            start_iter=global_iter,
            max_iters=total_iter,
            feature_dim=feature_dim,
        )

        global_iter += iters_done

        logger.info(f"Train - Image Loss: {img_loss:.4f}, Acc: {img_acc:.2f}%")
        logger.info(f"Train - Video Loss: {vid_loss:.4f}, Acc: {vid_acc:.2f}%")

        # ==================== VALIDATION ====================
        image_model.eval()
        img_correct, img_total = 0, 0
        with torch.no_grad():
            for images, labels in image_val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = image_model(images)
                _, pred = logits.max(1)
                img_total += labels.size(0)
                img_correct += pred.eq(labels).sum().item()
        val_img_acc = 100.0 * img_correct / img_total

        val_vid_loss, val_vid_acc = validate(video_model, video_val_loader, device)

        logger.info(
            f"Val   - Image Acc: {val_img_acc:.2f}%, Video Acc: {val_vid_acc:.2f}%"
        )

        image_threshold = calculate_youden_threshold(
            image_model, image_val_loader, device
        )
        video_threshold = calculate_youden_threshold(
            video_model, video_val_loader, device
        )

        logger.info(
            f"Optimal thresholds - Image: {image_threshold:.3f}, Video: {video_threshold:.3f}"
        )

        if early_stopping:
            if val_vid_acc > best_vid_acc:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info(
                    f"\nEarly stopping triggered after {patience} epochs without improvement"
                )
                logger.info(f"Best video accuracy: {best_vid_acc:.2f}%")
                break

        if video_loss_type == "triplet_standard":
            vid_loss_type_str = f"{video_loss_type}_{mining}"
        else:
            vid_loss_type_str = video_loss_type

        if val_img_acc > best_img_acc:
            best_img_acc = val_img_acc
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state": shared_backbone.state_dict(),
                    "image_head": image_model.state_dict(),
                    "val_acc": val_img_acc,
                    "threshold": image_threshold,
                },
                os.path.join(save_dir, "best_image_model.pth"),
            )
            logger.info(f"✓ Saved best image model (Acc: {val_img_acc:.2f}%)")

        if val_vid_acc > best_vid_acc:
            best_vid_acc = val_vid_acc
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state": shared_backbone.state_dict(),
                    "video_head": video_model.state_dict(),
                    "val_acc": val_vid_acc,
                    "threshold": video_threshold,
                },
                os.path.join(
                    save_dir, f"best_video_model_{vid_loss_type_str}_{num_frames}.pth"
                ),
            )
            logger.info(f"✓ Saved best video model (Acc: {val_vid_acc:.2f}%)")

    logger.info("\n" + "=" * 70)
    logger.info(f"Training completed!")
    logger.info(f"Best Image Acc: {best_img_acc:.2f}%")
    logger.info(f"Best Video Acc: {best_vid_acc:.2f}%")
    logger.info("=" * 70)

    cleanup_gpu_memory()
    return image_model, video_model


def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for frames, labels in pbar:
            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"}
            )

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    # Initialize logger ONCE at the start of training
    logger = setup_logger(training_file, clear_existing=True)

    # Dataset configuration
    IMAGE_ROOT_DIR = "./data/busi/"
    IMAGE_ANNOTATION = "data/busi_bboxes.json"
    VIDEO_ROOT_DIR = "./data"
    VIDEO_TRAIN_ANNOTATION = "imagenet_vid_train_15frames.json"
    VIDEO_VAL_ANNOTATION = "imagenet_vid_val.json"

    # Training configuration (Paper's values)
    IMAGE_BATCH_SIZE = 8  # Total batch=16, split 1:1 between image and video
    VIDEO_BATCH_SIZE = 8
    # NUM_FRAMES = 16  # Adjust based on GPU memory
    LEARNING_RATE = 0.005  # Paper's initial LR
    FEATURE_DIM = 512  # 2048 for no projection or 512/1024 for dimensionality reduction

    # Loss configuration
    LOSS_TYPE = "coherence"  # Paper uses coherence loss
    MINING = None  # Not used with coherence loss
    # ALPHA_CENTER = 0.5  # Weight for center loss

    # Path to save Checkpoints
    SAVE_DIR = "notebooks/joint_checkpoints"

    device = select_best_device()
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("KGA-Net Training Pipeline")
    print("=" * 70)

    # for LOSS_TYPE, MINING in [
    #     ("coherence", None),
    #     ("triplet_standard", "hard"),
    #     ("triplet_standard", "semi-hard"),
    #     ("triplet_standard", "all"),
    # ]:
    #     for ALPHA_CENTER, NUM_FRAMES in [
    #         (0.4, 64),
    #         (1, 64),
    #     ]:
    for ALPHA_CENTER in [0.2, 0.4, 0.6, 0.8, 1]:
        if option == 1:
            l = [8]
        elif option == 2:
            l = [16]
        else:
            l = [32]
        for NUM_FRAMES in l:

            logger.info("\n" + "=" * 70)
            logger.info("Training Configuration:")
            logger.info(f"Video Batch Size: {VIDEO_BATCH_SIZE}")
            logger.info(f"Image Batch Size: {IMAGE_BATCH_SIZE}")
            logger.info(f"Number of Frames: {NUM_FRAMES}")
            logger.info(f"Learning Rate: {LEARNING_RATE}")
            logger.info(f"Alpha Center: {ALPHA_CENTER}")
            if LOSS_TYPE == "triplet_standard":
                logger.info(f"Video Loss: {LOSS_TYPE} with {MINING} mining")
            else:
                logger.info(f"Loss Type: {LOSS_TYPE}")
            logger.info("=" * 70)

            try:
                trained_model = train_joint_kga_net(
                    image_root_dir=IMAGE_ROOT_DIR,
                    image_annotation_file=IMAGE_ANNOTATION,
                    video_root_dir=VIDEO_ROOT_DIR,
                    video_train_annotation=VIDEO_TRAIN_ANNOTATION,
                    video_val_annotation=VIDEO_VAL_ANNOTATION,
                    video_loss_type=LOSS_TYPE,
                    mining=MINING,
                    alpha_center=ALPHA_CENTER,
                    learning_rate=LEARNING_RATE,
                    image_batch_size=IMAGE_BATCH_SIZE,
                    video_batch_size=VIDEO_BATCH_SIZE,
                    num_frames=NUM_FRAMES,
                    save_dir=SAVE_DIR,
                    early_stopping=True,
                    patience=30,
                    device=device,
                    feature_dim=FEATURE_DIM,
                )
                del trained_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                logger.error(
                    f"Training failed for {NUM_FRAMES} frames with error: {e}",
                    exc_info=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                continue

    print("\n✓ Training completed successfully!")
    print(f"✓ Model checkpoints saved in: {SAVE_DIR}/")
