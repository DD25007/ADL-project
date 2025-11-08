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

# set random seed for reproducibility
torch.manual_seed(42)

training_file = "training.log"

if os.path.exists(training_file):
    os.remove(training_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(training_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class BUSIImageDataset(Dataset):
    """Dataset for BUSI ultrasound images with COCO annotations."""

    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir

        with open(annotation_file, "r") as f:
            self.coco_data = json.load(f)

        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.image_ids = list(self.images.keys())

        # Get labels from annotations
        self.image_labels = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_labels:
                self.image_labels[img_id] = ann["category_id"] - 1  # 0-indexed

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
        """Initialize the dataset."""
        self.root_dir = root_dir
        self.rawframes_dir = os.path.join(root_dir, "rawframes")
        self.num_frames = num_frames
        self.frame_format = frame_format
        self.use_train_frames = use_train_frames

        # Load annotations
        annotation_path = os.path.join(root_dir, annotation_file)
        with open(annotation_path, "r") as f:
            data = json.load(f)

        # Parse categories
        self.categories = {cat["id"]: cat["name"] for cat in data["categories"]}

        # Parse videos
        self.videos = data["videos"]
        self.video_id_to_idx = {
            video["id"]: idx for idx, video in enumerate(self.videos)
        }

        # Parse annotations to get video labels
        self.video_labels = {}
        for ann in data["annotations"]:
            video_id = ann["video_id"]
            category_id = ann["category_id"]
            if video_id not in self.video_labels:
                # Convert category_id (1, 2) to label (0, 1)
                self.video_labels[video_id] = category_id - 1

        # Default transform
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
            # Try alternative naming conventions
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
                # Fallback: use any available frame
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
        # Filter out -1 (invalid frames)
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

        # Get frame indices to load
        if self.use_train_frames and "vid_train_frames" in video:
            available_frames = video["vid_train_frames"]
            frame_indices = self._sample_frames(available_frames, self.num_frames)
        else:
            frame_indices = list(range(1, self.num_frames + 1))

        # Load frames
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
                # Will be filled later with nearest valid frame
                frames.append(None)

        # Fill missing frames with nearest valid frame
        for i, frame in enumerate(frames):
            if frame is None:
                if successfully_loaded:
                    # Find nearest successfully loaded frame
                    nearest_idx = min(successfully_loaded, key=lambda x: abs(x - i))
                    frames[i] = frames[nearest_idx].clone()
                else:
                    # No frames loaded - use black frame
                    if self.transform:
                        frames[i] = self.transform(
                            Image.new("RGB", (224, 224), (0, 0, 0))
                        )
                    else:
                        frames[i] = torch.zeros(3, 224, 224)

        frames = torch.stack(frames)

        # FIX: Use actual positions in the loaded sequence
        # These correspond to positions in the frames tensor (0-indexed)
        num_loaded = len(frame_indices)
        keyframe_indices = torch.tensor(
            [0, num_loaded // 2, num_loaded - 1], dtype=torch.long
        )

        return frames, label, keyframe_indices


def video_dataloader(
    root_dir,
    train_annotation="imagenet_vid_train_15frames.json",
    val_annotation="imagenet_vid_val.json",
    batch_size=4,
    num_frames=15,
    num_workers=4,
):
    """Create training and validation dataloaders."""

    # Data augmentation for training
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

    # No augmentation for validation
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
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

    # Create dataloaders
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
    """Frame attention module that produces scalar weight per frame.
    Paper Eq. 1: w_i = Sigmoid(FC(F_i))
    """

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
        # Global average pooling per frame
        frame_vectors = F.adaptive_avg_pool2d(
            frame_features.view(B * N, C, H, W), 1
        ).view(B, N, C)

        # Compute scalar attention weight per frame
        weights = torch.sigmoid(self.fc(frame_vectors))  # (B, N, 1)
        return weights.view(B, N, 1, 1, 1)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================


class CenterLoss(nn.Module):
    """Center Loss for discriminative feature learning.
    Args:
        num_classes: Number of classes
        feature_dim: Dimension of feature vectors
        device: Device to store the centers
    """

    def __init__(self, num_classes, feature_dim, device="cuda"):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim).to(device))
        # self.centers = nn.Parameter(torch.zeros(num_classes, feature_dim).to(device))

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers[labels]
        loss = torch.sum((features - centers_batch) ** 2) / (2.0 * batch_size)
        return loss


class CoherenceLoss(nn.Module):
    """Coherence Loss from paper Eq. 3
    L_Coh = ||Gram_w - Gram_d||_2

    Guides attention weights to align with feature distances to class centers
    """

    def __init__(self, in_features=2048, out_features=512, device="cuda"):
        super().__init__()
        # self.feature_projection = nn.Linear(in_features, out_features).to(device)

    def forward(self, frame_features, attention_weights, class_centers, labels):
        """
        Args:
            frame_features: (B, N, C, H, W) - frame feature maps (C=2048)
            attention_weights: (B, N, 1, 1, 1) - attention weights
            class_centers: (num_classes, 2048) - class centers
            labels: (B,) - video labels
        """
        B, N, C, H, W = frame_features.shape
        device = frame_features.device

        # Convert feature maps to vectors via global pooling
        frame_vectors = F.adaptive_avg_pool2d(
            frame_features.view(B * N, C, H, W), 1
        ).view(
            B, N, C
        )  # (B, N, 2048)

        total_loss = 0.0
        for i in range(B):
            # Get class center for this video
            center = class_centers[labels[i]]  # (2048,)

            # Compute feature distances (Paper Eq. 2)
            distances = torch.norm(
                frame_vectors[i] - center.unsqueeze(0), dim=1
            )  # (N,)

            # Get attention weights
            weights = attention_weights[i].squeeze()  # (N,)

            # Normalize
            d_norm = distances / (distances.norm() + 1e-8)
            w_norm = (1 - weights) / ((1 - weights).norm() + 1e-8)

            # Compute Gram matrices
            gram_d = torch.outer(d_norm, d_norm)  # (N, N)
            gram_w = torch.outer(w_norm, w_norm)  # (N, N)

            # L2 loss between Gram matrices
            loss = torch.norm(gram_w - gram_d)
            total_loss += loss

        return total_loss / B


class StandardTripletLoss(nn.Module):
    """Standard Triplet Loss with mining strategies.
    Args:
        margin: Margin for triplet loss
        mining: Mining strategy - 'hard', 'semi-hard', or 'all'
    """

    def __init__(self, margin=1.0, mining="hard"):
        super(StandardTripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining  # 'hard', 'semi-hard', or 'all'

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
                # Limit to avoid explosion
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
    """KGA-Net: Keyframe Guidance Attention Network.

    The last 2 layers (avgpool and fc) of ResNet-50 are removed to obtain feature maps. This helps in
    preserving spatial information for the KGA module.
    Args:
        num_classes: Number of output classes
        feature_dim: Dimension of feature maps from backbone
        reduction: Reduction ratio for KGA module
        aggregation_type: 'avg', 'max', or 'attention' for temporal aggregation
        pretrained: Whether to use pretrained backbone
        use_gradient_checkpointing: Whether to use gradient checkpointing
    """

    def __init__(self, num_classes=2, feature_dim=2048, backbone=None):
        super().__init__()

        # Backbone (ResNet-50 without avgpool and fc)
        if backbone is not None:
            self.backbone = backbone
        else:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Frame attention module
        self.frame_attention = FrameAttention(feature_dim)

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, video_frames, return_features=False) -> torch.Tensor:
        """
        Forward pass for video classification.

        Args:
            video_frames: (B, N, 3, H, W) - video frames
            return_features: whether to return intermediate features

        Note: keyframe_indices parameter removed - not used in the paper's method
        """
        batch_size, num_frames, rgb_channels, height, width = video_frames.shape

        # Extract frame features
        frames_flat = video_frames.view(
            batch_size * num_frames, rgb_channels, height, width
        )
        features_flat = self.backbone(frames_flat)  # (B*N, 2048, h, w)
        _, feat_dim, feat_h, feat_w = features_flat.shape
        frame_features = features_flat.view(
            batch_size, num_frames, feat_dim, feat_h, feat_w
        )  # (B, N, 2048, h, w)

        # Compute attention weights (Paper Eq. 1)
        attention_weights = self.frame_attention(frame_features)  # (B, N, 1, 1, 1)

        # Weighted aggregation
        weighted_features = frame_features * attention_weights
        aggregated = weighted_features.sum(dim=1)  # (B, C, h, w)

        # Classification
        pooled = self.global_pool(aggregated).view(batch_size, -1)  # (B, C)
        logits = self.classifier(pooled)

        if return_features:
            return logits, {
                "frame_features": frame_features,
                "attention_weights": attention_weights,
                "aggregated_features": aggregated,
            }
        return logits


class ImageClassificationNetwork(nn.Module):
    """Image classification network that can share the same 2D backbone as KGANet."""

    def __init__(self, num_classes=2, backbone=None):
        super(ImageClassificationNetwork, self).__init__()

        if backbone is not None:
            self.backbone = backbone
        else:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # For image classification we pool the spatial feature map to a vector
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        """To-DO: Experiment with adding LayerNorm here"""
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, num_classes))

    def forward(self, images: torch.Tensor, return_features: bool = False):
        # images: (Batch_size, 3, Height, Width) -> backbone -> (Batch_size, 2048, feat_H, feat_W)
        feats = self.backbone(images)
        pooled = self.global_pool(feats).view(feats.size(0), -1)  # (B, 2048)
        logits = self.classifier(pooled)

        if return_features:
            return logits, pooled  # Return 2048-dim features for center loss
        return logits


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def calculate_youden_threshold(model, val_loader, device):
    """Calculate optimal classification threshold using Youden's index.

    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run inference on

    Returns:
        float: Optimal threshold value
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x in val_loader:
            if len(x) == 2:
                inputs, labels = x
            else:
                inputs, labels, _ = x
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Positive class probability

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    probs = np.array(all_probs)
    labels = np.array(all_labels)

    # find optimal threshold
    fpr, tpr, thresholds = roc_curve(labels, probs)
    idx = np.argmax(tpr - fpr)  # Youden's index

    return thresholds[idx]


def train_joint_epoch(
    image_model,
    video_model,
    image_loader,
    video_loader,
    image_optimizer,
    video_optimizer,
    backbone_optimizer,
    center_optimizer,
    device,
    center_loss_fn,
    image_scheduler,
    video_scheduler,
    backbone_scheduler,
    alpha_center=0.5,
    video_loss_type="coherence",
    mining="hard",
    lambda_coh=1.0,  # Paper sets λ=1
    start_iter=0,
    max_iters=8000,
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

    # Classification loss
    cls_criterion = nn.CrossEntropyLoss()

    # Initialize auxiliary loss
    if video_loss_type == "coherence":
        video_aux_criterion = CoherenceLoss(device=device)
    elif video_loss_type == "triplet_standard":
        video_aux_criterion = StandardTripletLoss(margin=1.0, mining=mining)
    else:
        raise ValueError(f"Unknown video_loss_type: {video_loss_type}")

    # use_amp = device.type == "cuda"
    use_amp = False
    img_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    vid_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    image_iter = iter(image_loader)
    video_iter = iter(video_loader)

    # Use min() to process both together (1:1 sampling)
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

        # zero grads for backbone + image head + center
        backbone_optimizer.zero_grad(set_to_none=True)
        image_optimizer.zero_grad(set_to_none=True)
        center_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            img_logits, img_features = image_model(images, return_features=True)
            img_cls_loss = cls_criterion(img_logits, img_labels)
            img_center_loss = center_loss_fn(img_features, img_labels)
            # Paper Eq. 4 component: L^I_CE + L_Center
            img_loss = img_cls_loss + alpha_center * img_center_loss

        img_scaler.scale(img_loss).backward()
        img_scaler.unscale_(image_optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(image_model.classifier.parameters())
            + list(image_model.backbone.parameters()),
            max_norm=1.0,
        )

        # step optimizers: image head and backbone
        img_scaler.step(image_optimizer)
        img_scaler.step(backbone_optimizer)

        # Update centers with scaled gradient
        for param in center_loss_fn.parameters():
            if param.grad is not None:
                param.grad.data *= 1.0 / alpha_center
        img_scaler.step(center_optimizer)
        img_scaler.update()

        total_image_loss += img_loss.item()
        _, img_pred = img_logits.max(1)
        image_total += img_labels.size(0)
        image_correct += img_pred.eq(img_labels).sum().item()

        # ========== VIDEO BATCH ==========
        try:
            frames, vid_labels, _ = next(video_iter)
        except StopIteration:
            video_iter = iter(video_loader)
            frames, vid_labels, _ = next(video_iter)

        frames = frames.to(device, non_blocking=True)
        vid_labels = vid_labels.to(device, non_blocking=True)

        backbone_optimizer.zero_grad(set_to_none=True)
        video_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            vid_logits, vid_features = video_model(frames, return_features=True)

            # Video-level classification loss (Paper Eq. 4: L^V_CE)
            vid_cls_loss = cls_criterion(vid_logits, vid_labels)

            # Frame-level classification loss (Section 3.1, not in Eq. 4)
            # "the frame-level cross-entropy loss is also applied to facilitate training"
            batch_size, num_frames = frames.shape[0], frames.shape[1]
            frame_features = vid_features["frame_features"]  # (B, N, 2048, H, W)
            _, _, C, H, W = frame_features.shape

            # Pool and classify each frame
            frame_pooled = video_model.global_pool(
                frame_features.view(batch_size * num_frames, C, H, W)
            ).view(batch_size * num_frames, -1)
            frame_logits = video_model.classifier(frame_pooled)
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

            # Total video loss
            # Paper Eq. 4 only includes: L^V_CE + λ · L_Coh, where λ=1
            # But Section 3.1 also mentions frame-level loss, so we include it
            vid_loss = vid_cls_loss + frame_cls_loss + lambda_coh * vid_aux_loss

        # Backward pass
        vid_scaler.scale(vid_loss).backward()
        vid_scaler.unscale_(video_optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(video_model.frame_attention.parameters())
            + list(video_model.classifier.parameters())
            + list(video_model.backbone.parameters()),
            max_norm=1.0,
        )

        vid_scaler.step(video_optimizer)
        vid_scaler.step(backbone_optimizer)
        vid_scaler.update()

        total_video_loss += vid_loss.item()
        _, vid_pred = vid_logits.max(1)
        video_total += vid_labels.size(0)
        video_correct += vid_pred.eq(vid_labels).sum().item()

        iters_done += 1

        # Step schedulers per iteration
        image_scheduler.step()
        video_scheduler.step()
        backbone_scheduler.step()

        # Update progress bar
        pbar.set_postfix(
            {
                "iter": f"{start_iter + iters_done}/{max_iters}",
                "img_acc": f"{100.*image_correct/image_total:.1f}%",
                "vid_acc": f"{100.*video_correct/video_total:.1f}%",
            }
        )

        # Free memory
        del images, img_labels, frames, vid_labels

    return (
        total_image_loss / iters_done,  # average image loss
        total_video_loss / iters_done,  # average video loss
        100.0 * image_correct / image_total,  # image accuracy
        100.0 * video_correct / video_total,  # video accuracy
        iters_done,  # return iteration count
    )


def train_joint_kga_net(
    image_root_dir,
    image_annotation_file,
    video_root_dir,
    video_train_annotation,
    video_val_annotation,
    total_iter=8000,
    image_batch_size=32,
    video_batch_size=4,
    num_frames=32,
    learning_rate=0.005,
    alpha_center=0.5,
    lambda_coh=1.0,  # Paper: λ=1
    video_loss_type="triplet_standard",
    mining="hard",
    train_split=0.8,
    save_dir="checkpoints/joint_model",
    num_workers=4,
):
    """Train image and video models jointly."""
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    full_image_dataset = BUSIImageDataset(image_root_dir, image_annotation_file)
    total_size = len(full_image_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size

    torch.manual_seed(42)
    train_indices, val_indices = torch.utils.data.random_split(
        range(total_size), [train_size, val_size]
    )

    train_image_dataset = BUSIImageDataset(
        image_root_dir, image_annotation_file, train_img_transform
    )
    train_image_dataset = torch.utils.data.Subset(
        train_image_dataset, train_indices.indices
    )

    val_image_dataset = BUSIImageDataset(
        image_root_dir, image_annotation_file, val_img_transform
    )
    val_image_dataset = torch.utils.data.Subset(val_image_dataset, val_indices.indices)

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
        backbone=shared_backbone,
    ).to(device)

    video_model = KGANet(
        num_classes=2,
        feature_dim=2048,
        backbone=shared_backbone,
    ).to(device)

    # Center loss operates on 2048-dim features
    center_loss_fn = CenterLoss(num_classes=2, feature_dim=2048, device=device)

    # ==================== OPTIMIZERS ====================
    # Shared backbone optimizer
    backbone_optimizer = torch.optim.SGD(
        shared_backbone.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )

    # Image head optimizer (only classifier, backbone is separate)
    image_optimizer = torch.optim.SGD(
        image_model.classifier.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-3,
    )

    # Video head optimizer (attention + classifier, backbone is separate)
    video_head_params = []
    # collect video head params (exclude backbone)
    for name, p in video_model.named_parameters():
        if "backbone" not in name:
            video_head_params.append(p)

    video_optimizer = torch.optim.SGD(
        video_head_params,
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-3,
    )

    # Center optimizer
    center_optimizer = torch.optim.SGD(
        center_loss_fn.parameters(),
        lr=0.5,
        momentum=0.9,
        nesterov=True,
        dampening=0,
    )

    # ==================== SCHEDULERS ====================
    iterations_per_epoch = min(len(image_train_loader), len(video_train_loader))
    num_epochs = math.ceil(total_iter / iterations_per_epoch)

    logger.info(f"Iterations per epoch: {iterations_per_epoch}")
    logger.info(f"Training for {num_epochs} epochs to reach {total_iter} iterations")

    # Iteration-based LR schedule (paper: warmup + step decay)
    def lr_lambda(current_iter):
        if current_iter < 1000:  # warmup
            return float(current_iter) / 1000.0
        elif current_iter < 4000:
            return 1.0
        elif current_iter < 6000:
            return 0.1
        else:
            return 0.01

    image_scheduler = torch.optim.lr_scheduler.LambdaLR(image_optimizer, lr_lambda)
    video_scheduler = torch.optim.lr_scheduler.LambdaLR(video_optimizer, lr_lambda)
    backbone_scheduler = torch.optim.lr_scheduler.LambdaLR(
        backbone_optimizer, lr_lambda
    )

    # ==================== TRAINING LOOP ====================
    logger.info(f"\nStarting joint training...")
    logger.info(f"Alpha (Center Loss Weight): {alpha_center}")
    logger.info(f"Lambda (Coherence Loss Weight): {lambda_coh}")

    best_img_acc = 0.0
    best_vid_acc = 0.0
    global_iter = 0

    for epoch in range(num_epochs):
        if global_iter >= total_iter:
            break

        logger.info(f"\nEpoch {epoch+1}/{num_epochs} (Iter {global_iter}/{total_iter})")

        img_loss, vid_loss, img_acc, vid_acc, iters_done = train_joint_epoch(
            image_model,
            video_model,
            image_train_loader,
            video_train_loader,
            image_optimizer,
            video_optimizer,
            backbone_optimizer,
            center_optimizer,
            device,
            center_loss_fn,
            image_scheduler,
            video_scheduler,
            backbone_scheduler,
            alpha_center,
            video_loss_type,
            mining,
            lambda_coh,
            global_iter,
            total_iter,
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

        # Calculate optimal thresholds using Youden index
        image_threshold = calculate_youden_threshold(
            image_model, image_val_loader, device
        )
        video_threshold = calculate_youden_threshold(
            video_model, video_val_loader, device
        )

        logger.info(
            f"Optimal thresholds - Image: {image_threshold:.3f}, Video: {video_threshold:.3f}"
        )

        # Save checkpoints periodically
        if (epoch + 1) % 60 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state": shared_backbone.state_dict(),
                    "image_head": image_model.state_dict(),
                    "val_acc": val_img_acc,
                    "threshold": image_threshold,
                },
                os.path.join(save_dir, f"image_model_epoch_{epoch+1}.pth"),
            )
            logger.info(f"✓ Saved image model checkpoint for epoch {epoch+1}")

            if video_loss_type == "triplet_standard":
                vid_loss_type_str = f"{video_loss_type}_{mining}"
            else:
                vid_loss_type_str = video_loss_type
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state": shared_backbone.state_dict(),
                    "video_head": video_model.state_dict(),
                    "val_acc": val_vid_acc,
                    "threshold": video_threshold,
                },
                os.path.join(
                    save_dir, f"video_model_{vid_loss_type_str}_epoch_{epoch+1}.pth"
                ),
            )
            logger.info(f"✓ Saved video model checkpoint for epoch {epoch+1}")

        # Save best models
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
            if video_loss_type == "triplet_standard":
                vid_loss_type_str = f"{video_loss_type}_{mining}"
            else:
                vid_loss_type_str = video_loss_type
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state": shared_backbone.state_dict(),
                    "video_head": video_model.state_dict(),
                    "val_acc": val_vid_acc,
                    "threshold": video_threshold,
                },
                os.path.join(save_dir, f"best_video_model_{vid_loss_type_str}.pth"),
            )
            logger.info(f"✓ Saved best video model (Acc: {val_vid_acc:.2f}%)")

    logger.info("\n" + "=" * 70)
    logger.info(f"Training completed!")
    logger.info(f"Best Image Acc: {best_img_acc:.2f}%")
    logger.info(f"Best Video Acc: {best_vid_acc:.2f}%")
    logger.info("=" * 70)

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
        for frames, labels, _ in pbar:
            frames = frames.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(frames)
            loss = criterion(logits, labels)

            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"}
            )

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate_with_threshold(model, val_loader, device):
    """Validate using Youden's optimal threshold."""
    model.eval()
    # First pass: calculate optimal threshold
    threshold = calculate_youden_threshold(model, val_loader, device)

    # Second pass: evaluate with optimal threshold
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= threshold).int()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100
    tn = np.sum((all_preds == 0) & (all_labels == 0))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    tp = np.sum((all_preds == 1) & (all_labels == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "sensitivity": sensitivity * 100,
        "specificity": specificity * 100,
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Dataset configuration
    IMAGE_ROOT_DIR = "./data/busi/"
    IMAGE_ANNOTATION = "data/busi_bboxes.json"
    VIDEO_ROOT_DIR = "./data"
    VIDEO_TRAIN_ANNOTATION = "imagenet_vid_train_15frames.json"
    VIDEO_VAL_ANNOTATION = "imagenet_vid_val.json"

    # Training configuration
    IMAGE_BATCH_SIZE = 8
    VIDEO_BATCH_SIZE = 8
    NUM_FRAMES = 32

    # Loss configuration
    # Options: 'coherence', 'triplet_coherence', 'triplet_standard'
    LOSS_TYPE = "coherence"
    MINING = None  # 'hard', 'semi-hard', or 'all' (only for triplet_standard)
    ALPHA_CENTER = 0.5  # Weight for center loss

    # Path to save Checkpoints
    SAVE_DIR = "notebooks/joint_checkpoints"

    print("=" * 70)
    print("KGA-Net Training Pipeline")
    print("=" * 70)
    # for LOSS_TYPE, MINING in [
    #     ("coherence", None),
    #     # ("triplet_standard", "hard"),
    #     # ("triplet_standard", "semi-hard"),
    #     # ("triplet_standard", "all"),
    # ]:
    for NUM_FRAMES in [8, 16]:  # skipping 32 as gpu is giving out of memory error
        # for NUM_FRAMES in [32]:
        logger.info("\n" + "=" * 70)
        logger.info(f"Video Batch Size: {VIDEO_BATCH_SIZE}")
        logger.info(f"Image Batch Size: {IMAGE_BATCH_SIZE}")
        logger.info(f"Number of Frames: {NUM_FRAMES}")
        logger.info(f"Alpha Center (Center Loss Weight): {ALPHA_CENTER}")

        if LOSS_TYPE != "triplet_standard":
            logger.info(f"\nStarting training with loss type: {LOSS_TYPE}")
        else:
            logger.info(
                f"\nStarting training with loss type: {LOSS_TYPE}, Mining: {MINING}"
            )
        print("=" * 70)
        image_model, video_model = train_joint_kga_net(
            # Image dataset
            image_root_dir=IMAGE_ROOT_DIR,
            image_annotation_file=IMAGE_ANNOTATION,
            # Video dataset
            video_root_dir=VIDEO_ROOT_DIR,
            video_train_annotation=VIDEO_TRAIN_ANNOTATION,
            video_val_annotation=VIDEO_VAL_ANNOTATION,
            # Config
            video_loss_type=LOSS_TYPE,
            mining=MINING,
            alpha_center=ALPHA_CENTER,
            image_batch_size=IMAGE_BATCH_SIZE,
            video_batch_size=VIDEO_BATCH_SIZE,
            num_frames=NUM_FRAMES,
            save_dir=SAVE_DIR,
        )

    print("\n✓ Training completed successfully!")
    print(f"✓ Model checkpoints saved in: {SAVE_DIR}/")
    print("\n" + "=" * 70)
