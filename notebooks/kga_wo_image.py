import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import logging

# set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training_wo_image.log"), logging.StreamHandler()],
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
    """
    Dataset class for loading ultrasound videos stored as individual frames.
    """

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


class KeyframeFeatureCenter(nn.Module):
    """Keyframe Feature Center (KFC) module."""

    def __init__(self, feature_dim):
        super(KeyframeFeatureCenter, self).__init__()
        self.feature_dim = feature_dim

    def forward(self, keyframe_features):
        """
        Compute the center of keyframe features by averaging.

        Args:
            keyframe_features: Tensor of shape (batch_size, num_keyframes, channels, height, width)
                - batch_size: Number of videos in the batch
                - num_keyframes: Number of keyframes per video
                - channels: Number of feature channels (e.g., 2048 for ResNet-50)
                - height: Spatial height of feature maps
                - width: Spatial width of feature maps

        Returns:
            center: Tensor of shape (batch_size, channels, height, width)
                - Averaged keyframe features representing the keyframe center
        """
        center = torch.mean(keyframe_features, dim=1)
        return center


class KeyframeGuidanceAttention(nn.Module):
    """Keyframe Guidance Attention (KGA) module uses keyframe feature center to guide attention on video frames"""

    def __init__(self, feature_dim, reduction=16):
        super(KeyframeGuidanceAttention, self).__init__()
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(feature_dim * 2, feature_dim // reduction, 1)
        self.conv2 = nn.Conv2d(feature_dim // reduction, feature_dim, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, frame_features: torch.Tensor, keyframe_center: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = frame_features.shape

        keyframe_center_expanded = keyframe_center.unsqueeze(1).expand(
            -1, num_frames, -1, -1, -1
        )
        concatenated = torch.cat([frame_features, keyframe_center_expanded], dim=2)
        concatenated = concatenated.view(
            batch_size * num_frames, channels * 2, height, width
        )

        attention = self.conv1(concatenated)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        attention = attention.view(batch_size, num_frames, channels, height, width)

        attended_features = frame_features * attention
        return attended_features


class TemporalAggregation(nn.Module):
    """Temporal aggregation module to combine frame-level features"""

    def __init__(self, feature_dim, aggregation_type="avg"):
        super(TemporalAggregation, self).__init__()
        self.aggregation_type = aggregation_type

        if aggregation_type == "attention":
            self.attention_fc = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, 1),
            )

    def forward(self, features):
        """
        Args:
            features: (Batch_size, num_frames, Channels, Height, Width) - frame features
        Returns:
            aggregated: (Batch_size, Channels, Height, Width) - temporally aggregated features
        """
        if self.aggregation_type == "avg":
            aggregated = torch.mean(features, dim=1)
        elif self.aggregation_type == "max":
            aggregated = torch.max(features, dim=1)[0]
        elif self.aggregation_type == "attention":
            batch_size, num_frames, channels, height, width = features.shape
            frame_vectors = F.adaptive_avg_pool2d(
                features.view(batch_size * num_frames, channels, height, width), 1
            ).view(batch_size, num_frames, channels)

            attention_weights = self.attention_fc(frame_vectors)
            attention_weights = F.softmax(attention_weights, dim=1)
            attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
            aggregated = torch.sum(features * attention_weights, dim=1)

        return aggregated


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================


class CoherenceLoss(nn.Module):
    """Coherence Loss from the original KGA-Net paper."""

    def __init__(self):
        super(CoherenceLoss, self).__init__()

    def forward(self, frame_features, keyframe_center):
        batch_size, num_frames, channels, height, width = frame_features.shape
        keyframe_center_expanded = keyframe_center.unsqueeze(1).expand(
            -1, num_frames, -1, -1, -1
        )
        diff = frame_features - keyframe_center_expanded
        loss = torch.mean(diff**2)
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

    def forward(self, frame_features, labels):
        if len(frame_features.shape) == 5:
            batch_size, num_frames, channels, height, width = frame_features.shape
            frame_features = F.adaptive_avg_pool2d(
                frame_features.view(batch_size * num_frames, channels, height, width), 1
            ).view(batch_size * num_frames, channels)
            labels = labels.reshape(-1)

        num_samples, channels = frame_features.shape

        dist_matrix = torch.cdist(frame_features, frame_features, p=2)

        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
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

    def __init__(
        self,
        num_classes=2,
        feature_dim=2048,
        reduction=16,
        aggregation_type="attention",
        pretrained=True,
        backbone=None,
        use_gradient_checkpointing=True,
    ):
        super(KGANet, self).__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Use provided backbone if available (shared). Backbone should give spatial features (B, C, H, W)
        if backbone is not None:
            self.backbone = backbone
        else:
            if pretrained:
                weights = ResNet50_Weights.IMAGENET1K_V2
            else:
                weights = None
            resnet = resnet50(weights=weights)
            self.backbone = nn.Sequential(
                *list(resnet.children())[:-2]
            )  # Remove avgpool and fc

        self.kfc = KeyframeFeatureCenter(feature_dim)
        self.kga = KeyframeGuidanceAttention(feature_dim, reduction)
        self.temporal_agg = TemporalAggregation(feature_dim, aggregation_type)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),  # Increase from 0.5
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increase from 0.3
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        video_frames: torch.Tensor,
        keyframe_indices: torch.Tensor = None,
        return_features: bool = False,
    ):

        batch_size, num_frames, rgb_channels, height, width = video_frames.shape

        frames_flat = video_frames.view(
            batch_size * num_frames, rgb_channels, height, width
        )
        features_flat = self.backbone(frames_flat)  # (B*N, C, Hf, Wf)
        _, feat_channels, feat_height, feat_width = features_flat.shape
        frame_features = features_flat.view(
            batch_size, num_frames, feat_channels, feat_height, feat_width
        )

        if keyframe_indices is None:
            keyframe_features = frame_features[:, 0:1, :, :, :]
        else:
            # gather keyframe features
            k = keyframe_indices.shape[1]
            idx = keyframe_indices.view(batch_size, k, 1, 1, 1).expand(
                -1, -1, feat_channels, feat_height, feat_width
            )
            keyframe_features = torch.gather(frame_features, dim=1, index=idx)

        keyframe_center = self.kfc(keyframe_features)
        attended_features = self.kga(frame_features, keyframe_center)
        aggregated_features = self.temporal_agg(attended_features)

        pooled = self.global_pool(aggregated_features).view(batch_size, -1)
        logits = self.classifier(pooled)

        if return_features:
            features_dict = {
                "frame_features": frame_features,
                "keyframe_center": keyframe_center,
                "attended_features": attended_features,
                "aggregated_features": aggregated_features,
            }
            return logits, features_dict

        return logits


# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================


def train_one_epoch(
    video_model,
    video_train_loader: DataLoader,
    video_head_optimizer: torch.optim.Optimizer,
    video_scheduler: torch.optim.lr_scheduler._LRScheduler,
    device,
    alpha_video=0.5,
    video_loss_type="triplet_standard",
    mining="hard",
):
    """Train for one epoch."""
    video_model.train()

    total_video_loss = 0.0
    total_video_cls_loss = 0.0
    correct = 0
    total = 0

    # Classification loss
    cls_criterion = nn.CrossEntropyLoss()

    # use_amp = device.type == "cuda"
    use_amp = False
    vid_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    pbar = tqdm(video_train_loader, desc="Training")

    for frames, vid_labels, keyframe_indices in pbar:
        frames = frames.to(device)
        vid_labels = vid_labels.to(device)
        keyframe_indices = keyframe_indices.to(device)

        video_head_optimizer.zero_grad(set_to_none=True)

        # Forward pass
        with torch.amp.autocast("cuda", enabled=use_amp):
            vid_logits, vid_features = video_model(
                frames, keyframe_indices, return_features=True
            )

            vid_cls_loss = cls_criterion(vid_logits, vid_labels)

            # Auxiliary loss
            batch_size, num_frames = frames.shape[0], frames.shape[1]

            if video_loss_type == "coherence":
                aux_criterion = CoherenceLoss()
                vid_aux_loss = aux_criterion(
                    vid_features["frame_features"], vid_features["keyframe_center"]
                )
            elif video_loss_type == "triplet_standard":
                frame_labels = vid_labels.unsqueeze(1).expand(-1, num_frames)
                aux_criterion = StandardTripletLoss(margin=1.0, mining=mining)
                vid_aux_loss = aux_criterion(
                    vid_features["frame_features"], frame_labels
                )

            # Combined loss: classification + alpha * coherence/triplet loss
            vid_loss = vid_cls_loss + alpha_video * vid_aux_loss

        # Backward pass
        vid_scaler.scale(vid_loss).backward()
        vid_scaler.unscale_(video_head_optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(video_model.kga.parameters())
            + list(video_model.kfc.parameters())
            + list(video_model.temporal_agg.parameters())
            + list(video_model.classifier.parameters())
            + list(video_model.backbone.parameters()),
            max_norm=1.0,
        )

        vid_scaler.step(video_head_optimizer)
        vid_scaler.update()

        # Statistics
        total_video_loss += vid_loss.item()
        total_video_cls_loss += vid_cls_loss.item()

        _, vid_pred = vid_logits.max(1)
        video_total += vid_labels.size(0)
        video_correct += vid_pred.eq(vid_labels).sum().item()

        # Step schedulers per iteration
        video_scheduler.step()

        # Update progress bar
        pbar.set_postfix(
            {
                "vid_loss": f"{vid_loss.item():.4f}",
                "vid_acc": f"{100.*video_correct/video_total:.1f}%",
            }
        )

        # free memory
        del frames, vid_labels, keyframe_indices

    avg_video_loss = total_video_loss / len(video_train_loader)
    avg_video_cls_loss = total_video_cls_loss / len(video_train_loader)
    video_accuracy = 100.0 * correct / total

    return avg_video_loss, avg_video_cls_loss, video_accuracy


def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for frames, labels, keyframe_indices in pbar:
            frames = frames.to(device)
            labels = labels.to(device)
            keyframe_indices = keyframe_indices.to(device)

            # Forward pass
            logits = model(frames, keyframe_indices)
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


def train_kga_net(
    video_root_dir,
    video_train_annotation="imagenet_vid_train_15frames.json",
    video_val_annotation="imagenet_vid_val.json",
    # Training params
    num_epochs=50,
    video_batch_size=4,
    num_frames=32,
    video_lr=0.0001,
    alpha_video=0.5,
    video_loss_type="triplet_standard",
    mining="hard",
    train_split=0.8,
    save_dir="checkpoints/video_only_checkpoints",
    num_workers=4,
):
    """
    Complete training pipeline for KGA-Net.
    """

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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

    logger.info(f"Training samples: {len(video_train_loader.dataset)}")
    logger.info(f"Validation samples: {len(video_val_loader.dataset)}")
    logger.info(f"Training batches: {len(video_train_loader)}")
    logger.info(f"Validation batches: {len(video_val_loader)}")

    # ==================== CREATE MODELS ====================
    logger.info("\nInitializing models...")

    video_model = KGANet(
        num_classes=2,
        feature_dim=2048,
        reduction=16,
        aggregation_type="attention",
        pretrained=False,
        backbone=None,
    ).to(device)

    logger.info(
        f"Model parameters: {sum(p.numel() for p in video_model.parameters()):,}"
    )

    # ================= OPTIMIZERS and SCHEDULERS =================
    video_head_optimizer = torch.optim.SGD(
        video_model.parameters(),
        lr=video_lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-3,  # Increase from 1e-4
    )

    # Iteration-based LR schedule
    def lr_lambda(current_iter):
        if current_iter < 1000:  # warmup
            return float(current_iter) / 1000.0
        elif current_iter < 4000:
            return 1.0
        elif current_iter < 6000:
            return 0.1
        else:
            return 0.01

    video_scheduler = torch.optim.lr_scheduler.LambdaLR(video_head_optimizer, lr_lambda)

    # Training loop
    print(f"\nStarting training with {video_loss_type} loss...")
    print("=" * 70)

    best_vid_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)

        # Train
        avg_video_loss, avg_video_cls_loss, video_accuracy = train_one_epoch(
            video_model=video_model,
            video_train_loader=video_train_loader,
            video_head_optimizer=video_head_optimizer,
            video_scheduler=video_scheduler,
            device=device,
            video_loss_type=video_loss_type,
            alpha_video=alpha_video,
        )

        logger.info(
            f"Train - Video Loss: {avg_video_loss:.4f}, Acc: {video_accuracy:.2f}%, Cls Loss: {avg_video_cls_loss:.4f}"
        )

        # Validate
        val_vid_loss, val_vid_acc = validate(video_model, val_loader, device)

        print(f"Val   - Loss: {val_vid_loss:.4f}, Acc: {val_vid_acc:.2f}%")

        # Learning rate scheduling
        old_lr = video_head_optimizer.param_groups[0]["lr"]
        video_scheduler.step(val_vid_acc)
        new_lr = video_head_optimizer.param_groups[0]["lr"]
        if old_lr != new_lr:
            print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

        # Save best model
        if val_vid_acc > best_vid_acc:
            best_vid_acc = val_vid_acc
            checkpoint_path = os.path.join(save_dir, f"best_model_{loss_type}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": video_model.state_dict(),
                    "optimizer_state_dict": video_head_optimizer.state_dict(),
                    "val_acc": val_vid_acc,
                    "train_acc": video_accuracy,
                },
                checkpoint_path,
            )
            print(f"✓ Saved best model (Val Acc: {val_vid_acc:.2f}%)")

        # Save checkpoint every 60 epochs
        if (epoch + 1) % 60 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": video_model.state_dict(),
                    "optimizer_state_dict": video_head_optimizer.state_dict(),
                    "val_acc": val_vid_acc,
                    "train_acc": video_accuracy,
                },
                checkpoint_path,
            )

    logger.info("\n" + "=" * 70)
    logger.info(f"Training completed!")
    logger.info(f"Best Video Acc: {best_vid_acc:.2f}%")
    logger.info("=" * 70)

    return video_model


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # ========================================================================
    # Configuration
    # ========================================================================

    # Dataset configuration
    ROOT_DIR = "./data"  # Change this to your dataset path
    TRAIN_ANNOTATION = "imagenet_vid_train_15frames.json"
    VAL_ANNOTATION = "imagenet_vid_val.json"

    # Training configuration
    TOTAL_ITERATIONS = 8000
    NUM_EPOCHS = 422
    VIDEO_BATCH_SIZE = 8
    NUM_FRAMES = 32
    LEARNING_RATE = 0.005

    # Loss configuration
    # Options: 'coherence', 'triplet_coherence', 'triplet_standard'
    LOSS_TYPE = "coherence"
    MINING = None  # 'hard', 'semi-hard', or 'all' (only for triplet_standard)
    ALPHA = 0.5  # Weight for auxiliary loss
    ALPHA_CENTER = 0.5  # Weight for center loss

    # Save configuration
    SAVE_DIR = "notebooks/video_only_checkpoints"

    # ========================================================================
    # Start Training
    # ========================================================================

    print("=" * 70)
    print("KGA-Net Training Pipeline")
    print("=" * 70)
    logger.info("\n" + "=" * 70)
    logger.info(f"Video Batch Size: {VIDEO_BATCH_SIZE}")
    logger.info(f"Number of Frames: {NUM_FRAMES}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Alpha (Auxiliary Loss Weight): {ALPHA}")
    logger.info(f"Alpha Center (Center Loss Weight): {ALPHA_CENTER}")
    logger.info(f"Total Iterations:{TOTAL_ITERATIONS}")

    for loss_type in ["coherence", "triplet_coherence", "triplet_standard"]:
        print(f"\nStarting training with loss type: {loss_type}")
        trained_model = train_kga_net(
            root_dir=ROOT_DIR,
            train_annotation=TRAIN_ANNOTATION,
            val_annotation=VAL_ANNOTATION,
            num_epochs=NUM_EPOCHS,
            batch_size=VIDEO_BATCH_SIZE,
            num_frames=NUM_FRAMES,
            learning_rate=LEARNING_RATE,
            loss_type=LOSS_TYPE,
            alpha=ALPHA,
            save_dir=SAVE_DIR,
        )

        print("\n✓ Training completed successfully!")
        print(f"✓ Model checkpoints saved in: {SAVE_DIR}/")

        # ====================================================================
        # Optional: Load and test the best model
        # ====================================================================

        print("\n" + "=" * 70)
        print("Testing Best Model")
        print("=" * 70)

        # Load best model
        checkpoint_path = os.path.join(SAVE_DIR, f"best_model_{LOSS_TYPE}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            trained_model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✓ Loaded best model from epoch {checkpoint['epoch']+1}")
            print(f"  Training Accuracy: {checkpoint['train_acc']:.2f}%")
            print(f"  Validation Accuracy: {checkpoint['val_acc']:.2f}%")

            # Test on validation set
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            trained_model.eval()

            _, val_loader = video_dataloader(
                root_dir=ROOT_DIR,
                train_annotation=TRAIN_ANNOTATION,
                val_annotation=VAL_ANNOTATION,
                batch_size=VIDEO_BATCH_SIZE,
                num_frames=NUM_FRAMES,
                num_workers=4,
            )

            val_vid_loss, val_vid_acc = validate(trained_model, val_loader, device)
            print(f"\nFinal Test Results:")
            print(f"  Loss: {val_vid_loss:.4f}")
            print(f"  Accuracy: {val_vid_acc:.2f}%")

        print("\n" + "=" * 70)
