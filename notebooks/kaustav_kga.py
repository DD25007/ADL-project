import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.checkpoint import checkpoint
import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATASET CLASS FOR FRAME-BY-FRAME VIDEO DATA
# ============================================================================


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
        self.category_to_id = {cat["name"]: cat["id"] for cat in data["categories"]}

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


def create_dataloaders(
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
        num_frames=num_frames,
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
    """
    Keyframe Guidance Attention (KGA) module
    Uses keyframe feature center to guide attention on video frames"""

    def __init__(self, feature_dim, reduction=16):
        super(KeyframeGuidanceAttention, self).__init__()
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(feature_dim * 2, feature_dim // reduction, 1)
        self.conv2 = nn.Conv2d(feature_dim // reduction, feature_dim, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, frame_features, keyframe_center):
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


class TripletCoherenceLoss(nn.Module):
    """
    Uses keyframe center as anchor, same-class frames as positive,
    different-class frames as negative
    """

    def __init__(self, margin=1.0):
        super(TripletCoherenceLoss, self).__init__()
        self.margin = margin

    def forward(self, frame_features, keyframe_center, labels, keyframe_labels):
        """
        Args:
            frame_features: (Batch_size, num_frames, Channels, Height, Width) - features from all frames
            keyframe_center: (Batch_size, Channels, Height, Width) - keyframe feature center
            labels: (Batch_size, num_frames) - labels for each frame in video
            keyframe_labels: (Batch_size,) - labels for keyframe center
        Returns:
            loss: scalar - triplet loss value
        """
        batch_size, num_frames, channels, height, width = frame_features.shape

        keyframe_center_vec = F.adaptive_avg_pool2d(keyframe_center, 1).view(
            batch_size, channels
        )
        frame_features_vec = F.adaptive_avg_pool2d(
            frame_features.view(batch_size * num_frames, channels, height, width), 1
        ).view(batch_size, num_frames, channels)

        total_loss = 0.0
        count = 0

        for batch_idx in range(batch_size):
            anchor = keyframe_center_vec[batch_idx]
            anchor_label = keyframe_labels[batch_idx]

            positive_mask = labels[batch_idx] == anchor_label
            negative_mask = labels[batch_idx] != anchor_label

            if positive_mask.sum() == 0 or negative_mask.sum() == 0:
                continue

            positives = frame_features_vec[batch_idx][positive_mask]
            negatives = frame_features_vec[batch_idx][negative_mask]

            pos_dist = torch.sum((positives - anchor.unsqueeze(0)) ** 2, dim=1)
            neg_dist = torch.sum((negatives - anchor.unsqueeze(0)) ** 2, dim=1)

            for pos_d in pos_dist:
                for neg_d in neg_dist:
                    loss = torch.clamp(pos_d - neg_d + self.margin, min=0.0)
                    total_loss += loss
                    count += 1

        if count > 0:
            total_loss = total_loss / count
        else:
            # Better way to return zero loss that maintains gradient flow
            total_loss = frame_features.sum() * 0.0

        return total_loss


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
        use_gradient_checkpointing=True,
    ):
        super(KGANet, self).__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing

        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        resnet = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.kfc = KeyframeFeatureCenter(feature_dim)
        self.kga = KeyframeGuidanceAttention(feature_dim, reduction)
        self.temporal_agg = TemporalAggregation(feature_dim, aggregation_type)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, video_frames, keyframe_indices=None, return_features=False):

        batch_size, num_frames, rgb_channels, height, width = video_frames.shape

        frames_flat = video_frames.view(
            batch_size * num_frames, rgb_channels, height, width
        )
        features_flat = self.backbone(frames_flat)
        _, feat_channels, feat_height, feat_width = features_flat.shape
        frame_features = features_flat.view(
            batch_size, num_frames, feat_channels, feat_height, feat_width
        )

        if keyframe_indices is None:
            keyframe_features = frame_features[:, 0:1, :, :, :]
        else:
            # keyframe_indices: (batch, k)
            # Use torch.gather to avoid Python loop. Build index tensor of shape (batch, k, C, H, W)
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

    # Add parameter groups for different learning rates
    def get_parameter_groups(self):
        return [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.classifier.parameters(), "lr_mult": 1.0},
        ]


# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================


def train_one_epoch(
    model, train_loader, optimizer, device, loss_type="triplet_standard", alpha=0.5
):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_aux_loss = 0.0
    correct = 0
    total = 0

    cls_criterion = nn.CrossEntropyLoss()
    if loss_type == "coherence":
        aux_criterion = CoherenceLoss()
        aux_requires_labels = False
    elif loss_type == "triplet_coherence":
        aux_criterion = TripletCoherenceLoss(margin=1.0)
        aux_requires_labels = True
    else:
        aux_criterion = StandardTripletLoss(margin=1.0, mining="hard")
        aux_requires_labels = True

    use_amp = device == "cuda" and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (frames, labels, keyframe_indices) in enumerate(pbar):
        # Skip batch if OOM occurred previously
        try:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            keyframe_indices = keyframe_indices.to(device)

            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, features = model(frames, keyframe_indices, return_features=True)

                cls_loss = cls_criterion(logits, labels)

                batch_size, num_frames = frames.shape[0], frames.shape[1]
                if aux_requires_labels and loss_type == "triplet_standard":
                    frame_labels = labels.unsqueeze(1).expand(-1, num_frames)
                    aux_loss = aux_criterion(features["frame_features"], frame_labels)
                elif aux_requires_labels:
                    frame_labels = labels.unsqueeze(1).expand(-1, num_frames)
                    aux_loss = aux_criterion(
                        features["frame_features"],
                        features["keyframe_center"],
                        frame_labels,
                        labels,
                    )
                else:
                    aux_loss = aux_criterion(
                        features["frame_features"], features["keyframe_center"]
                    )

                loss = cls_loss + alpha * aux_loss

            scaler.scale(loss).backward()

            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # Statistics
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_aux_loss += aux_loss.item()

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100. * correct / total:.2f}%"}
            )

            # Clear cache more frequently
            if torch.cuda.is_available() and (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()

            # Delete intermediate variables
            del (
                frames,
                labels,
                keyframe_indices,
                logits,
                features,
                loss,
                cls_loss,
                aux_loss,
            )

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n OOM at batch {batch_idx}, skipping...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue
            else:
                raise e

    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_aux_loss = total_aux_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, avg_cls_loss, avg_aux_loss, accuracy


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
    root_dir,
    train_annotation="imagenet_vid_train_15frames.json",
    val_annotation="imagenet_vid_val.json",
    num_epochs=50,
    batch_size=4,
    num_frames=15,
    learning_rate=0.0001,
    loss_type="triplet_standard",
    alpha=0.5,
    save_dir="checkpoints",
):
    """
    Complete training pipeline for KGA-Net.

    Args:
        root_dir: Path to dataset root
        train_annotation: Training JSON filename
        val_annotation: Validation JSON filename
        num_epochs: Number of training epochs
        batch_size: Batch size
        num_frames: Number of frames per video
        learning_rate: Learning rate
        loss_type: 'coherence', 'triplet_coherence', or 'triplet_standard'
        alpha: Weight for auxiliary loss
        save_dir: Directory to save checkpoints
    """

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader = create_dataloaders(
        root_dir=root_dir,
        train_annotation=train_annotation,
        val_annotation=val_annotation,
        batch_size=batch_size,
        num_frames=num_frames,
        num_workers=4,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Create model
    print("\nInitializing model...")
    model = KGANet(
        num_classes=2,
        feature_dim=2048,
        reduction=16,
        aggregation_type="attention",
        pretrained=True,
    )
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Training loop
    print(f"\nStarting training with {loss_type} loss...")
    print("=" * 70)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)

        # Train
        train_loss, train_cls_loss, train_aux_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, loss_type, alpha
        )

        print(
            f"Train - Loss: {train_loss:.4f}, Cls Loss: {train_cls_loss:.4f}, "
            f"Aux Loss: {train_aux_loss:.4f}, Acc: {train_acc:.2f}%"
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, device)

        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]["lr"]
        if old_lr != new_lr:
            print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(save_dir, f"best_model_{loss_type}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "train_acc": train_acc,
                },
                checkpoint_path,
            )
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                save_dir, f"checkpoint_epoch_{epoch+1}_{loss_type}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "train_acc": train_acc,
                },
                checkpoint_path,
            )

    print("\n" + "=" * 70)
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 70)

    return model


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
    NUM_EPOCHS = 50
    BATCH_SIZE = 6  # 4
    NUM_FRAMES = 32  # Match the number of frames in vid_train_frames
    LEARNING_RATE = 0.0005

    # Loss configuration
    # Options: 'coherence', 'triplet_coherence', 'triplet_standard'
    LOSS_TYPE = "triplet_standard"
    ALPHA = 0.5  # Weight for auxiliary loss

    # Save configuration
    SAVE_DIR = "notebooks/checkpoints"

    # ========================================================================
    # Start Training
    # ========================================================================

    print("=" * 70)
    print("KGA-Net Training Pipeline")
    print("=" * 70)
    print(f"Dataset Root: {ROOT_DIR}")
    print(f"Loss Type: {LOSS_TYPE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of Frames: {NUM_FRAMES}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Alpha (Auxiliary Loss Weight): {ALPHA}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print("=" * 70)

    # Check if dataset exists
    if not os.path.exists(ROOT_DIR):
        print(f"\n❌ Error: Dataset directory not found: {ROOT_DIR}")
        print("\nPlease update ROOT_DIR to point to your dataset location.")
        print("Expected structure:")
        print("  data_Amss007_ultrasound_dat/")
        print("  ├── rawframes/")
        print("  │   ├── benign/")
        print("  │   │   └── x28f299ceb056964c/")
        print("  │   └── malignant/")
        print("  ├── imagenet_vid_train_15frames.json")
        print("  └── imagenet_vid_val.json")
    else:
        # Start training
        # for loss_type in ["coherence", "triplet_coherence", "triplet_standard"]:
        for loss_type in ["triplet_coherence", "triplet_standard"]:
            print(f"\nStarting training with loss type: {loss_type}")
            trained_model = train_kga_net(
                root_dir=ROOT_DIR,
                train_annotation=TRAIN_ANNOTATION,
                val_annotation=VAL_ANNOTATION,
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                num_frames=NUM_FRAMES,
                learning_rate=LEARNING_RATE,
                loss_type=loss_type,
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
            checkpoint_path = os.path.join(SAVE_DIR, f"best_model_{loss_type}.pth")
            if os.path.exists(checkpoint_path):
                checkpoint_model = torch.load(checkpoint_path)
                trained_model.load_state_dict(checkpoint_model["model_state_dict"])
                print(f"✓ Loaded best model from epoch {checkpoint_model['epoch']+1}")
                print(f"  Training Accuracy: {checkpoint_model['train_acc']:.2f}%")
                print(f"  Validation Accuracy: {checkpoint_model['val_acc']:.2f}%")

                # Test on validation set
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                trained_model.eval()

                _, val_loader = create_dataloaders(
                    root_dir=ROOT_DIR,
                    train_annotation=TRAIN_ANNOTATION,
                    val_annotation=VAL_ANNOTATION,
                    batch_size=BATCH_SIZE,
                    num_frames=NUM_FRAMES,
                    num_workers=4,
                )

                val_loss, val_acc = validate(trained_model, val_loader, device)
                print(f"\nFinal Test Results:")
                print(f"  Loss: {val_loss:.4f}")
                print(f"  Accuracy: {val_acc:.2f}%")

        print("\n" + "=" * 70)
