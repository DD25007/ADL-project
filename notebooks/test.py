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

# Clear logging file
if os.path.exists("training.log"):
    os.remove("training.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATASET CLASS FOR FRAME-BY-FRAME VIDEO DATA
# ============================================================================


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
            self._owns_backbone = False
        else:
            if pretrained:
                weights = ResNet50_Weights.IMAGENET1K_V2
            else:
                weights = None
            resnet = resnet50(weights=weights)
            self.backbone = nn.Sequential(
                *list(resnet.children())[:-2]
            )  # Remove avgpool and fc
            self._owns_backbone = True

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


class ImageClassificationNetwork(nn.Module):
    """Image classification network that can share the same 2D backbone as KGANet."""

    def __init__(self, num_classes=2, feature_dim=512, pretrained=True, backbone=None):
        super(ImageClassificationNetwork, self).__init__()

        # If backbone is provided it should be the same conv stack as KGANet (children()[:-2])
        if backbone is not None:
            self.backbone = backbone
            self._owns_backbone = False
        else:
            if pretrained:
                weights = ResNet50_Weights.IMAGENET1K_V2
            else:
                weights = None
            resnet = resnet50(weights=weights)
            # keep convs only, not avgpool/fc
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            self._owns_backbone = True

        # For image classification we pool the spatial feature map to a vector
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # convert 2048 -> feature_dim
        self.feature_layer = nn.Sequential(
            nn.Linear(2048, feature_dim), nn.ReLU(), nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, images, return_features=False):
        # images: (B, 3, H, W) -> backbone -> (B, 2048, Hf, Wf)
        feats = self.backbone(images)
        pooled = self.global_pool(feats).view(feats.size(0), -1)  # (B, 2048)
        features = self.feature_layer(pooled)  # (B, feature_dim)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits


# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================


def train_joint_epoch(
    image_model,
    video_model,
    image_loader,
    video_loader,
    image_head_optimizer,
    video_head_optimizer,
    backbone_optimizer,
    center_optimizer,
    device,
    center_loss_fn,
    alpha_center=0.5,
    alpha_video=0.5,
    video_loss_type="triplet_standard",
    mining="hard",
):
    """Train both image and video models jointly for one epoch.
    Args:
        image_model: Image classification model
        video_model: Video classification model
        image_loader: DataLoader for image data
        video_loader: DataLoader for video data
        image_optimizer: Optimizer for image model
        video_optimizer: Optimizer for video model
        center_optimizer: Optimizer for center loss parameters
        device: Device to run training on
        center_loss_fn: Center loss function
        alpha_center: Weight for center loss
        alpha_video: Weight for video auxiliary loss
        video_loss_type: Type of auxiliary loss for video model
        mining: Mining strategy for triplet loss
    Returns:
        tuple:
        - avg_image_loss: Average loss for image model
        - avg_video_loss: Average loss for video model
        - image_accuracy: Accuracy for image model
        - video_accuracy: Accuracy for video model
    """

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
        video_aux_criterion = CoherenceLoss()
        aux_requires_labels = False
    elif video_loss_type == "triplet_coherence":
        video_aux_criterion = TripletCoherenceLoss(margin=1.0)
        aux_requires_labels = True
    else:
        video_aux_criterion = StandardTripletLoss(margin=1.0, mining=mining)
        aux_requires_labels = True

    # use_amp = device.type == "cuda"
    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    image_iter = iter(image_loader)
    video_iter = iter(video_loader)

    max_iters = max(len(image_loader), len(video_loader))
    pbar = tqdm(range(max_iters), desc="Joint Training")

    for _ in pbar:
        # ---------- IMAGE BATCH ----------
        try:
            images, img_labels = next(image_iter)
        except StopIteration:
            image_iter = iter(image_loader)
            images, img_labels = next(image_iter)

        images = images.to(device, non_blocking=True)
        img_labels = img_labels.to(device, non_blocking=True)

        # zero grads for backbone + image head + center
        backbone_optimizer.zero_grad(set_to_none=True)
        image_head_optimizer.zero_grad(set_to_none=True)
        center_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            img_logits, img_features = image_model(images, return_features=True)
            img_cls_loss = cls_criterion(img_logits, img_labels)
            img_center_loss = center_loss_fn(img_features, img_labels)
            img_loss = img_cls_loss + alpha_center * img_center_loss

        scaler.scale(img_loss).backward()
        # clip grads (applies to backbone + image head)
        scaler.unscale_(image_head_optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(image_model.feature_layer.parameters())
            + list(image_model.classifier.parameters())
            + list(image_model.backbone.parameters()),
            max_norm=1.0,
        )

        # step optimizers: image head and backbone
        scaler.step(image_head_optimizer)
        scaler.step(backbone_optimizer)

        # update centers
        for param in center_loss_fn.parameters():
            if param.grad is not None:
                param.grad.data *= 1.0 / alpha_center
        scaler.step(center_optimizer)
        scaler.update()

        total_image_loss += img_loss.item()
        _, img_pred = img_logits.max(1)
        image_total += img_labels.size(0)
        image_correct += img_pred.eq(img_labels).sum().item()

        # ---------- VIDEO BATCH ----------
        try:
            frames, vid_labels, keyframe_indices = next(video_iter)
        except StopIteration:
            video_iter = iter(video_loader)
            frames, vid_labels, keyframe_indices = next(video_iter)

        frames = frames.to(device, non_blocking=True)
        vid_labels = vid_labels.to(device, non_blocking=True)
        keyframe_indices = keyframe_indices.to(device)

        backbone_optimizer.zero_grad(set_to_none=True)
        video_head_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            vid_logits, vid_features = video_model(
                frames, keyframe_indices, return_features=True
            )
            vid_cls_loss = cls_criterion(vid_logits, vid_labels)

            batch_size, num_frames = frames.shape[0], frames.shape[1]
            if aux_requires_labels and video_loss_type == "triplet_standard":
                frame_labels = vid_labels.unsqueeze(1).expand(-1, num_frames)
                vid_aux_loss = video_aux_criterion(
                    vid_features["frame_features"], frame_labels
                )
            elif aux_requires_labels:
                frame_labels = vid_labels.unsqueeze(1).expand(-1, num_frames)
                vid_aux_loss = video_aux_criterion(
                    vid_features["frame_features"],
                    vid_features["keyframe_center"],
                    frame_labels,
                    vid_labels,
                )
            else:
                vid_aux_loss = video_aux_criterion(
                    vid_features["frame_features"], vid_features["keyframe_center"]
                )

            vid_loss = vid_cls_loss + alpha_video * vid_aux_loss

        scaler.scale(vid_loss).backward()
        scaler.unscale_(video_head_optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(video_model.kga.parameters())
            + list(video_model.kfc.parameters())
            + list(video_model.temporal_agg.parameters())
            + list(video_model.classifier.parameters())
            + list(video_model.backbone.parameters()),
            max_norm=1.0,
        )

        scaler.step(video_head_optimizer)
        scaler.step(backbone_optimizer)  # update shared backbone again for video batch
        scaler.update()

        total_video_loss += vid_loss.item()
        _, vid_pred = vid_logits.max(1)
        video_total += vid_labels.size(0)
        video_correct += vid_pred.eq(vid_labels).sum().item()

        pbar.set_postfix(
            {
                "img_acc": f"{100.*image_correct/image_total:.1f}%",
                "vid_acc": f"{100.*video_correct/video_total:.1f}%",
            }
        )

        # free memory
        del images, img_labels, frames, vid_labels, keyframe_indices

    return (
        total_image_loss / len(image_loader),
        total_video_loss / len(video_loader),
        100.0 * image_correct / image_total,
        100.0 * video_correct / video_total,
    )


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


def train_joint_kga_net(
    image_root_dir,
    image_annotation_file,
    video_root_dir,
    video_train_annotation,
    video_val_annotation,
    # Training params
    num_epochs=50,
    image_batch_size=32,
    video_batch_size=4,
    num_frames=32,
    image_lr=0.001,
    video_lr=0.0001,
    alpha_center=0.5,
    alpha_video=0.5,
    feature_dim=512,
    video_loss_type="triplet_standard",
    mining="hard",
    train_split=0.8,
    save_dir="checkpoints/joint_model",
    num_workers=4,
):
    """
    Train image and video models jointly.

    Args:
        image_root_dir: Root directory for image dataset
        image_annotation_file: Annotation file for image dataset
        video_root_dir: Root directory for video dataset
        video_train_annotation: Training annotation file for video dataset
        video_val_annotation: Validation annotation file for video dataset
        num_epochs: Number of training epochs
        image_batch_size: Batch size for image model
        video_batch_size: Batch size for video model
        num_frames: Number of frames per video
        image_lr: Learning rate for image model
        video_lr: Learning rate for video model
        alpha_center: Weight for center loss
        alpha_video: Weight for video auxiliary loss
        feature_dim: Feature dimension for image model
        video_loss_type: Type of auxiliary loss for video model
        mining: Mining strategy for triplet loss ('hard', 'semi-hard', 'all')
        train_split: Train/validation split ratio for image dataset
        save_dir: Directory to save checkpoints
        num_workers: Number of DataLoader workers

    Returns:
        tuple:
        - image_model: Trained image classification model
        - video_model: Trained video classification model

    """

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
        feature_dim=feature_dim,
        pretrained=False,
        backbone=shared_backbone,
    ).to(device)

    video_model = KGANet(
        num_classes=2,
        feature_dim=2048,
        reduction=16,
        aggregation_type="attention",
        pretrained=False,
        backbone=shared_backbone,
    ).to(device)

    # center loss (feature_dim must match image_model feature_layer output)
    center_loss_fn = CenterLoss(num_classes=2, feature_dim=feature_dim, device=device)

    # ---------------- OPTIMIZERS ----------------
    # backbone optimizer (shared)
    backbone_optimizer = torch.optim.Adam(
        shared_backbone.parameters(), lr=video_lr, weight_decay=1e-4
    )

    # image-specific optimizer: feature_layer + classifier
    image_head_params = list(image_model.feature_layer.parameters()) + list(
        image_model.classifier.parameters()
    )
    image_head_optimizer = torch.optim.Adam(
        image_head_params, lr=image_lr, weight_decay=1e-4
    )

    # video-specific optimizer: kga, kfc, temporal_agg, classifier
    video_head_params = []
    # collect video head params (exclude backbone)
    for name, p in video_model.named_parameters():
        if "backbone" in name:
            continue
        video_head_params.append(p)
    video_head_optimizer = torch.optim.Adam(
        video_head_params, lr=video_lr, weight_decay=1e-4
    )

    center_optimizer = torch.optim.SGD(center_loss_fn.parameters(), lr=0.5)

    image_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        image_head_optimizer, mode="max", factor=0.5, patience=5
    )
    video_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        video_head_optimizer, mode="max", factor=0.5, patience=5
    )

    logger.info(f"\nStarting joint training for {num_epochs} epochs...")
    best_img_acc = 0.0
    best_vid_acc = 0.0

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        img_loss, vid_loss, img_acc, vid_acc = train_joint_epoch(
            image_model,
            video_model,
            image_train_loader,
            video_train_loader,
            image_head_optimizer,
            video_head_optimizer,
            backbone_optimizer,
            center_optimizer,
            device,
            center_loss_fn,
            alpha_center,
            alpha_video,
            video_loss_type,
            mining,
        )

        logger.info(f"Train - Image Loss: {img_loss:.4f}, Acc: {img_acc:.2f}%")
        logger.info(f"Train - Video Loss: {vid_loss:.4f}, Acc: {vid_acc:.2f}%")

        # Validate image model
        image_model.eval()
        val_img_loss, val_img_acc = 0.0, 0.0
        img_correct, img_total = 0, 0
        with torch.no_grad():
            for images, labels in image_val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = image_model(images)
                _, pred = logits.max(1)
                img_total += labels.size(0)
                img_correct += pred.eq(labels).sum().item()
        val_img_acc = 100.0 * img_correct / img_total

        # Validate video model
        val_vid_loss, val_vid_acc = validate(video_model, video_val_loader, device)

        logger.info(f"Val   - Image Loss: {val_img_loss:.4f}, Acc: {val_img_acc:.2f}%")
        logger.info(f"Val   - Video Loss: {val_vid_loss:.4f}, Acc: {val_vid_acc:.2f}%")

        # Schedulers
        image_scheduler.step(val_img_acc)
        video_scheduler.step(val_vid_acc)

        # Save image and video model checkpoints after each 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state": shared_backbone.state_dict(),
                    "image_head": image_model.state_dict(),
                    "val_acc": val_img_acc,
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
                    "image_head": image_model.state_dict(),  # contains backbone entries too but same values
                    "val_acc": val_img_acc,
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
                },
                os.path.join(save_dir, "best_video_model.pth"),
            )
            logger.info(f"✓ Saved best video model (Acc: {val_vid_acc:.2f}%)")

    logger.info("\n" + "=" * 70)
    logger.info(f"Training completed!")
    logger.info(f"Best Image Acc: {best_img_acc:.2f}%")
    logger.info(f"Best Video Acc: {best_vid_acc:.2f}%")
    logger.info("=" * 70)

    return image_model, video_model


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # ========================================================================
    # Configuration
    # ========================================================================

    # Dataset configuration
    IMAGE_ROOT_DIR = "./data/busi/"
    IMAGE_ANNOTATION = "data/busi_bboxes.json"
    VIDEO_ROOT_DIR = "./data"
    VIDEO_TRAIN_ANNOTATION = "imagenet_vid_train_15frames.json"
    VIDEO_VAL_ANNOTATION = "imagenet_vid_val.json"

    # Training configuration
    NUM_EPOCHS = 50
    IMAGE_BATCH_SIZE = 32
    VIDEO_BATCH_SIZE = 4
    NUM_FRAMES = 32  # Match the number of frames in vid_train_frames
    LEARNING_RATE = 0.0005  # 0.001, 5e-3, 7e-4

    # Loss configuration
    # Options: 'coherence', 'triplet_coherence', 'triplet_standard'
    LOSS_TYPE = "triplet_standard"
    MINING = "hard"  # 'hard', 'semi-hard', or 'all' (only for triplet_standard)
    ALPHA = 0.5  # Weight for auxiliary loss

    # Path to save Checkpoints
    SAVE_DIR = "notebooks/joint_checkpoints"

    # ========================================================================
    # Start Training
    # ========================================================================
    print("=" * 70)
    print("KGA-Net Training Pipeline")
    print("=" * 70)
    # Start training
    for LOSS_TYPE, MINING in [
        ("coherence", None),
        ("triplet_coherence", None),
        ("triplet_standard", "hard"),
        ("triplet_standard", "semi-hard"),
        ("triplet_standard", "all"),
    ]:
        print(f"Video Batch Size: {VIDEO_BATCH_SIZE}")
        print(f"Image Batch Size: {IMAGE_BATCH_SIZE}")
        print(f"Number of Frames: {NUM_FRAMES}")
        print(f"Learning Rate: {LEARNING_RATE}")
        print(f"Alpha (Auxiliary Loss Weight): {ALPHA}")
        print(f"Number of Epochs: {NUM_EPOCHS}")

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
            num_epochs=NUM_EPOCHS,
            video_loss_type=LOSS_TYPE,
            mining=MINING,
            alpha_video=ALPHA,
            video_lr=LEARNING_RATE,
            image_lr=LEARNING_RATE,
            image_batch_size=IMAGE_BATCH_SIZE,
            video_batch_size=VIDEO_BATCH_SIZE,
            num_frames=NUM_FRAMES,
            save_dir=SAVE_DIR,
        )

        print("\n✓ Training completed successfully!")
        print(f"✓ Model checkpoints saved in: {SAVE_DIR}/")
        print("\n" + "=" * 70)
