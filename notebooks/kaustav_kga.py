import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class KeyframeFeatureCenter(nn.Module):
    """
    Keyframe Feature Center (KFC) module
    Computes the center of keyframe features for guidance
    """
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
    Uses keyframe feature center to guide attention on video frames
    """
    def __init__(self, feature_dim, reduction=16):
        super(KeyframeGuidanceAttention, self).__init__()
        self.feature_dim = feature_dim
        
        self.conv1 = nn.Conv2d(feature_dim * 2, feature_dim // reduction, 1)
        self.conv2 = nn.Conv2d(feature_dim // reduction, feature_dim, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, frame_features, keyframe_center):
        """
        Args:
            frame_features: (B, T, C, H, W) - features from all frames
            keyframe_center: (B, C, H, W) - keyframe feature center
        Returns:
            attended_features: (B, T, C, H, W) - attention-weighted features
        """
        B, T, C, H, W = frame_features.shape
        
        keyframe_center_expanded = keyframe_center.unsqueeze(1).expand(-1, T, -1, -1, -1)
        concatenated = torch.cat([frame_features, keyframe_center_expanded], dim=2)
        concatenated = concatenated.view(B * T, C * 2, H, W)
        
        attention = self.conv1(concatenated)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        attention = attention.view(B, T, C, H, W)
        
        attended_features = frame_features * attention
        return attended_features


class TemporalAggregation(nn.Module):
    """
    Temporal aggregation module to combine frame-level features
    """
    def __init__(self, feature_dim, aggregation_type='avg'):
        super(TemporalAggregation, self).__init__()
        self.aggregation_type = aggregation_type
        
        if aggregation_type == 'attention':
            self.attention_fc = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, 1)
            )
    
    def forward(self, features):
        """
        Args:
            features: (B, T, C, H, W) - frame features
        Returns:
            aggregated: (B, C, H, W) - temporally aggregated features
        """
        if self.aggregation_type == 'avg':
            aggregated = torch.mean(features, dim=1)
        elif self.aggregation_type == 'max':
            aggregated = torch.max(features, dim=1)[0]
        elif self.aggregation_type == 'attention':
            B, T, C, H, W = features.shape
            frame_vectors = F.adaptive_avg_pool2d(
                features.view(B * T, C, H, W), 1
            ).view(B, T, C)
            
            attention_weights = self.attention_fc(frame_vectors)
            attention_weights = F.softmax(attention_weights, dim=1)
            attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
            aggregated = torch.sum(features * attention_weights, dim=1)
        
        return aggregated


# ============================================================================
# LOSS FUNCTIONS - THREE OPTIONS
# ============================================================================

class CoherenceLoss(nn.Module):
    """
    Option 1: Coherence Loss from the original KGA-Net paper
    Ensures consistency between keyframe features and video frame features
    """
    def __init__(self):
        super(CoherenceLoss, self).__init__()
        
    def forward(self, frame_features, keyframe_center):
        """
        Args:
            frame_features: (B, T, C, H, W) - features from all frames
            keyframe_center: (B, C, H, W) - keyframe feature center
        Returns:
            loss: scalar - coherence loss value
        """
        B, T, C, H, W = frame_features.shape
        keyframe_center_expanded = keyframe_center.unsqueeze(1).expand(-1, T, -1, -1, -1)
        diff = frame_features - keyframe_center_expanded
        loss = torch.mean(diff ** 2)
        return loss


class TripletCoherenceLoss(nn.Module):
    """
    Option 2: Custom Triplet Loss variant for coherence
    Uses keyframe center as anchor, same-class frames as positive,
    different-class frames as negative
    """
    def __init__(self, margin=1.0):
        super(TripletCoherenceLoss, self).__init__()
        self.margin = margin
        
    def forward(self, frame_features, keyframe_center, labels, keyframe_labels):
        """
        Args:
            frame_features: (B, T, C, H, W) - features from all frames
            keyframe_center: (B, C, H, W) - keyframe feature center
            labels: (B, T) - labels for each frame in video
            keyframe_labels: (B,) - labels for keyframe center
        Returns:
            loss: scalar - triplet loss value
        """
        B, T, C, H, W = frame_features.shape
        
        keyframe_center_vec = F.adaptive_avg_pool2d(keyframe_center, 1).view(B, C)
        frame_features_vec = F.adaptive_avg_pool2d(
            frame_features.view(B * T, C, H, W), 1
        ).view(B, T, C)
        
        total_loss = 0.0
        count = 0
        
        for b in range(B):
            anchor = keyframe_center_vec[b]
            anchor_label = keyframe_labels[b]
            
            positive_mask = (labels[b] == anchor_label)
            negative_mask = (labels[b] != anchor_label)
            
            if positive_mask.sum() == 0 or negative_mask.sum() == 0:
                continue
            
            positives = frame_features_vec[b][positive_mask]
            negatives = frame_features_vec[b][negative_mask]
            
            pos_dist = torch.mean((positives - anchor.unsqueeze(0)) ** 2, dim=1)
            neg_dist = torch.mean((negatives - anchor.unsqueeze(0)) ** 2, dim=1)
            
            for pos_d in pos_dist:
                for neg_d in neg_dist:
                    loss = torch.clamp(pos_d - neg_d + self.margin, min=0.0)
                    total_loss += loss
                    count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        return total_loss


class StandardTripletLoss(nn.Module):
    """
    Option 3: Standard Triplet Loss for video frames
    Classic metric learning approach with triplet mining
    This is the standard triplet loss you'd find in most papers!
    """
    def __init__(self, margin=1.0, mining='hard'):
        super(StandardTripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining  # 'hard', 'semi-hard', or 'all'
        
    def forward(self, frame_features, labels):
        """
        Args:
            frame_features: (B, T, C, H, W) - features from frames
            labels: (B, T) - labels for each frame
        Returns:
            loss: scalar - triplet loss value
        """
        # Flatten to (N, C)
        if len(frame_features.shape) == 5:
            B, T, C, H, W = frame_features.shape
            frame_features = F.adaptive_avg_pool2d(
                frame_features.view(B * T, C, H, W), 1
            ).view(B * T, C)
            labels = labels.view(-1)
        
        N, C = frame_features.shape
        
        # Compute pairwise distance matrix
        dist_matrix = torch.cdist(frame_features, frame_features, p=2)
        
        # Create masks for positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # Remove diagonal (self-comparisons)
        mask_diag = torch.eye(N, dtype=torch.bool, device=frame_features.device)
        labels_equal = labels_equal & ~mask_diag
        
        total_loss = 0.0
        count = 0
        
        for i in range(N):
            pos_mask = labels_equal[i]
            neg_mask = labels_not_equal[i]
            
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            
            pos_dists = dist_matrix[i][pos_mask]
            neg_dists = dist_matrix[i][neg_mask]
            
            if self.mining == 'hard':
                # Hardest positive (furthest same-class) and hardest negative (closest different-class)
                hardest_pos = pos_dists.max()
                hardest_neg = neg_dists.min()
                loss = torch.clamp(hardest_pos - hardest_neg + self.margin, min=0.0)
                total_loss += loss
                count += 1
                
            elif self.mining == 'semi-hard':
                # Semi-hard negatives: d(a,n) > d(a,p) but within margin
                for pos_d in pos_dists:
                    semi_hard_negs = neg_dists[(neg_dists > pos_d) & (neg_dists < pos_d + self.margin)]
                    if len(semi_hard_negs) > 0:
                        for neg_d in semi_hard_negs:
                            loss = torch.clamp(pos_d - neg_d + self.margin, min=0.0)
                            total_loss += loss
                            count += 1
                            
            elif self.mining == 'all':
                # All possible triplets
                for pos_d in pos_dists:
                    for neg_d in neg_dists:
                        loss = torch.clamp(pos_d - neg_d + self.margin, min=0.0)
                        total_loss += loss
                        count += 1
        
        if count > 0:
            total_loss = total_loss / count
        else:
            total_loss = torch.tensor(0.0, device=frame_features.device)
        
        return total_loss


# ============================================================================
# KGA-NET MODEL
# ============================================================================

class KGANet(nn.Module):
    """
    KGA-Net: Keyframe Guidance Attention Network
    For Breast Ultrasound Video Classification
    """
    def __init__(self, num_classes=2, feature_dim=2048, reduction=16, 
                 aggregation_type='attention', pretrained=True):
        super(KGANet, self).__init__()
        
        resnet = resnet50(pretrained=pretrained)
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
            nn.Linear(512, num_classes)
        )
        
    def forward(self, video_frames, keyframe_indices=None, return_features=False):
        """
        Args:
            video_frames: (B, T, C, H, W) - video frames
            keyframe_indices: (B, N_key) - indices of keyframes (optional)
            return_features: bool - if True, return intermediate features
        Returns:
            logits: (B, num_classes) - classification logits
            features_dict: dict - intermediate features (if return_features=True)
        """
        B, T, C, H, W = video_frames.shape
        
        frames_flat = video_frames.view(B * T, C, H, W)
        features_flat = self.backbone(frames_flat)
        _, C_feat, H_feat, W_feat = features_flat.shape
        frame_features = features_flat.view(B, T, C_feat, H_feat, W_feat)
        
        if keyframe_indices is None:
            keyframe_features = frame_features[:, 0:1, :, :, :]
        else:
            keyframe_features = []
            for i in range(B):
                key_feats = frame_features[i, keyframe_indices[i], :, :, :]
                keyframe_features.append(key_feats)
            keyframe_features = torch.stack(keyframe_features, dim=0)
        
        keyframe_center = self.kfc(keyframe_features)
        attended_features = self.kga(frame_features, keyframe_center)
        aggregated_features = self.temporal_agg(attended_features)
        
        pooled = self.global_pool(aggregated_features).view(B, -1)
        logits = self.classifier(pooled)
        
        if return_features:
            features_dict = {
                'frame_features': frame_features,
                'keyframe_center': keyframe_center,
                'attended_features': attended_features,
                'aggregated_features': aggregated_features
            }
            return logits, features_dict
        
        return logits


# ============================================================================
# TRAINING FUNCTION WITH MULTIPLE LOSS OPTIONS
# ============================================================================

def train_step(model, video_frames, labels, keyframe_indices=None, 
               loss_type='coherence', alpha=0.5, triplet_mining='hard'):
    """
    Training step with different loss options
    
    Args:
        model: KGANet model
        video_frames: (B, T, 3, H, W) - input video frames
        labels: (B,) - video-level labels
        keyframe_indices: (B, N_key) - keyframe indices
        loss_type: str - 'coherence', 'triplet_coherence', or 'triplet_standard'
        alpha: float - weight for auxiliary loss
        triplet_mining: str - 'hard', 'semi-hard', or 'all'
    
    Returns:
        total_loss: combined loss
        cls_loss: classification loss
        aux_loss: auxiliary loss (coherence or triplet)
    """
    logits, features = model(video_frames, keyframe_indices, return_features=True)
    
    # Classification loss
    cls_criterion = nn.CrossEntropyLoss()
    cls_loss = cls_criterion(logits, labels)
    
    # Auxiliary loss
    B, T = video_frames.shape[0], video_frames.shape[1]
    
    if loss_type == 'coherence':
        # Option 1: Original coherence loss
        aux_criterion = CoherenceLoss()
        aux_loss = aux_criterion(
            features['frame_features'],
            features['keyframe_center']
        )
        
    elif loss_type == 'triplet_coherence':
        # Option 2: Keyframe-centered triplet loss
        frame_labels = labels.unsqueeze(1).expand(-1, T)
        keyframe_labels = labels
        
        aux_criterion = TripletCoherenceLoss(margin=1.0)
        aux_loss = aux_criterion(
            features['frame_features'],
            features['keyframe_center'],
            frame_labels,
            keyframe_labels
        )
        
    elif loss_type == 'triplet_standard':
        # Option 3: Standard triplet loss (RECOMMENDED FOR YOUR EXPERIMENT)
        frame_labels = labels.unsqueeze(1).expand(-1, T)
        
        aux_criterion = StandardTripletLoss(margin=1.0, mining=triplet_mining)
        aux_loss = aux_criterion(
            features['frame_features'],
            frame_labels
        )
        
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    total_loss = cls_loss + alpha * aux_loss
    return total_loss, cls_loss, aux_loss


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    batch_size = 4
    num_frames = 16
    num_classes = 2
    img_size = 224
    
    model = KGANet(num_classes=num_classes, aggregation_type='attention')
    
    video_input = torch.randn(batch_size, num_frames, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    keyframe_indices = torch.tensor([[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8, 13]])
    
    print("=" * 70)
    print("OPTION 1: Coherence Loss (Original Paper)")
    print("=" * 70)
    total_loss, cls_loss, aux_loss = train_step(
        model, video_input, labels, keyframe_indices, 
        loss_type='coherence', alpha=0.5
    )
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Classification Loss: {cls_loss.item():.4f}")
    print(f"Coherence Loss: {aux_loss.item():.4f}")
    
    print("\n" + "=" * 70)
    print("OPTION 2: Triplet Coherence Loss (Keyframe-centered)")
    print("=" * 70)
    total_loss, cls_loss, aux_loss = train_step(
        model, video_input, labels, keyframe_indices, 
        loss_type='triplet_coherence', alpha=0.5
    )
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Classification Loss: {cls_loss.item():.4f}")
    print(f"Triplet Coherence Loss: {aux_loss.item():.4f}")
    
    print("\n" + "=" * 70)
    print("OPTION 3: Standard Triplet Loss (RECOMMENDED - Hard Mining)")
    print("=" * 70)
    total_loss, cls_loss, aux_loss = train_step(
        model, video_input, labels, keyframe_indices, 
        loss_type='triplet_standard',  # <-- USE THIS FOR YOUR EXPERIMENT
        alpha=0.5,
        triplet_mining='hard'
    )
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Classification Loss: {cls_loss.item():.4f}")
    print(f"Standard Triplet Loss (Hard): {aux_loss.item():.4f}")
    
    print("\n" + "=" * 70)
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 70)