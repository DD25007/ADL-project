import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
import pandas as pd
from tqdm import tqdm
import os

# Set style for better-looking plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["legend.fontsize"] = 12

# ============================================================================
# MODEL EVALUATION FUNCTIONS
# ============================================================================


def evaluate_model_detailed(
    model, dataloader, device, model_name="Model", is_video=True
):
    """
    Comprehensive evaluation of a model.
    Returns predictions, labels, probabilities, and features for analysis.

    Args:
        model: The model to evaluate (KGANet or ImageClassificationNetwork)
        dataloader: DataLoader for the dataset
        device: Device to run evaluation on
        model_name: Name of the model for logging
        is_video: True if video model, False if image model
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating {model_name}")
        for batch_data in pbar:
            if is_video:
                # Video model: (frames, labels, keyframe_indices)
                frames, labels, keyframe_indices = batch_data
                frames = frames.to(device)
                keyframe_indices = keyframe_indices.to(device)

                # Get predictions and features
                logits, features = model(frames, keyframe_indices, return_features=True)

                # Extract aggregated features for visualization
                pooled_features = nn.AdaptiveAvgPool2d(1)(
                    features["aggregated_features"]
                )
                pooled_features = pooled_features.view(pooled_features.size(0), -1)

            else:
                # Image model: (images, labels)
                images, labels = batch_data
                images = images.to(device)

                # Get predictions and features
                logits, img_features = model(images, return_features=True)

                # Features are already 1D vectors for image model
                pooled_features = img_features

            labels = labels.to(device)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.append(pooled_features.cpu().numpy())

    # Stack features properly
    all_features = np.vstack(all_features)

    return {
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
        "probabilities": np.array(all_probs),
        "features": all_features,
        "accuracy": np.mean(np.array(all_preds) == np.array(all_labels)) * 100,
    }


def load_video_model_and_evaluate(checkpoint_path, dataloader, device, model_name):
    """Load a video model from checkpoint and evaluate it."""
    from kaustav_kga import KGANet
    from torchvision.models import resnet50, ResNet50_Weights

    # Create shared backbone
    resnet = resnet50(weights=None)
    shared_backbone = nn.Sequential(*list(resnet.children())[:-2]).to(device)

    # Create video model
    model = KGANet(
        num_classes=2,
        feature_dim=2048,
        reduction=16,
        aggregation_type="attention",
        pretrained=False,
        backbone=shared_backbone,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load backbone state
    if "backbone_state" in checkpoint:
        shared_backbone.load_state_dict(checkpoint["backbone_state"])

    # Load video head state
    if "video_head" in checkpoint:
        model.load_state_dict(checkpoint["video_head"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate
    results = evaluate_model_detailed(
        model, dataloader, device, model_name, is_video=True
    )
    results["checkpoint_info"] = {
        "epoch": checkpoint.get("epoch", "N/A"),
        "val_acc": checkpoint.get("val_acc", "N/A"),
    }

    return results, model


def load_image_model_and_evaluate(
    checkpoint_path, dataloader, device, model_name, feature_dim=512
):
    """Load an image model from checkpoint and evaluate it."""
    from kaustav_kga import ImageClassificationNetwork
    from torchvision.models import resnet50

    # Create shared backbone
    resnet = resnet50(weights=None)
    shared_backbone = nn.Sequential(*list(resnet.children())[:-2]).to(device)

    # Create image model
    model = ImageClassificationNetwork(
        num_classes=2,
        feature_dim=feature_dim,
        pretrained=False,
        backbone=shared_backbone,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load backbone state
    if "backbone_state" in checkpoint:
        shared_backbone.load_state_dict(checkpoint["backbone_state"])

    # Load image head state
    if "image_head" in checkpoint:
        model.load_state_dict(checkpoint["image_head"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate
    results = evaluate_model_detailed(
        model, dataloader, device, model_name, is_video=False
    )
    results["checkpoint_info"] = {
        "epoch": checkpoint.get("epoch", "N/A"),
        "val_acc": checkpoint.get("val_acc", "N/A"),
    }

    return results, model


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def parse_training_log(log_file="training.log"):
    """Parse training log to extract metrics."""
    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found. Skipping training curves.")
        return None

    data = {
        "coherence": {"epochs": [], "train_loss": [], "val_acc": []},
        "triplet_coherence": {"epochs": [], "train_loss": [], "val_acc": []},
        "triplet_standard_hard": {"epochs": [], "train_loss": [], "val_acc": []},
        "triplet_standard_semi-hard": {"epochs": [], "train_loss": [], "val_acc": []},
        "triplet_standard_all": {"epochs": [], "train_loss": [], "val_acc": []},
    }

    current_loss_type = None

    with open(log_file, "r") as f:
        for line in f:
            # Detect loss type
            if "loss type: coherence" in line.lower():
                current_loss_type = "coherence"
            elif "loss type: triplet_coherence" in line.lower():
                current_loss_type = "triplet_coherence"
            elif "loss type: triplet_standard" in line.lower():
                if "mining: hard" in line.lower():
                    current_loss_type = "triplet_standard_hard"
                elif "mining: semi-hard" in line.lower():
                    current_loss_type = "triplet_standard_semi-hard"
                elif "mining: all" in line.lower():
                    current_loss_type = "triplet_standard_all"

            # Extract epoch number
            if "Epoch" in line and "/" in line and current_loss_type:
                try:
                    epoch_str = line.split("Epoch")[1].split("/")[0].strip()
                    epoch = int(epoch_str)
                except:
                    continue

            # Extract training loss
            if "Train - Video Loss:" in line and current_loss_type:
                try:
                    loss_val = float(line.split("Loss:")[1].split(",")[0].strip())
                    data[current_loss_type]["train_loss"].append(loss_val)
                    data[current_loss_type]["epochs"].append(epoch)
                except:
                    continue

            # Extract validation accuracy
            if "Val   - Video Loss:" in line and current_loss_type:
                try:
                    acc_val = float(line.split("Acc:")[1].replace("%", "").strip())
                    data[current_loss_type]["val_acc"].append(acc_val)
                except:
                    continue

    return data


def plot_training_curves(
    log_file="training.log", save_path="plots/training_curves.png"
):
    """Parse training log and plot training curves."""
    data = parse_training_log(log_file)

    if data is None:
        print("Skipping training curves plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    models_map = {
        "coherence": "Coherence Loss",
        "triplet_coherence": "Triplet-Coherence",
        "triplet_standard_hard": "Hard Mining",
        "triplet_standard_semi-hard": "Semi-Hard Mining",
        "triplet_standard_all": "All Mining",
    }
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    # Plot 1: Training Loss
    ax = axes[0]
    for idx, (key, name) in enumerate(models_map.items()):
        if data[key]["epochs"]:
            ax.plot(
                data[key]["epochs"],
                data[key]["train_loss"],
                label=name,
                linewidth=2,
                color=colors[idx],
                marker="o",
                markersize=3,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation Accuracy
    ax = axes[1]
    for idx, (key, name) in enumerate(models_map.items()):
        if data[key]["epochs"]:
            ax.plot(
                data[key]["epochs"],
                data[key]["val_acc"],
                label=name,
                linewidth=2,
                color=colors[idx],
                marker="o",
                markersize=3,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Validation Accuracy Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved training curves to {save_path}")
    plt.close()


def plot_confusion_matrices(
    results_dict,
    class_names=["Benign", "Malignant"],
    save_path="plots/confusion_matrices.png",
):
    """Plot confusion matrices for all models."""
    n_models = len(results_dict)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols

    # Always create at least 1 row and 1 col, ensure we get iterable axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows), squeeze=False)

    # Now axes is always 2D, flatten it
    axes = axes.flatten()

    for idx, (model_name, results) in enumerate(results_dict.items()):
        cm = confusion_matrix(results["labels"], results["predictions"])

        # Normalize
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        ax = axes[idx]
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Percentage"},
            ax=ax,
        )

        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.2f}%')

        # Add counts in parentheses
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(
                    j + 0.5,
                    i + 0.7,
                    f"({cm[i, j]})",
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=9,
                )

    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved confusion matrices to {save_path}")
    plt.close()


def plot_roc_curves(results_dict, save_path="plots/roc_curves.png"):
    """Plot ROC curves for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#e67e22"]

    # Plot 1: All ROC curves together
    for idx, (model_name, results) in enumerate(results_dict.items()):
        probs = results["probabilities"][:, 1]
        fpr, tpr, _ = roc_curve(results["labels"], probs)
        roc_auc = auc(fpr, tpr)

        ax1.plot(
            fpr,
            tpr,
            linewidth=2,
            color=colors[idx % len(colors)],
            label=f"{model_name} (AUC = {roc_auc:.3f})",
        )

    ax1.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curves - All Models")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: AUC Bar Chart
    model_names = list(results_dict.keys())
    aucs = []
    for results in results_dict.values():
        probs = results["probabilities"][:, 1]
        fpr, tpr, _ = roc_curve(results["labels"], probs)
        aucs.append(auc(fpr, tpr))

    bars = ax2.bar(
        range(len(model_names)), aucs, color=colors[: len(model_names)], alpha=0.8
    )
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha="right")
    ax2.set_ylabel("AUC Score")
    ax2.set_title("AUC Comparison")
    ax2.set_ylim([0.5, 1.0])
    ax2.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, auc_val in zip(bars, aucs):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{auc_val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved ROC curves to {save_path}")
    plt.close()


def plot_performance_comparison(
    results_dict, save_path="plots/performance_comparison.png"
):
    """Comprehensive performance comparison across all metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    model_names = list(results_dict.keys())
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#e67e22"]

    # Collect metrics
    from sklearn.metrics import precision_recall_fscore_support

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for results in results_dict.values():
        accuracies.append(results["accuracy"])
        precision, recall, f1, _ = precision_recall_fscore_support(
            results["labels"], results["predictions"], average="weighted"
        )
        precisions.append(precision * 100)
        recalls.append(recall * 100)
        f1_scores.append(f1 * 100)

    x_pos = np.arange(len(model_names))

    # Plot 1: Accuracy
    ax = axes[0, 0]
    bars = ax.bar(x_pos, accuracies, color=colors[: len(model_names)], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Comparison")
    ax.set_ylim([0, 100])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    # Plot 2: Precision
    ax = axes[0, 1]
    bars = ax.bar(x_pos, precisions, color=colors[: len(model_names)], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision Comparison")
    ax.set_ylim([0, 100])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, prec in zip(bars, precisions):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{prec:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    # Plot 3: Recall
    ax = axes[1, 0]
    bars = ax.bar(x_pos, recalls, color=colors[: len(model_names)], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Recall Comparison")
    ax.set_ylim([0, 100])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, rec in zip(bars, recalls):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{rec:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    # Plot 4: F1-Score
    ax = axes[1, 1]
    bars = ax.bar(x_pos, f1_scores, color=colors[: len(model_names)], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel("F1-Score (%)")
    ax.set_title("F1-Score Comparison")
    ax.set_ylim([0, 100])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{f1:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved performance comparison to {save_path}")
    plt.close()


def plot_feature_tsne(results_dict, save_path="plots/tsne_visualization.png"):
    """Plot t-SNE visualization of learned features."""
    n_models = len(results_dict)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (model_name, results) in enumerate(results_dict.items()):
        # Perform t-SNE
        print(f"Computing t-SNE for {model_name}...")
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, len(results["features"]) - 1),
        )
        features_2d = tsne.fit_transform(results["features"])

        ax = axes[idx]

        # Plot each class with different color
        for class_idx, class_name in enumerate(["Benign", "Malignant"]):
            mask = results["labels"] == class_idx
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                label=class_name,
                alpha=0.6,
                s=50,
            )

        ax.set_title(f"{model_name}\nFeature Space Visualization")
        ax.legend()
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved t-SNE visualization to {save_path}")
    plt.close()


def plot_per_class_performance(
    results_dict, save_path="plots/per_class_performance.png"
):
    """Plot per-class performance metrics."""
    from sklearn.metrics import precision_recall_fscore_support

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    class_names = ["Benign", "Malignant"]
    model_names = list(results_dict.keys())
    x_pos = np.arange(len(model_names))

    # Collect per-class metrics
    benign_precision = []
    benign_recall = []
    malignant_precision = []
    malignant_recall = []

    for results in results_dict.values():
        precision, recall, _, _ = precision_recall_fscore_support(
            results["labels"], results["predictions"], average=None
        )
        benign_precision.append(precision[0] * 100)
        benign_recall.append(recall[0] * 100)
        malignant_precision.append(precision[1] * 100)
        malignant_recall.append(recall[1] * 100)

    # Plot 1: Benign Precision
    ax = axes[0, 0]
    bars = ax.bar(x_pos, benign_precision, color="skyblue", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Benign Class - Precision")
    ax.set_ylim([0, 100])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, benign_precision):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 2: Benign Recall
    ax = axes[0, 1]
    bars = ax.bar(x_pos, benign_recall, color="lightcoral", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Benign Class - Recall")
    ax.set_ylim([0, 100])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, benign_recall):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 3: Malignant Precision
    ax = axes[1, 0]
    bars = ax.bar(x_pos, malignant_precision, color="lightgreen", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Malignant Class - Precision")
    ax.set_ylim([0, 100])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, malignant_precision):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 4: Malignant Recall
    ax = axes[1, 1]
    bars = ax.bar(x_pos, malignant_recall, color="plum", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Malignant Class - Recall")
    ax.set_ylim([0, 100])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, malignant_recall):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved per-class performance to {save_path}")
    plt.close()


def plot_radar_chart(results_dict, save_path="plots/radar_comparison.png"):
    """Create radar chart comparing all models across multiple metrics."""
    from math import pi
    from sklearn.metrics import precision_recall_fscore_support

    # Calculate metrics for each model
    metrics_data = {}
    for model_name, results in results_dict.items():
        precision, recall, f1, _ = precision_recall_fscore_support(
            results["labels"], results["predictions"], average="weighted"
        )

        # Get AUC
        probs = results["probabilities"][:, 1]
        fpr, tpr, _ = roc_curve(results["labels"], probs)
        roc_auc = auc(fpr, tpr)

        # Get specificity
        cm = confusion_matrix(results["labels"], results["predictions"])
        specificity = np.mean(
            [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0 for i in range(len(cm))]
        )

        metrics_data[model_name] = {
            "Accuracy": results["accuracy"],
            "Precision": precision * 100,
            "Recall": recall * 100,
            "F1-Score": f1 * 100,
            "AUC": roc_auc * 100,
            "Specificity": specificity * 100,
        }

    # Setup radar chart
    categories = list(list(metrics_data.values())[0].keys())
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#e67e22"]

    for idx, (model_name, metrics) in enumerate(metrics_data.items()):
        values = list(metrics.values())
        values += values[:1]

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=model_name,
            color=colors[idx % len(colors)],
        )
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
    ax.grid(True)

    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Performance Radar Chart", size=16, pad=20)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved radar chart to {save_path}")
    plt.close()


def generate_summary_table(results_dict, save_path="plots/summary_table.png"):
    """Generate a comprehensive summary table."""
    from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score

    summary_data = []

    for model_name, results in results_dict.items():
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            results["labels"], results["predictions"], average="weighted"
        )

        # AUC
        probs = results["probabilities"][:, 1]
        fpr, tpr, _ = roc_curve(results["labels"], probs)
        roc_auc = auc(fpr, tpr)

        # Cohen's Kappa
        kappa = cohen_kappa_score(results["labels"], results["predictions"])

        summary_data.append(
            {
                "Model": model_name,
                "Accuracy (%)": f"{results['accuracy']:.2f}",
                "Precision (%)": f"{precision*100:.2f}",
                "Recall (%)": f"{recall*100:.2f}",
                "F1-Score (%)": f"{f1*100:.2f}",
                "AUC": f"{roc_auc:.3f}",
                "Cohen's κ": f"{kappa:.3f}",
            }
        )

    df = pd.DataFrame(summary_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(summary_data) * 0.8 + 1))
    ax.axis("tight")
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colColours=["lightgray"] * len(df.columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Highlight best values
    for col_idx, col in enumerate(df.columns[1:], 1):
        values = [float(row.replace("%", "")) for row in df.iloc[:, col_idx]]
        best_idx = np.argmax(values)
        table[(best_idx + 1, col_idx)].set_facecolor("#90EE90")

    plt.title("Model Performance Summary", fontsize=16, fontweight="bold", pad=20)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved summary table to {save_path}")
    plt.close()

    # Also save as CSV
    csv_path = save_path.replace(".png", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved summary CSV to {csv_path}")


def plot_image_vs_video_comparison(
    image_results, video_results_dict, save_path="plots/image_vs_video_comparison.png"
):
    """Compare image model performance with video models."""
    from sklearn.metrics import precision_recall_fscore_support

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Prepare data
    all_models = ["Image Model"] + list(video_results_dict.keys())
    all_results = [image_results] + list(video_results_dict.values())

    accuracies = [r["accuracy"] for r in all_results]

    precisions = []
    recalls = []
    f1_scores = []

    for r in all_results:
        p, rec, f1, _ = precision_recall_fscore_support(
            r["labels"], r["predictions"], average="weighted"
        )
        precisions.append(p * 100)
        recalls.append(rec * 100)
        f1_scores.append(f1 * 100)

    colors = ["#FF6B6B"] + ["#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DFE6E9"]
    x_pos = np.arange(len(all_models))

    # Plot 1: Accuracy
    ax = axes[0, 0]
    bars = ax.bar(x_pos, accuracies, color=colors[: len(all_models)], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_models, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy: Image vs Video Models")
    ax.set_ylim([0, 100])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 2: Precision
    ax = axes[0, 1]
    bars = ax.bar(x_pos, precisions, color=colors[: len(all_models)], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_models, rotation=45, ha="right")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision: Image vs Video Models")
    ax.set_ylim([0, 100])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, prec in zip(bars, precisions):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{prec:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 3: Recall
    ax = axes[1, 0]
    bars = ax.bar(x_pos, recalls, color=colors[: len(all_models)], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_models, rotation=45, ha="right")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Recall: Image vs Video Models")
    ax.set_ylim([0, 100])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, rec in zip(bars, recalls):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{rec:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 4: F1-Score
    ax = axes[1, 1]
    bars = ax.bar(x_pos, f1_scores, color=colors[: len(all_models)], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_models, rotation=45, ha="right")
    ax.set_ylabel("F1-Score (%)")
    ax.set_title("F1-Score: Image vs Video Models")
    ax.set_ylim([0, 100])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, f1 in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{f1:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved image vs video comparison to {save_path}")
    plt.close()


def analyze_models(results_dict, image_results=None):
    """Provide detailed textual analysis of model performance."""
    from sklearn.metrics import precision_recall_fscore_support

    print("\n1. OVERALL PERFORMANCE RANKING")
    print("-" * 70)

    # Rank by accuracy
    accuracies = [(name, res["accuracy"]) for name, res in results_dict.items()]
    if image_results:
        accuracies.append(("Image Model", image_results["accuracy"]))
    accuracies.sort(key=lambda x: x[1], reverse=True)

    for rank, (name, acc) in enumerate(accuracies, 1):
        print(f"   {rank}. {name}: {acc:.2f}%")

    print("\n2. BEST MODEL FOR EACH METRIC")
    print("-" * 70)

    # Include image model in comparisons if available
    all_results = dict(results_dict)
    if image_results:
        all_results["Image Model"] = image_results

    # Accuracy
    best_acc_model = max(all_results.items(), key=lambda x: x[1]["accuracy"])
    print(
        f"   Best Accuracy: {best_acc_model[0]} ({best_acc_model[1]['accuracy']:.2f}%)"
    )

    # AUC
    best_auc = None
    best_auc_score = 0
    for name, results in all_results.items():
        probs = results["probabilities"][:, 1]
        fpr, tpr, _ = roc_curve(results["labels"], probs)
        auc_score = auc(fpr, tpr)
        if auc_score > best_auc_score:
            best_auc_score = auc_score
            best_auc = name
    print(f"   Best AUC: {best_auc} ({best_auc_score:.3f})")

    # F1-Score
    best_f1 = None
    best_f1_score = 0
    for name, results in all_results.items():
        _, _, f1, _ = precision_recall_fscore_support(
            results["labels"], results["predictions"], average="weighted"
        )
        if f1 > best_f1_score:
            best_f1_score = f1
            best_f1 = name
    print(f"   Best F1-Score: {best_f1} ({best_f1_score*100:.2f}%)")

    print("\n3. CLASS-WISE PERFORMANCE ANALYSIS")
    print("-" * 70)

    for class_idx, class_name in enumerate(["Benign", "Malignant"]):
        print(f"\n   {class_name} Class:")
        best_precision = (None, 0)
        best_recall = (None, 0)

        for name, results in all_results.items():
            precision, recall, _, _ = precision_recall_fscore_support(
                results["labels"], results["predictions"], average=None
            )

            if len(precision) > class_idx and precision[class_idx] > best_precision[1]:
                best_precision = (name, precision[class_idx])
            if len(recall) > class_idx and recall[class_idx] > best_recall[1]:
                best_recall = (name, recall[class_idx])

        print(
            f"      Best Precision: {best_precision[0]} ({best_precision[1]*100:.2f}%)"
        )
        print(f"      Best Recall: {best_recall[0]} ({best_recall[1]*100:.2f}%)")

    print("\n4. KEY INSIGHTS & RECOMMENDATIONS")
    print("-" * 70)

    # Insight 1: Image vs Video
    if image_results:
        print("\n   Image vs Video Models:")
        img_acc = image_results["accuracy"]
        avg_video_acc = np.mean([r["accuracy"] for r in results_dict.values()])
        print(f"      • Image Model Accuracy: {img_acc:.2f}%")
        print(f"      • Average Video Model Accuracy: {avg_video_acc:.2f}%")
        if img_acc > avg_video_acc:
            print(
                f"      → Image model outperforms video models by {img_acc - avg_video_acc:.2f}%"
            )
            print(
                f"        (Static images may contain sufficient diagnostic information)"
            )
        else:
            print(
                f"      → Video models outperform image model by {avg_video_acc - img_acc:.2f}%"
            )
            print(
                f"        (Temporal information provides additional diagnostic value)"
            )

    # Insight 2: Loss function comparison (video models only)
    print("\n   Video Loss Function Impact:")
    coherence_models = [
        k for k in results_dict.keys() if "Coherence" in k or "coherence" in k.lower()
    ]
    triplet_models = [
        k
        for k in results_dict.keys()
        if "Triplet" in k or "triplet" in k.lower() or "Mining" in k
    ]

    if coherence_models and triplet_models:
        avg_coherence = np.mean([results_dict[k]["accuracy"] for k in coherence_models])
        avg_triplet = np.mean([results_dict[k]["accuracy"] for k in triplet_models])

        if avg_triplet > avg_coherence:
            print(f"      • Triplet-based losses (avg {avg_triplet:.2f}%) outperform")
            print(
                f"        coherence losses (avg {avg_coherence:.2f}%) by {avg_triplet-avg_coherence:.2f}%"
            )
        else:
            print(f"      • Coherence losses (avg {avg_coherence:.2f}%) outperform")
            print(
                f"        triplet-based losses (avg {avg_triplet:.2f}%) by {avg_coherence-avg_triplet:.2f}%"
            )

    # Insight 3: Mining strategy comparison
    print("\n   Mining Strategy Impact:")
    mining_strategies = ["Hard", "Semi-Hard", "All"]
    mining_results = []

    for strategy in mining_strategies:
        matching_models = [k for k in results_dict.keys() if strategy in k]
        if matching_models:
            avg_acc = np.mean([results_dict[k]["accuracy"] for k in matching_models])
            mining_results.append((strategy, avg_acc))
            print(f"      • {strategy} Mining: {avg_acc:.2f}%")

    if mining_results:
        best_mining = max(mining_results, key=lambda x: x[1])
        print(f"      → Best mining strategy: {best_mining[0]} ({best_mining[1]:.2f}%)")

    print("\n5. RECOMMENDED MODEL")
    print("-" * 70)

    # Recommendation based on accuracy and F1
    scores = {}
    for name, results in all_results.items():
        _, _, f1, _ = precision_recall_fscore_support(
            results["labels"], results["predictions"], average="weighted"
        )
        # Combined score: 70% accuracy, 30% F1
        combined_score = 0.7 * results["accuracy"] + 0.3 * (f1 * 100)
        scores[name] = combined_score

    recommended = max(scores.items(), key=lambda x: x[1])
    print(f"\n   Recommended Model: {recommended[0]}")
    print(f"   Combined Score: {recommended[1]:.2f}")
    print(f"   Accuracy: {all_results[recommended[0]]['accuracy']:.2f}%")

    _, _, f1, _ = precision_recall_fscore_support(
        all_results[recommended[0]]["labels"],
        all_results[recommended[0]]["predictions"],
        average="weighted",
    )
    print(f"   F1-Score: {f1*100:.2f}%")

    print("\n   Rationale:")
    if "Image" in recommended[0]:
        print(f"      • Simpler architecture, faster inference")
        print(f"      • Excellent performance on static frames")
        print(f"      • Ideal for real-time clinical deployment")
    else:
        print(f"      • Leverages temporal information from video sequences")
        print(f"      • Balanced performance across metrics")
        print(f"      • Suitable for comprehensive ultrasound analysis")


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================


def run_complete_analysis(
    checkpoint_dir,
    video_val_loader,
    image_val_loader=None,
    device="cuda",
    output_dir="plots",
    include_image_model=True,
):
    """
    Run complete analysis and generate all visualizations.

    Args:
        checkpoint_dir: Directory containing model checkpoints
        video_val_loader: Validation DataLoader for video data
        image_val_loader: Validation DataLoader for image data (optional)
        device: Device to run on
        output_dir: Directory to save plots
        include_image_model: Whether to include image model analysis
    """

    print("=" * 70)
    print("KGA-Net Joint Model Comparison & Analysis")
    print("=" * 70)

    # Define video model configurations
    video_configs = {
        "Coherence Loss": "best_video_model.pth",  # or use epoch-specific
        "Triplet-Coherence": "video_model_triplet_coherence_epoch_50.pth",
        "Hard Mining": "video_model_triplet_standard_hard_epoch_50.pth",
        "Semi-Hard Mining": "video_model_triplet_standard_semi-hard_epoch_50.pth",
        "All Mining": "video_model_triplet_standard_all_epoch_50.pth",
    }

    # Load and evaluate video models
    video_results_dict = {}
    for model_name, checkpoint_file in video_configs.items():
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            continue

        print(f"\nEvaluating Video Model: {model_name}...")
        results, model = load_video_model_and_evaluate(
            checkpoint_path, video_val_loader, device, model_name
        )
        video_results_dict[model_name] = results

        print(f"  Accuracy: {results['accuracy']:.2f}%")
        print(
            f"  Checkpoint Info: Epoch {results['checkpoint_info']['epoch']}, "
            f"Val Acc: {results['checkpoint_info']['val_acc']:.2f}%"
        )

    # Load and evaluate image model
    image_results = None
    if include_image_model and image_val_loader is not None:
        image_checkpoint = os.path.join(checkpoint_dir, "best_image_model.pth")

        if os.path.exists(image_checkpoint):
            print(f"\nEvaluating Image Model...")
            image_results, _ = load_image_model_and_evaluate(
                image_checkpoint, image_val_loader, device, "Image Model"
            )
            print(f"  Accuracy: {image_results['accuracy']:.2f}%")
        else:
            print(f"Warning: Image model checkpoint not found: {image_checkpoint}")

    if len(video_results_dict) == 0:
        print("\nError: No video models were successfully loaded!")
        return

    print(f"\n{'='*70}")
    print(
        f"Successfully loaded {len(video_results_dict)} video models"
        + (f" and 1 image model" if image_results else "")
    )
    print(f"{'='*70}\n")

    # Generate all visualizations
    os.makedirs(output_dir, exist_ok=True)

    print("Generating visualizations...")
    print("-" * 70)

    # 1. Training curves
    plot_training_curves(save_path=f"{output_dir}/1_training_curves.png")

    # 2. Confusion matrices (video models)
    plot_confusion_matrices(
        video_results_dict, save_path=f"{output_dir}/2_video_confusion_matrices.png"
    )

    # 3. ROC curves (video models)
    plot_roc_curves(
        video_results_dict, save_path=f"{output_dir}/3_video_roc_curves.png"
    )

    # 4. Performance comparison (video models)
    plot_performance_comparison(
        video_results_dict, save_path=f"{output_dir}/4_video_performance.png"
    )

    # 5. Per-class performance (video models)
    plot_per_class_performance(
        video_results_dict, save_path=f"{output_dir}/5_video_per_class.png"
    )

    # 6. t-SNE visualization (video models)
    plot_feature_tsne(video_results_dict, save_path=f"{output_dir}/6_video_tsne.png")

    # 7. Radar chart (video models)
    plot_radar_chart(video_results_dict, save_path=f"{output_dir}/7_video_radar.png")

    # 8. Summary table (video models)
    generate_summary_table(
        video_results_dict, save_path=f"{output_dir}/8_video_summary.png"
    )

    # 9. Image vs Video comparison (if image model available)
    if image_results:
        plot_image_vs_video_comparison(
            image_results,
            video_results_dict,
            save_path=f"{output_dir}/9_image_vs_video.png",
        )

        # Combined confusion matrix for image model
        plot_confusion_matrices(
            {"Image Model": image_results},
            save_path=f"{output_dir}/10_image_confusion.png",
        )

    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"{'='*70}")
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. training_curves.png - Training/validation curves over epochs")
    print("  2. video_confusion_matrices.png - Confusion matrices for video models")
    print("  3. video_roc_curves.png - ROC curves and AUC comparison")
    print("  4. video_performance.png - Accuracy, Precision, Recall, F1")
    print("  5. video_per_class.png - Per-class metrics breakdown")
    print("  6. video_tsne.png - Feature space visualization")
    print("  7. video_radar.png - Multi-metric radar chart")
    print("  8. video_summary.png & .csv - Comprehensive metrics table")
    if image_results:
        print("  9. image_vs_video.png - Image vs video model comparison")
        print(" 10. image_confusion.png - Image model confusion matrix")

    # Print detailed analysis
    print(f"\n{'='*70}")
    print("DETAILED ANALYSIS & INSIGHTS")
    print(f"{'='*70}\n")

    analyze_models(video_results_dict, image_results)

    return video_results_dict, image_results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """Example usage - adjust paths according to your setup"""

    # Configuration
    CHECKPOINT_DIR = "notebooks/joint_checkpoints"
    VIDEO_DATA_ROOT = "./data"
    IMAGE_DATA_ROOT = "./data/busi/"
    IMAGE_ANNOTATION = "data/busi_bboxes.json"
    VIDEO_VAL_ANNOTATION = "imagenet_vid_val.json"
    BATCH_SIZE = 8
    NUM_FRAMES = 128
    OUTPUT_DIR = "analysis_plots"

    # Import dataset loaders
    from kaustav_kga import video_dataloader, BUSIImageDataset
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    print("\nPreparing for analysis...")
    print("=" * 70)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load video validation data
    print("\nLoading video validation dataset...")
    _, video_val_loader = video_dataloader(
        root_dir=VIDEO_DATA_ROOT,
        val_annotation=VIDEO_VAL_ANNOTATION,
        batch_size=BATCH_SIZE,
        num_frames=NUM_FRAMES,
        num_workers=4,
    )
    print(f"Video validation samples: {len(video_val_loader.dataset)}")

    # Load image validation data
    print("\nLoading image validation dataset...")
    val_img_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_dataset = BUSIImageDataset(
        IMAGE_DATA_ROOT, IMAGE_ANNOTATION, val_img_transform
    )
    # Use a subset for validation (last 20%)
    val_size = int(len(image_dataset) * 0.2)
    train_size = len(image_dataset) - val_size
    _, image_val_dataset = torch.utils.data.random_split(
        image_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    image_val_loader = DataLoader(
        image_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Image validation samples: {len(image_val_dataset)}")

    # Run complete analysis
    video_results, image_results = run_complete_analysis(
        checkpoint_dir=CHECKPOINT_DIR,
        video_val_loader=video_val_loader,
        image_val_loader=image_val_loader,
        device=device,
        output_dir=OUTPUT_DIR,
        include_image_model=True,
    )

    print("\n✅ All analysis completed!")
    print(f"Check the '{OUTPUT_DIR}' directory for all visualizations.")
    print("\nThese plots are ready for your PPT presentation!")
    print("=" * 70)
