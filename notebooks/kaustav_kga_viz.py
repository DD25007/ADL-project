import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)
import pandas as pd
import json
from datetime import datetime

# Import your model and dataset
from kaustav_kga import KGANet, create_dataloaders


class LossComparisonAnalyzer:
    """Comprehensive comparison of different loss functions."""

    def __init__(self, save_dir="loss_comparison_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Define loss function names and their checkpoint paths
        self.loss_configs = {
            "Coherence": {
                "path": "notebooks/checkpoints/best_model_coherence.pth",
                "color": "#1f77b4",
                "description": "Basic MSE-based coherence loss",
            },
            "Triplet Coherence": {
                "path": "notebooks/checkpoints/best_model_triplet_coherence.pth",
                "color": "#ff7f0e",
                "description": "Keyframe-guided triplet loss",
            },
            "Standard Triplet": {
                "path": "notebooks/checkpoints/best_model_triplet_standard.pth",
                "color": "#2ca02c",
                "description": "Standard triplet loss with hard mining",
            },
        }

        self.class_names = ["Benign", "Malignant"]
        self.results = {}

    def evaluate_model(self, model, val_loader, device, loss_name):
        """Evaluate a single model comprehensively."""
        print(f"\n{'='*70}")
        print(f"Evaluating: {loss_name}")
        print(f"{'='*70}")

        model.eval()

        # Metrics storage
        all_predictions = []
        all_labels = []
        all_probs = []
        all_logits = []

        # Per-sample storage for detailed analysis
        sample_results = []

        with torch.no_grad():
            for batch_idx, (frames, labels, keyframe_indices) in enumerate(
                tqdm(val_loader, desc=f"Evaluating {loss_name}")
            ):
                frames = frames.to(device)
                labels = labels.to(device)
                keyframe_indices = keyframe_indices.to(device)

                logits = model(frames, keyframe_indices)
                probs = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                # Store batch results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of malignant
                all_logits.extend(logits.cpu().numpy())

                # Store per-sample info
                for i in range(len(labels)):
                    sample_results.append(
                        {
                            "true_label": labels[i].item(),
                            "pred_label": predictions[i].item(),
                            "prob_benign": probs[i, 0].item(),
                            "prob_malignant": probs[i, 1].item(),
                            "confidence": probs[i, predictions[i]].item(),
                            "correct": (predictions[i] == labels[i]).item(),
                        }
                    )

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_logits = np.array(all_logits)

        # Calculate comprehensive metrics
        accuracy = 100.0 * np.mean(all_predictions == all_labels)

        # Per-class accuracy
        class_accuracies = {}
        for cls in [0, 1]:
            mask = all_labels == cls
            if mask.sum() > 0:
                class_acc = 100.0 * np.mean(all_predictions[mask] == all_labels[mask])
                class_accuracies[self.class_names[cls]] = class_acc

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # ROC and PR curves
        fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)

        precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_probs)
        avg_precision = average_precision_score(all_labels, all_probs)

        # F1 score
        f1 = f1_score(all_labels, all_predictions)

        # Sensitivity and Specificity
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Store results
        results = {
            "loss_name": loss_name,
            "accuracy": accuracy,
            "class_accuracies": class_accuracies,
            "confusion_matrix": cm,
            "predictions": all_predictions,
            "labels": all_labels,
            "probs": all_probs,
            "logits": all_logits,
            "roc_curve": (fpr, tpr, roc_thresholds),
            "roc_auc": roc_auc,
            "pr_curve": (precision, recall, pr_thresholds),
            "avg_precision": avg_precision,
            "f1_score": f1,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "sample_results": sample_results,
        }

        # Print summary
        print(f"\n{loss_name} Results:")
        print(f"  Overall Accuracy: {accuracy:.2f}%")
        for cls_name, cls_acc in class_accuracies.items():
            print(f"  {cls_name} Accuracy: {cls_acc:.2f}%")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Sensitivity: {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")

        return results

    def compare_all_losses(self, val_loader, device="cuda"):
        """Evaluate all loss functions and compare."""

        print("\n" + "=" * 70)
        print("COMPREHENSIVE LOSS FUNCTION COMPARISON")
        print("=" * 70)

        # Check which checkpoints exist
        available_losses = []
        for loss_name, config in self.loss_configs.items():
            if Path(config["path"]).exists():
                available_losses.append(loss_name)
                print(f"✓ Found checkpoint: {loss_name}")
            else:
                print(f"✗ Missing checkpoint: {loss_name} ({config['path']})")

        if len(available_losses) == 0:
            print("\n❌ No checkpoints found! Please train models first.")
            return None

        print(f"\nComparing {len(available_losses)} loss functions...")

        # Evaluate each model
        for loss_name in available_losses:
            # Load model
            model = KGANet(
                num_classes=2,
                feature_dim=2048,
                reduction=16,
                aggregation_type="attention",
                pretrained=False,
            )

            checkpoint = torch.load(
                self.loss_configs[loss_name]["path"], map_location=device
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)

            # Evaluate
            self.results[loss_name] = self.evaluate_model(
                model, val_loader, device, loss_name
            )

            # Add checkpoint info
            self.results[loss_name]["checkpoint_epoch"] = checkpoint.get("epoch", "N/A")
            self.results[loss_name]["checkpoint_train_acc"] = checkpoint.get(
                "train_acc", "N/A"
            )
            self.results[loss_name]["color"] = self.loss_configs[loss_name]["color"]

        # Generate all comparison visualizations
        self.create_all_visualizations()

        # Save detailed report
        self.save_detailed_report()

        return self.results

    def create_all_visualizations(self):
        """Create all comparison visualizations."""

        print(f"\n{'='*70}")
        print("Generating Comparison Visualizations")
        print(f"{'='*70}\n")

        # 1. Overview metrics
        self.plot_overview_metrics()

        # 2. ROC curves
        self.plot_roc_curves()

        # 3. Precision-Recall curves
        self.plot_pr_curves()

        # 4. Confusion matrices
        self.plot_confusion_matrices()

        # 5. Confidence distributions
        self.plot_confidence_distributions()

        # 6. Per-class performance
        self.plot_per_class_performance()

        # 7. Error analysis
        self.plot_error_analysis()

        # 8. Calibration curves
        self.plot_calibration_curves()

        print(f"\n✓ All visualizations saved to: {self.save_dir}/")

    def plot_overview_metrics(self):
        """Plot overview of all metrics."""

        metrics = [
            "accuracy",
            "roc_auc",
            "avg_precision",
            "f1_score",
            "sensitivity",
            "specificity",
        ]
        metric_labels = [
            "Accuracy (%)",
            "ROC AUC",
            "Avg Precision",
            "F1 Score",
            "Sensitivity",
            "Specificity",
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]

            loss_names = list(self.results.keys())
            values = []
            colors = []

            for loss_name in loss_names:
                val = self.results[loss_name][metric]
                if metric == "accuracy":
                    val = val  # Already in percentage
                values.append(val)
                colors.append(self.results[loss_name]["color"])

            bars = ax.bar(
                loss_names, values, color=colors, edgecolor="black", linewidth=1.5
            )
            ax.set_ylabel(label, fontsize=12, fontweight="bold")
            ax.set_title(label, fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                if metric == "accuracy":
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.5,
                        f"{val:.2f}%",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{val:.4f}",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                    )

            # Set y-limits
            if metric == "accuracy":
                ax.set_ylim([0, 105])
            else:
                ax.set_ylim([0, 1.05])

        plt.suptitle(
            "Comprehensive Metrics Comparison Across Loss Functions",
            fontsize=16,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()
        plt.savefig(
            self.save_dir / "overview_metrics.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        print("✓ Saved overview metrics")

    def plot_roc_curves(self):
        """Plot ROC curves for all models."""

        fig, ax = plt.subplots(figsize=(10, 8))

        for loss_name, result in self.results.items():
            fpr, tpr, _ = result["roc_curve"]
            roc_auc = result["roc_auc"]
            color = result["color"]

            ax.plot(
                fpr,
                tpr,
                linewidth=3,
                label=f"{loss_name} (AUC = {roc_auc:.4f})",
                color=color,
            )

        # Random classifier line
        ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random (AUC = 0.5000)")

        ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
        ax.set_title("ROC Curves Comparison", fontsize=16, fontweight="bold")
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()
        plt.savefig(self.save_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
        plt.close()

        print("✓ Saved ROC curves")

    def plot_pr_curves(self):
        """Plot Precision-Recall curves."""

        fig, ax = plt.subplots(figsize=(10, 8))

        for loss_name, result in self.results.items():
            precision, recall, _ = result["pr_curve"]
            avg_prec = result["avg_precision"]
            color = result["color"]

            ax.plot(
                recall,
                precision,
                linewidth=3,
                label=f"{loss_name} (AP = {avg_prec:.4f})",
                color=color,
            )

        ax.set_xlabel("Recall", fontsize=14, fontweight="bold")
        ax.set_ylabel("Precision", fontsize=14, fontweight="bold")
        ax.set_title(
            "Precision-Recall Curves Comparison", fontsize=16, fontweight="bold"
        )
        ax.legend(loc="lower left", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()
        plt.savefig(self.save_dir / "pr_curves.png", dpi=150, bbox_inches="tight")
        plt.close()

        print("✓ Saved PR curves")

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""

        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        for idx, (loss_name, result) in enumerate(self.results.items()):
            cm = result["confusion_matrix"]
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            # Create annotation with both count and percentage
            annot = np.array(
                [
                    [f"{cm[i,j]}\n({cm_norm[i,j]:.1%})" for j in range(cm.shape[1])]
                    for i in range(cm.shape[0])
                ]
            )

            sns.heatmap(
                cm_norm,
                annot=annot,
                fmt="",
                cmap="Blues",
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=axes[idx],
                cbar_kws={"label": "Proportion"},
                vmin=0,
                vmax=1,
            )

            axes[idx].set_title(
                f'{loss_name}\n(Acc: {result["accuracy"]:.2f}%)',
                fontsize=12,
                fontweight="bold",
            )
            axes[idx].set_ylabel("True Label", fontsize=11)
            axes[idx].set_xlabel("Predicted Label", fontsize=11)

        plt.suptitle("Confusion Matrices Comparison", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            self.save_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        print("✓ Saved confusion matrices")

    def plot_confidence_distributions(self):
        """Plot prediction confidence distributions."""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Overall confidence distribution
        ax = axes[0, 0]
        for loss_name, result in self.results.items():
            confidences = [s["confidence"] for s in result["sample_results"]]
            ax.hist(
                confidences,
                bins=30,
                alpha=0.5,
                label=loss_name,
                color=result["color"],
                edgecolor="black",
            )

        ax.set_xlabel("Prediction Confidence", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Overall Confidence Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 2. Confidence for correct predictions
        ax = axes[0, 1]
        for loss_name, result in self.results.items():
            confidences = [
                s["confidence"] for s in result["sample_results"] if s["correct"]
            ]
            ax.hist(
                confidences,
                bins=30,
                alpha=0.5,
                label=loss_name,
                color=result["color"],
                edgecolor="black",
            )

        ax.set_xlabel("Prediction Confidence", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            "Confidence Distribution (Correct Predictions)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 3. Confidence for incorrect predictions
        ax = axes[1, 0]
        for loss_name, result in self.results.items():
            confidences = [
                s["confidence"] for s in result["sample_results"] if not s["correct"]
            ]
            if len(confidences) > 0:
                ax.hist(
                    confidences,
                    bins=20,
                    alpha=0.5,
                    label=loss_name,
                    color=result["color"],
                    edgecolor="black",
                )

        ax.set_xlabel("Prediction Confidence", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            "Confidence Distribution (Incorrect Predictions)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 4. Average confidence comparison
        ax = axes[1, 1]
        loss_names = list(self.results.keys())
        correct_conf = []
        incorrect_conf = []
        colors = []

        for loss_name in loss_names:
            result = self.results[loss_name]
            correct_c = [
                s["confidence"] for s in result["sample_results"] if s["correct"]
            ]
            incorrect_c = [
                s["confidence"] for s in result["sample_results"] if not s["correct"]
            ]

            correct_conf.append(np.mean(correct_c) if correct_c else 0)
            incorrect_conf.append(np.mean(incorrect_c) if incorrect_c else 0)
            colors.append(result["color"])

        x = np.arange(len(loss_names))
        width = 0.35

        ax.bar(
            x - width / 2,
            correct_conf,
            width,
            label="Correct",
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        ax.bar(
            x + width / 2,
            incorrect_conf,
            width,
            label="Incorrect",
            color=colors,
            alpha=0.4,
            edgecolor="black",
            hatch="//",
        )

        ax.set_ylabel("Average Confidence", fontsize=12)
        ax.set_title("Average Confidence Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(loss_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(
            self.save_dir / "confidence_distributions.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        print("✓ Saved confidence distributions")

    def plot_per_class_performance(self):
        """Plot per-class performance metrics."""

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        loss_names = list(self.results.keys())
        x = np.arange(len(loss_names))
        width = 0.35

        # Extract per-class accuracies
        benign_accs = [
            self.results[ln]["class_accuracies"]["Benign"] for ln in loss_names
        ]
        malignant_accs = [
            self.results[ln]["class_accuracies"]["Malignant"] for ln in loss_names
        ]
        colors = [self.results[ln]["color"] for ln in loss_names]

        # Bar plot
        ax = axes[0]
        ax.bar(
            x - width / 2,
            benign_accs,
            width,
            label="Benign",
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        ax.bar(
            x + width / 2,
            malignant_accs,
            width,
            label="Malignant",
            color=colors,
            alpha=0.4,
            edgecolor="black",
            hatch="//",
        )

        ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
        ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(loss_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 105])

        # Add value labels
        for i, (b_acc, m_acc) in enumerate(zip(benign_accs, malignant_accs)):
            ax.text(
                i - width / 2,
                b_acc + 1,
                f"{b_acc:.1f}%",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
            ax.text(
                i + width / 2,
                m_acc + 1,
                f"{m_acc:.1f}%",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        # Sensitivity vs Specificity
        ax = axes[1]
        sensitivities = [self.results[ln]["sensitivity"] * 100 for ln in loss_names]
        specificities = [self.results[ln]["specificity"] * 100 for ln in loss_names]

        ax.bar(
            x - width / 2,
            sensitivities,
            width,
            label="Sensitivity (Recall)",
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        ax.bar(
            x + width / 2,
            specificities,
            width,
            label="Specificity",
            color=colors,
            alpha=0.4,
            edgecolor="black",
            hatch="//",
        )

        ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
        ax.set_title("Sensitivity vs Specificity", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(loss_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 105])

        # Add value labels
        for i, (sens, spec) in enumerate(zip(sensitivities, specificities)):
            ax.text(
                i - width / 2,
                sens + 1,
                f"{sens:.1f}%",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
            ax.text(
                i + width / 2,
                spec + 1,
                f"{spec:.1f}%",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            self.save_dir / "per_class_performance.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        print("✓ Saved per-class performance")

    def plot_error_analysis(self):
        """Analyze types of errors made by each model."""

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Error counts
        ax = axes[0]
        loss_names = list(self.results.keys())

        benign_to_malignant = []
        malignant_to_benign = []
        colors = []

        for loss_name in loss_names:
            cm = self.results[loss_name]["confusion_matrix"]
            benign_to_malignant.append(cm[0, 1])  # FP
            malignant_to_benign.append(cm[1, 0])  # FN
            colors.append(self.results[loss_name]["color"])

        x = np.arange(len(loss_names))
        width = 0.35

        ax.bar(
            x - width / 2,
            benign_to_malignant,
            width,
            label="Benign → Malignant (FP)",
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        ax.bar(
            x + width / 2,
            malignant_to_benign,
            width,
            label="Malignant → Benign (FN)",
            color=colors,
            alpha=0.4,
            edgecolor="black",
            hatch="//",
        )

        ax.set_ylabel("Error Count", fontsize=12, fontweight="bold")
        ax.set_title("Types of Classification Errors", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(loss_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for i, (fp, fn) in enumerate(zip(benign_to_malignant, malignant_to_benign)):
            ax.text(
                i - width / 2,
                fp + 0.5,
                f"{fp}",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
            ax.text(
                i + width / 2,
                fn + 0.5,
                f"{fn}",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        # Total errors
        ax = axes[1]
        total_errors = [b + m for b, m in zip(benign_to_malignant, malignant_to_benign)]

        bars = ax.bar(
            loss_names, total_errors, color=colors, edgecolor="black", linewidth=2
        )
        ax.set_ylabel("Total Errors", fontsize=12, fontweight="bold")
        ax.set_title("Total Classification Errors", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, err in zip(bars, total_errors):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{err}",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(self.save_dir / "error_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

        print("✓ Saved error analysis")

    def plot_calibration_curves(self):
        """Plot calibration curves to assess confidence calibration."""

        from sklearn.calibration import calibration_curve

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Calibration curve
        ax = axes[0]
        for loss_name, result in self.results.items():
            prob_true, prob_pred = calibration_curve(
                result["labels"], result["probs"], n_bins=10, strategy="uniform"
            )

            ax.plot(
                prob_pred,
                prob_true,
                marker="o",
                linewidth=2,
                markersize=8,
                label=loss_name,
                color=result["color"],
            )

        ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect Calibration")
        ax.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight="bold")
        ax.set_ylabel("Fraction of Positives", fontsize=12, fontweight="bold")
        ax.set_title("Calibration Curves", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Expected Calibration Error (ECE)
        ax = axes[1]
        loss_names = list(self.results.keys())
        eces = []
        colors = []

        for loss_name in loss_names:
            result = self.results[loss_name]
            prob_true, prob_pred = calibration_curve(
                result["labels"], result["probs"], n_bins=10, strategy="uniform"
            )
            ece = np.mean(np.abs(prob_true - prob_pred))
            eces.append(ece)
            colors.append(result["color"])

        bars = ax.bar(loss_names, eces, color=colors, edgecolor="black", linewidth=2)
        ax.set_ylabel("Expected Calibration Error", fontsize=12, fontweight="bold")
        ax.set_title("Model Calibration Quality", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, ece in zip(bars, eces):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.005,
                f"{ece:.4f}",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            self.save_dir / "calibration_curves.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        print("✓ Saved calibration curves")

    def save_detailed_report(self):
        """Save detailed text and CSV reports."""

        # Create summary DataFrame
        summary_data = []
        for loss_name, result in self.results.items():
            cm = result["confusion_matrix"]
            tn, fp, fn, tp = cm.ravel()

            summary_data.append(
                {
                    "Loss Function": loss_name,
                    "Accuracy (%)": f"{result['accuracy']:.2f}",
                    "Benign Acc (%)": f"{result['class_accuracies']['Benign']:.2f}",
                    "Malignant Acc (%)": f"{result['class_accuracies']['Malignant']:.2f}",
                    "ROC AUC": f"{result['roc_auc']:.4f}",
                    "Avg Precision": f"{result['avg_precision']:.4f}",
                    "F1 Score": f"{result['f1_score']:.4f}",
                    "Sensitivity": f"{result['sensitivity']:.4f}",
                    "Specificity": f"{result['specificity']:.4f}",
                    "True Positives": tp,
                    "True Negatives": tn,
                    "False Positives": fp,
                    "False Negatives": fn,
                    "Total Errors": fp + fn,
                    "Checkpoint Epoch": result["checkpoint_epoch"],
                }
            )

        df_summary = pd.DataFrame(summary_data)

        # Save CSV
        csv_path = self.save_dir / "comparison_summary.csv"
        df_summary.to_csv(csv_path, index=False)
        print(f"✓ Saved summary CSV to {csv_path}")

        # Save detailed text report
        report_path = self.save_dir / "detailed_report.txt"
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("KGA-NET LOSS FUNCTION COMPARISON - DETAILED REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of models compared: {len(self.results)}\n")
            f.write("=" * 80 + "\n\n")

            # Overall ranking
            f.write("OVERALL PERFORMANCE RANKING\n")
            f.write("-" * 80 + "\n")
            sorted_by_acc = sorted(
                self.results.items(), key=lambda x: x[1]["accuracy"], reverse=True
            )
            for rank, (loss_name, result) in enumerate(sorted_by_acc, 1):
                f.write(f"{rank}. {loss_name}: {result['accuracy']:.2f}% accuracy\n")
            f.write("\n")

            # Detailed metrics for each model
            for loss_name, result in self.results.items():
                f.write("=" * 80 + "\n")
                f.write(f"{loss_name.upper()}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Description: {self.loss_configs[loss_name]['description']}\n")
                f.write(f"Checkpoint: {self.loss_configs[loss_name]['path']}\n")
                f.write(f"Trained Epoch: {result['checkpoint_epoch']}\n")
                f.write(f"Training Accuracy: {result['checkpoint_train_acc']}\n\n")

                f.write("OVERALL METRICS:\n")
                f.write(f"  Overall Accuracy:     {result['accuracy']:.2f}%\n")
                f.write(f"  ROC AUC:              {result['roc_auc']:.4f}\n")
                f.write(f"  Average Precision:    {result['avg_precision']:.4f}\n")
                f.write(f"  F1 Score:             {result['f1_score']:.4f}\n\n")

                f.write("PER-CLASS METRICS:\n")
                f.write(
                    f"  Benign Accuracy:      {result['class_accuracies']['Benign']:.2f}%\n"
                )
                f.write(
                    f"  Malignant Accuracy:   {result['class_accuracies']['Malignant']:.2f}%\n"
                )
                f.write(f"  Sensitivity (Recall): {result['sensitivity']:.4f}\n")
                f.write(f"  Specificity:          {result['specificity']:.4f}\n\n")

                f.write("CONFUSION MATRIX:\n")
                cm = result["confusion_matrix"]
                f.write(f"                    Predicted\n")
                f.write(f"                Benign  Malignant\n")
                f.write(f"Actual Benign     {cm[0,0]:4d}    {cm[0,1]:4d}\n")
                f.write(f"       Malignant  {cm[1,0]:4d}    {cm[1,1]:4d}\n\n")

                tn, fp, fn, tp = cm.ravel()
                f.write("ERROR ANALYSIS:\n")
                f.write(f"  False Positives (Benign → Malignant):   {fp}\n")
                f.write(f"  False Negatives (Malignant → Benign):   {fn}\n")
                f.write(f"  Total Errors:                           {fp + fn}\n\n")

                # Confidence statistics
                confidences = [s["confidence"] for s in result["sample_results"]]
                correct_conf = [
                    s["confidence"] for s in result["sample_results"] if s["correct"]
                ]
                incorrect_conf = [
                    s["confidence"]
                    for s in result["sample_results"]
                    if not s["correct"]
                ]

                f.write("CONFIDENCE STATISTICS:\n")
                f.write(
                    f"  Overall Mean Confidence:       {np.mean(confidences):.4f}\n"
                )
                f.write(
                    f"  Correct Predictions Mean:      {np.mean(correct_conf) if correct_conf else 0:.4f}\n"
                )
                f.write(
                    f"  Incorrect Predictions Mean:    {np.mean(incorrect_conf) if incorrect_conf else 0:.4f}\n"
                )
                f.write(
                    f"  Confidence Std Dev:            {np.std(confidences):.4f}\n\n"
                )

            # Comparison summary
            f.write("=" * 80 + "\n")
            f.write("COMPARATIVE ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            # Best in each metric
            metrics_to_compare = [
                ("accuracy", "Highest Accuracy", "%.2f%%"),
                ("roc_auc", "Highest ROC AUC", "%.4f"),
                ("avg_precision", "Highest Average Precision", "%.4f"),
                ("f1_score", "Highest F1 Score", "%.4f"),
                ("sensitivity", "Highest Sensitivity", "%.4f"),
                ("specificity", "Highest Specificity", "%.4f"),
            ]

            f.write("BEST PERFORMERS BY METRIC:\n")
            for metric, label, fmt in metrics_to_compare:
                best_loss = max(self.results.items(), key=lambda x: x[1][metric])
                value = best_loss[1][metric]
                if metric == "accuracy":
                    f.write(f"  {label:35s}: {best_loss[0]} ({fmt % value})\n")
                else:
                    f.write(f"  {label:35s}: {best_loss[0]} ({fmt % value})\n")

            f.write("\n")
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 80 + "\n")

            # Find best overall
            best_overall = max(self.results.items(), key=lambda x: x[1]["accuracy"])
            f.write(f"1. BEST OVERALL: {best_overall[0]}\n")
            f.write(f"   - Highest accuracy: {best_overall[1]['accuracy']:.2f}%\n")
            f.write(f"   - Recommended for general use\n\n")

            # Find best for sensitivity
            best_sensitivity = max(
                self.results.items(), key=lambda x: x[1]["sensitivity"]
            )
            f.write(f"2. BEST FOR MALIGNANT DETECTION: {best_sensitivity[0]}\n")
            f.write(
                f"   - Highest sensitivity: {best_sensitivity[1]['sensitivity']:.4f}\n"
            )
            f.write(f"   - Recommended when missing malignant cases is critical\n\n")

            # Find best for specificity
            best_specificity = max(
                self.results.items(), key=lambda x: x[1]["specificity"]
            )
            f.write(f"3. BEST FOR REDUCING FALSE ALARMS: {best_specificity[0]}\n")
            f.write(
                f"   - Highest specificity: {best_specificity[1]['specificity']:.4f}\n"
            )
            f.write(f"   - Recommended when reducing false positives is important\n\n")

            # Find most balanced
            balances = {
                name: abs(result["sensitivity"] - result["specificity"])
                for name, result in self.results.items()
            }
            most_balanced = min(balances.items(), key=lambda x: x[1])
            f.write(f"4. MOST BALANCED: {most_balanced[0]}\n")
            sens = self.results[most_balanced[0]]["sensitivity"]
            spec = self.results[most_balanced[0]]["specificity"]
            f.write(f"   - Sensitivity: {sens:.4f}, Specificity: {spec:.4f}\n")
            f.write(f"   - Difference: {most_balanced[1]:.4f}\n")
            f.write(f"   - Recommended for balanced performance\n\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"✓ Saved detailed report to {report_path}")

        # Save results as JSON for programmatic access
        json_results = {}
        for loss_name, result in self.results.items():
            json_results[loss_name] = {
                "accuracy": float(result["accuracy"]),
                "roc_auc": float(result["roc_auc"]),
                "avg_precision": float(result["avg_precision"]),
                "f1_score": float(result["f1_score"]),
                "sensitivity": float(result["sensitivity"]),
                "specificity": float(result["specificity"]),
                "class_accuracies": {
                    k: float(v) for k, v in result["class_accuracies"].items()
                },
                "confusion_matrix": result["confusion_matrix"].tolist(),
                "checkpoint_epoch": str(result["checkpoint_epoch"]),
                "description": self.loss_configs[loss_name]["description"],
            }

        json_path = self.save_dir / "comparison_results.json"
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=4)

        print(f"✓ Saved JSON results to {json_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function."""

    print("\n" + "=" * 80)
    print("KGA-NET: COMPREHENSIVE LOSS FUNCTION COMPARISON")
    print("=" * 80)
    print("\nThis script compares ALL THREE loss functions:")
    print("  1. Coherence Loss")
    print("  2. Triplet Coherence Loss")
    print("  3. Standard Triplet Loss")
    print("=" * 80 + "\n")

    # Configuration
    ROOT_DIR = "./data"
    VAL_ANNOTATION = "imagenet_vid_val.json"
    BATCH_SIZE = 4
    NUM_FRAMES = 32
    SAVE_DIR = "loss_comparison_results"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load validation dataloader
    print("Loading validation dataset...")
    _, val_loader = create_dataloaders(
        root_dir=ROOT_DIR,
        train_annotation="imagenet_vid_train_15frames.json",
        val_annotation=VAL_ANNOTATION,
        batch_size=BATCH_SIZE,
        num_frames=NUM_FRAMES,
        num_workers=4,
    )
    print(f"✓ Loaded {len(val_loader.dataset)} validation samples\n")

    # Initialize analyzer
    analyzer = LossComparisonAnalyzer(save_dir=SAVE_DIR)

    # Run comprehensive comparison
    results = analyzer.compare_all_losses(val_loader, device=device)

    if results is None:
        print("\n❌ Comparison failed - no checkpoints found!")
        print("\nPlease ensure you have trained models with all three loss functions.")
        print("Run the training script (kaustav_kga.py) first to generate checkpoints.")
        return

    # Print rankings
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )
    print("\n🏆 RANKING BY ACCURACY:")
    for rank, (loss_name, result) in enumerate(sorted_results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        print(f"  {medal} {rank}. {loss_name:20s} {result['accuracy']:6.2f}%")

    # Best metrics
    print("\n⭐ BEST IN EACH CATEGORY:")
    best_auc = max(results.items(), key=lambda x: x[1]["roc_auc"])
    best_f1 = max(results.items(), key=lambda x: x[1]["f1_score"])
    best_sens = max(results.items(), key=lambda x: x[1]["sensitivity"])

    print(f"  • ROC AUC:       {best_auc[0]:20s} ({best_auc[1]['roc_auc']:.4f})")
    print(f"  • F1 Score:      {best_f1[0]:20s} ({best_f1[1]['f1_score']:.4f})")
    print(f"  • Sensitivity:   {best_sens[0]:20s} ({best_sens[1]['sensitivity']:.4f})")

if __name__ == "__main__":
    main()
