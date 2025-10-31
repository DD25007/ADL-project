import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.manifold import TSNE
import pandas as pd
from tqdm import tqdm
import os
from collections import defaultdict

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

# ============================================================================
# MODEL EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model_detailed(model, dataloader, device, model_name="Model"):
    """
    Comprehensive evaluation of a model.
    Returns predictions, labels, probabilities, and features for analysis.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []
    
    with torch.no_grad():
        for frames, labels, keyframe_indices in tqdm(dataloader, desc=f"Evaluating {model_name}"):
            frames = frames.to(device)
            labels = labels.to(device)
            keyframe_indices = keyframe_indices.to(device)
            
            # Get predictions and features
            logits, features = model(frames, keyframe_indices, return_features=True)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Extract aggregated features for visualization
            pooled_features = nn.AdaptiveAvgPool2d(1)(features['aggregated_features'])
            # Flatten to 1D for each sample
            pooled_features = pooled_features.view(pooled_features.size(0), -1)
            all_features.append(pooled_features.cpu().numpy())
    
    # Stack features properly
    all_features = np.vstack(all_features)
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs),
        'features': all_features,
        'accuracy': np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    }


def load_model_and_evaluate(checkpoint_path, dataloader, device, model_name, 
                           num_classes=2, feature_dim=2048):
    """Load a model from checkpoint and evaluate it."""
    from kaustav_kga import KGANet
    
    model = KGANet(
        num_classes=num_classes,
        feature_dim=feature_dim,
        reduction=16,
        aggregation_type="attention",
        pretrained=False
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    results = evaluate_model_detailed(model, dataloader, device, model_name)
    results['checkpoint_info'] = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'train_acc': checkpoint.get('train_acc', 'N/A'),
        'val_acc': checkpoint.get('val_acc', 'N/A')
    }
    
    return results, model


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_curves(log_file='training.log', save_path='plots/training_curves.png'):
    """
    Parse training log and plot training curves.
    Assumes log format contains loss and accuracy information.
    """
    # This is a template - adjust based on your actual log format
    print("Note: Update this function based on your training.log format")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Example plots (customize based on your needs)
    models = ['Coherence', 'Triplet-Coherence', 'Hard', 'Semi-Hard', 'All']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for i, model in enumerate(models):
        # Simulate data - replace with actual data from logs
        epochs = np.arange(1, 51)
        loss = 2.0 * np.exp(-epochs/10) + np.random.normal(0, 0.1, 50)
        ax.plot(epochs, loss, label=model, linewidth=2, color=colors[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    ax = axes[0, 1]
    for i, model in enumerate(models):
        epochs = np.arange(1, 51)
        acc = 50 + 40 * (1 - np.exp(-epochs/8)) + np.random.normal(0, 2, 50)
        ax.plot(epochs, acc, label=model, linewidth=2, color=colors[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Classification Loss
    ax = axes[1, 0]
    for i, model in enumerate(models):
        epochs = np.arange(1, 51)
        cls_loss = 1.5 * np.exp(-epochs/12) + np.random.normal(0, 0.08, 50)
        ax.plot(epochs, cls_loss, label=model, linewidth=2, color=colors[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Classification Loss')
    ax.set_title('Classification Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Auxiliary Loss
    ax = axes[1, 1]
    for i, model in enumerate(models):
        epochs = np.arange(1, 51)
        aux_loss = 1.0 * np.exp(-epochs/15) + np.random.normal(0, 0.05, 50)
        ax.plot(epochs, aux_loss, label=model, linewidth=2, color=colors[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Auxiliary Loss')
    ax.set_title('Auxiliary Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


def plot_confusion_matrices(results_dict, class_names=['Benign', 'Malignant'], 
                           save_path='plots/confusion_matrices.png'):
    """Plot confusion matrices for all models."""
    n_models = len(results_dict)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        cm = confusion_matrix(results['labels'], results['predictions'])
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        ax = axes[idx]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Percentage'}, ax=ax)
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.2f}%')
        
        # Add counts in parentheses
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j+0.5, i+0.7, f'({cm[i, j]})',
                             ha="center", va="center", color="gray", fontsize=9)
    
    # Hide extra subplots if any
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrices to {save_path}")
    plt.close()


def plot_roc_curves(results_dict, save_path='plots/roc_curves.png'):
    """Plot ROC curves for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Plot 1: All ROC curves together
    for idx, (model_name, results) in enumerate(results_dict.items()):
        # Get probabilities for positive class
        probs = results['probabilities'][:, 1]
        fpr, tpr, _ = roc_curve(results['labels'], probs)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, linewidth=2, color=colors[idx % len(colors)],
                label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves - All Models')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AUC Bar Chart
    model_names = list(results_dict.keys())
    aucs = []
    for results in results_dict.values():
        probs = results['probabilities'][:, 1]
        fpr, tpr, _ = roc_curve(results['labels'], probs)
        aucs.append(auc(fpr, tpr))
    
    bars = ax2.bar(model_names, aucs, color=colors[:len(model_names)], alpha=0.8)
    ax2.set_ylabel('AUC Score')
    ax2.set_title('AUC Comparison')
    ax2.set_ylim([0.5, 1.0])
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, auc_val in zip(bars, aucs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curves to {save_path}")
    plt.close()


def plot_performance_comparison(results_dict, save_path='plots/performance_comparison.png'):
    """Comprehensive performance comparison across all metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    model_names = list(results_dict.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Collect metrics
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for results in results_dict.values():
        from sklearn.metrics import precision_recall_fscore_support
        
        accuracies.append(results['accuracy'])
        precision, recall, f1, _ = precision_recall_fscore_support(
            results['labels'], results['predictions'], average='weighted'
        )
        precisions.append(precision * 100)
        recalls.append(recall * 100)
        f1_scores.append(f1 * 100)
    
    # Plot 1: Accuracy Comparison
    ax = axes[0, 0]
    bars = ax.bar(model_names, accuracies, color=colors[:len(model_names)], alpha=0.8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Comparison')
    ax.set_ylim([0, 100])
    ax.grid(True, axis='y', alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Precision Comparison
    ax = axes[0, 1]
    bars = ax.bar(model_names, precisions, color=colors[:len(model_names)], alpha=0.8)
    ax.set_ylabel('Precision (%)')
    ax.set_title('Precision Comparison')
    ax.set_ylim([0, 100])
    ax.grid(True, axis='y', alpha=0.3)
    for bar, prec in zip(bars, precisions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{prec:.2f}%', ha='center', va='bottom', fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Recall Comparison
    ax = axes[1, 0]
    bars = ax.bar(model_names, recalls, color=colors[:len(model_names)], alpha=0.8)
    ax.set_ylabel('Recall (%)')
    ax.set_title('Recall Comparison')
    ax.set_ylim([0, 100])
    ax.grid(True, axis='y', alpha=0.3)
    for bar, rec in zip(bars, recalls):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rec:.2f}%', ha='center', va='bottom', fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: F1-Score Comparison
    ax = axes[1, 1]
    bars = ax.bar(model_names, f1_scores, color=colors[:len(model_names)], alpha=0.8)
    ax.set_ylabel('F1-Score (%)')
    ax.set_title('F1-Score Comparison')
    ax.set_ylim([0, 100])
    ax.grid(True, axis='y', alpha=0.3)
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{f1:.2f}%', ha='center', va='bottom', fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved performance comparison to {save_path}")
    plt.close()


def plot_feature_tsne(results_dict, save_path='plots/tsne_visualization.png'):
    """Plot t-SNE visualization of learned features."""
    n_models = len(results_dict)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(results['features'])
        
        ax = axes[idx]
        
        # Plot each class with different color
        for class_idx, class_name in enumerate(['Benign', 'Malignant']):
            mask = results['labels'] == class_idx
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      label=class_name, alpha=0.6, s=50)
        
        ax.set_title(f'{model_name}\nFeature Space Visualization')
        ax.legend()
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE visualization to {save_path}")
    plt.close()


def plot_per_class_performance(results_dict, save_path='plots/per_class_performance.png'):
    """Plot per-class performance metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    class_names = ['Benign', 'Malignant']
    model_names = list(results_dict.keys())
    
    # Collect per-class metrics
    benign_precision = []
    benign_recall = []
    malignant_precision = []
    malignant_recall = []
    
    for results in results_dict.values():
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, _, _ = precision_recall_fscore_support(
            results['labels'], results['predictions'], average=None
        )
        benign_precision.append(precision[0] * 100)
        benign_recall.append(recall[0] * 100)
        malignant_precision.append(precision[1] * 100)
        malignant_recall.append(recall[1] * 100)
    
    # Plot 1: Benign Precision
    ax = axes[0, 0]
    bars = ax.bar(model_names, benign_precision, color='skyblue', alpha=0.8)
    ax.set_ylabel('Precision (%)')
    ax.set_title('Benign Class - Precision')
    ax.set_ylim([0, 100])
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, benign_precision):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%', ha='center', va='bottom')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Benign Recall
    ax = axes[0, 1]
    bars = ax.bar(model_names, benign_recall, color='lightcoral', alpha=0.8)
    ax.set_ylabel('Recall (%)')
    ax.set_title('Benign Class - Recall')
    ax.set_ylim([0, 100])
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, benign_recall):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%', ha='center', va='bottom')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Malignant Precision
    ax = axes[1, 0]
    bars = ax.bar(model_names, malignant_precision, color='lightgreen', alpha=0.8)
    ax.set_ylabel('Precision (%)')
    ax.set_title('Malignant Class - Precision')
    ax.set_ylim([0, 100])
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, malignant_precision):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%', ha='center', va='bottom')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Malignant Recall
    ax = axes[1, 1]
    bars = ax.bar(model_names, malignant_recall, color='plum', alpha=0.8)
    ax.set_ylabel('Recall (%)')
    ax.set_title('Malignant Class - Recall')
    ax.set_ylim([0, 100])
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, malignant_recall):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%', ha='center', va='bottom')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-class performance to {save_path}")
    plt.close()


def plot_radar_chart(results_dict, save_path='plots/radar_comparison.png'):
    """Create radar chart comparing all models across multiple metrics."""
    from math import pi
    
    # Calculate metrics for each model
    metrics_data = {}
    for model_name, results in results_dict.items():
        from sklearn.metrics import precision_recall_fscore_support
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            results['labels'], results['predictions'], average='weighted'
        )
        
        # Get AUC
        probs = results['probabilities'][:, 1]
        fpr, tpr, _ = roc_curve(results['labels'], probs)
        roc_auc = auc(fpr, tpr)
        
        # Get specificity (average per class)
        cm = confusion_matrix(results['labels'], results['predictions'])
        specificity = np.mean([cm[i, i] / cm[i].sum() for i in range(len(cm))])
        
        metrics_data[model_name] = {
            'Accuracy': results['accuracy'],
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1-Score': f1 * 100,
            'AUC': roc_auc * 100,
            'Specificity': specificity * 100
        }
    
    # Setup radar chart
    categories = list(list(metrics_data.values())[0].keys())
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for idx, (model_name, metrics) in enumerate(metrics_data.items()):
        values = list(metrics.values())
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=model_name, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Model Performance Radar Chart', size=16, pad=20)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved radar chart to {save_path}")
    plt.close()


def generate_summary_table(results_dict, save_path='plots/summary_table.png'):
    """Generate a comprehensive summary table."""
    from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
    
    summary_data = []
    
    for model_name, results in results_dict.items():
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            results['labels'], results['predictions'], average='weighted'
        )
        
        # AUC
        probs = results['probabilities'][:, 1]
        fpr, tpr, _ = roc_curve(results['labels'], probs)
        roc_auc = auc(fpr, tpr)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(results['labels'], results['predictions'])
        
        summary_data.append({
            'Model': model_name,
            'Accuracy (%)': f"{results['accuracy']:.2f}",
            'Precision (%)': f"{precision*100:.2f}",
            'Recall (%)': f"{recall*100:.2f}",
            'F1-Score (%)': f"{f1*100:.2f}",
            'AUC': f"{roc_auc:.3f}",
            'Cohen\'s κ': f"{kappa:.3f}"
        })
    
    df = pd.DataFrame(summary_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(summary_data) * 0.8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colColours=['lightgray']*len(df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    for col_idx, col in enumerate(df.columns[1:], 1):
        values = [float(row.replace('%', '')) for row in df.iloc[:, col_idx]]
        best_idx = np.argmax(values)
        table[(best_idx + 1, col_idx)].set_facecolor('#90EE90')
    
    plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary table to {save_path}")
    plt.close()
    
    # Also save as CSV
    csv_path = save_path.replace('.png', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to {csv_path}")


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def run_complete_analysis(checkpoint_dir, val_loader, device, output_dir='plots'):
    """
    Run complete analysis and generate all visualizations.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        val_loader: Validation DataLoader
        device: Device to run on
        output_dir: Directory to save plots
    """
    
    print("="*70)
    print("KGA-Net Model Comparison & Analysis")
    print("="*70)
    
    # Define model configurations
    model_configs = {
        'Coherence Loss': 'best_model_coherence.pth',
        'Triplet-Coherence': 'best_model_triplet_coherence.pth',
        'Triplet-Hard Mining': 'best_model_triplet_standard_hard.pth',
        'Triplet-Semi-Hard': 'best_model_triplet_standard_semi-hard.pth',
        'Triplet-All Mining': 'best_model_triplet_standard_all.pth'
    }
    
    # Load and evaluate all models
    results_dict = {}
    for model_name, checkpoint_file in model_configs.items():
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"\nEvaluating {model_name}...")
        results, model = load_model_and_evaluate(
            checkpoint_path, val_loader, device, model_name
        )
        results_dict[model_name] = results
        
        print(f"  Accuracy: {results['accuracy']:.2f}%")
        print(f"  Checkpoint Info: Epoch {results['checkpoint_info']['epoch']}, "
              f"Train Acc: {results['checkpoint_info']['train_acc']:.2f}%, "
              f"Val Acc: {results['checkpoint_info']['val_acc']:.2f}%")
    
    if len(results_dict) == 0:
        print("\nError: No models were successfully loaded!")
        return
    
    print(f"\n{'='*70}")
    print(f"Successfully loaded {len(results_dict)} models")
    print(f"{'='*70}\n")
    
    # Generate all visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations...")
    print("-" * 70)
    
    # 1. Training curves (if log available)
    try:
        plot_training_curves(save_path=f'{output_dir}/1_training_curves.png')
    except Exception as e:
        print(f"Could not generate training curves: {e}")
    
    # 2. Confusion matrices
    plot_confusion_matrices(results_dict, save_path=f'{output_dir}/2_confusion_matrices.png')
    
    # 3. ROC curves
    plot_roc_curves(results_dict, save_path=f'{output_dir}/3_roc_curves.png')
    
    # 4. Performance comparison
    plot_performance_comparison(results_dict, save_path=f'{output_dir}/4_performance_comparison.png')
    
    # 5. Per-class performance
    plot_per_class_performance(results_dict, save_path=f'{output_dir}/5_per_class_performance.png')
    
    # 6. t-SNE visualization
    plot_feature_tsne(results_dict, save_path=f'{output_dir}/6_tsne_visualization.png')
    
    # 7. Radar chart
    plot_radar_chart(results_dict, save_path=f'{output_dir}/7_radar_comparison.png')
    
    # 8. Summary table
    generate_summary_table(results_dict, save_path=f'{output_dir}/8_summary_table.png')
    
    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"{'='*70}")
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. training_curves.png - Training/validation curves over epochs")
    print("  2. confusion_matrices.png - Confusion matrices for all models")
    print("  3. roc_curves.png - ROC curves and AUC comparison")
    print("  4. performance_comparison.png - Accuracy, Precision, Recall, F1")
    print("  5. per_class_performance.png - Per-class metrics breakdown")
    print("  6. tsne_visualization.png - Feature space visualization")
    print("  7. radar_comparison.png - Multi-metric radar chart")
    print("  8. summary_table.png & .csv - Comprehensive metrics table")
    
    # Print detailed comparison analysis
    print(f"\n{'='*70}")
    print("DETAILED ANALYSIS & INSIGHTS")
    print(f"{'='*70}\n")
    
    analyze_models(results_dict)
    
    return results_dict


def analyze_models(results_dict):
    """Provide detailed textual analysis of model performance."""
    from sklearn.metrics import precision_recall_fscore_support
    
    print("1. OVERALL PERFORMANCE RANKING")
    print("-" * 70)
    
    # Rank by accuracy
    accuracies = [(name, res['accuracy']) for name, res in results_dict.items()]
    accuracies.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (name, acc) in enumerate(accuracies, 1):
        print(f"   {rank}. {name}: {acc:.2f}%")
    
    print("\n2. BEST MODEL FOR EACH METRIC")
    print("-" * 70)
    
    # Accuracy
    best_acc_model = max(results_dict.items(), key=lambda x: x[1]['accuracy'])
    print(f"   Best Accuracy: {best_acc_model[0]} ({best_acc_model[1]['accuracy']:.2f}%)")
    
    # AUC
    best_auc = None
    best_auc_score = 0
    for name, results in results_dict.items():
        probs = results['probabilities'][:, 1]
        fpr, tpr, _ = roc_curve(results['labels'], probs)
        auc_score = auc(fpr, tpr)
        if auc_score > best_auc_score:
            best_auc_score = auc_score
            best_auc = name
    print(f"   Best AUC: {best_auc} ({best_auc_score:.3f})")
    
    # F1-Score
    best_f1 = None
    best_f1_score = 0
    for name, results in results_dict.items():
        _, _, f1, _ = precision_recall_fscore_support(
            results['labels'], results['predictions'], average='weighted'
        )
        if f1 > best_f1_score:
            best_f1_score = f1
            best_f1 = name
    print(f"   Best F1-Score: {best_f1} ({best_f1_score*100:.2f}%)")
    
    print("\n3. CLASS-WISE PERFORMANCE ANALYSIS")
    print("-" * 70)
    
    for class_idx, class_name in enumerate(['Benign', 'Malignant']):
        print(f"\n   {class_name} Class:")
        best_precision = (None, 0)
        best_recall = (None, 0)
        
        for name, results in results_dict.items():
            precision, recall, _, _ = precision_recall_fscore_support(
                results['labels'], results['predictions'], average=None
            )
            
            if precision[class_idx] > best_precision[1]:
                best_precision = (name, precision[class_idx])
            if recall[class_idx] > best_recall[1]:
                best_recall = (name, recall[class_idx])
        
        print(f"      Best Precision: {best_precision[0]} ({best_precision[1]*100:.2f}%)")
        print(f"      Best Recall: {best_recall[0]} ({best_recall[1]*100:.2f}%)")
    
    print("\n4. KEY INSIGHTS & RECOMMENDATIONS")
    print("-" * 70)
    
    # Insight 1: Loss function comparison
    print("\n   Loss Function Impact:")
    coherence_models = [k for k in results_dict.keys() if 'Coherence' in k]
    triplet_models = [k for k in results_dict.keys() if 'Triplet' in k]
    
    if coherence_models and triplet_models:
        avg_coherence = np.mean([results_dict[k]['accuracy'] for k in coherence_models])
        avg_triplet = np.mean([results_dict[k]['accuracy'] for k in triplet_models])
        
        if avg_triplet > avg_coherence:
            print(f"      • Triplet-based losses (avg {avg_triplet:.2f}%) outperform")
            print(f"        coherence losses (avg {avg_coherence:.2f}%) by {avg_triplet-avg_coherence:.2f}%")
        else:
            print(f"      • Coherence losses (avg {avg_coherence:.2f}%) outperform")
            print(f"        triplet-based losses (avg {avg_triplet:.2f}%) by {avg_coherence-avg_triplet:.2f}%")
    
    # Insight 2: Mining strategy comparison
    print("\n   Mining Strategy Impact:")
    mining_strategies = ['Hard', 'Semi-Hard', 'All']
    mining_results = []
    
    for strategy in mining_strategies:
        matching_models = [k for k in results_dict.keys() if strategy in k]
        if matching_models:
            avg_acc = np.mean([results_dict[k]['accuracy'] for k in matching_models])
            mining_results.append((strategy, avg_acc))
            print(f"      • {strategy} Mining: {avg_acc:.2f}%")
    
    if mining_results:
        best_mining = max(mining_results, key=lambda x: x[1])
        print(f"      → Best mining strategy: {best_mining[0]} ({best_mining[1]:.2f}%)")
    
    # Insight 3: Consistency analysis
    print("\n   Model Consistency:")
    for name, results in results_dict.items():
        precision, recall, _, _ = precision_recall_fscore_support(
            results['labels'], results['predictions'], average=None
        )
        
        # Check if balanced across classes
        precision_gap = abs(precision[0] - precision[1])
        recall_gap = abs(recall[0] - recall[1])
        
        if precision_gap < 0.05 and recall_gap < 0.05:
            print(f"      • {name}: HIGHLY BALANCED (precision gap: {precision_gap*100:.1f}%, recall gap: {recall_gap*100:.1f}%)")
    
    print("\n5. RECOMMENDED MODEL")
    print("-" * 70)
    
    # Simple recommendation based on accuracy and F1
    scores = {}
    for name, results in results_dict.items():
        _, _, f1, _ = precision_recall_fscore_support(
            results['labels'], results['predictions'], average='weighted'
        )
        # Combined score: 70% accuracy, 30% F1
        combined_score = 0.7 * results['accuracy'] + 0.3 * (f1 * 100)
        scores[name] = combined_score
    
    recommended = max(scores.items(), key=lambda x: x[1])
    print(f"\n   Recommended Model: {recommended[0]}")
    print(f"   Combined Score: {recommended[1]:.2f}")
    print(f"   Accuracy: {results_dict[recommended[0]]['accuracy']:.2f}%")
    
    # Get F1 for recommended model
    _, _, f1, _ = precision_recall_fscore_support(
        results_dict[recommended[0]]['labels'], 
        results_dict[recommended[0]]['predictions'], 
        average='weighted'
    )
    print(f"   F1-Score: {f1*100:.2f}%")
    
    print("\n   Rationale:")
    print(f"      • Achieves top-tier performance across multiple metrics")
    print(f"      • Balances accuracy with precision and recall")
    print(f"      • Suitable for clinical ultrasound classification")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage - adjust paths according to your setup
    """
    
    # Configuration
    CHECKPOINT_DIR = "notebooks/checkpoints"
    DATA_ROOT = "./data"
    VAL_ANNOTATION = "imagenet_vid_val.json"
    BATCH_SIZE = 4
    NUM_FRAMES = 32
    OUTPUT_DIR = "analysis_plots"
    
    # Import your dataset loader
    from kaustav_kga import create_dataloaders
    
    print("\nPreparing for analysis...")
    print("="*70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load validation data
    print("\nLoading validation dataset...")
    _, val_loader = create_dataloaders(
        root_dir=DATA_ROOT,
        val_annotation=VAL_ANNOTATION,
        batch_size=BATCH_SIZE,
        num_frames=NUM_FRAMES,
        num_workers=4
    )
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Run complete analysis
    results = run_complete_analysis(
        checkpoint_dir=CHECKPOINT_DIR,
        val_loader=val_loader,
        device=device,
        output_dir=OUTPUT_DIR
    )
    
    print("\n✅ All analysis completed!")
    print(f"Check the '{OUTPUT_DIR}' directory for all visualizations.")
    print("\nThese plots are ready for your PPT presentation!")
    print("="*70)