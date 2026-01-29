"""
Visualization utilities for analyzing advanced fusion behavior.

This script helps understand:
1. How gating mechanism works (which modality is trusted more)
2. Attention patterns between image and text
3. Feature space analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_gate_weights(
    model,
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    labels: List[str],
    save_path: str = None
):
    """
    Visualize gate weights from GatedFusionClassifier.
    
    Shows how much the model trusts image vs text for each sample.
    
    Args:
        model: GatedFusionClassifier instance
        img_emb: Image embeddings (B, D_img)
        txt_emb: Text embeddings (B, D_txt)
        labels: Ground truth labels for each sample
        save_path: Path to save figure (optional)
    """
    model.eval()
    
    with torch.no_grad():
        # Get gate weights
        gate_weights = model.get_gate_weights(img_emb, txt_emb)  # (B, d_fused)
        
        # Average across dimensions to get scalar per sample
        gate_avg = gate_weights.mean(dim=1).cpu().numpy()  # (B,)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # ========================================================================
    # Plot 1: Gate weights distribution by class
    # ========================================================================
    ax = axes[0]
    
    # Group by label
    label_names = ['Safe', 'Aggressive', 'Sexual', 'Superstition']
    gate_by_label = {label: [] for label in label_names}
    
    for i, label in enumerate(labels):
        if label in gate_by_label:
            gate_by_label[label].append(gate_avg[i])
    
    # Box plot
    data_to_plot = [gate_by_label[label] for label in label_names if gate_by_label[label]]
    labels_to_plot = [label for label in label_names if gate_by_label[label]]
    
    box = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
    
    # Color boxes
    colors = ['green', 'red', 'orange', 'purple', 'blue']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Balanced (0.5)')
    ax.set_ylabel('Gate Weight (0=Text, 1=Image)', fontsize=12)
    ax.set_title('Gate Weights Distribution by Class', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotations
    ax.text(0.02, 0.98, '↑ Higher = Trust Image More', 
            transform=ax.transAxes, fontsize=10, va='top', color='blue')
    ax.text(0.02, 0.02, '↓ Lower = Trust Text More', 
            transform=ax.transAxes, fontsize=10, va='bottom', color='red')
    
    # ========================================================================
    # Plot 2: Individual sample gate weights
    # ========================================================================
    ax = axes[1]
    
    # Sort by gate weight for better visualization
    sorted_indices = np.argsort(gate_avg)
    gate_sorted = gate_avg[sorted_indices]
    labels_sorted = [labels[i] for i in sorted_indices]
    
    # Color by label
    color_map = {
        'Safe': 'green',
        'Aggressive': 'red',
        'Sexual': 'purple',
        'Superstition': 'blue'
    }
    colors = [color_map.get(label, 'gray') for label in labels_sorted]
    
    x = np.arange(len(gate_sorted))
    ax.bar(x, gate_sorted, color=colors, alpha=0.6)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Sample Index (sorted by gate weight)', fontsize=12)
    ax.set_ylabel('Gate Weight', fontsize=12)
    ax.set_title('Individual Sample Gate Weights', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[label], alpha=0.6, label=label) 
                      for label in label_names]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved: {save_path}")
    
    plt.show()
    
    # Statistics
    logger.info("=" * 60)
    logger.info("GATE WEIGHT STATISTICS")
    logger.info("=" * 60)
    for label in label_names:
        if gate_by_label[label]:
            weights = np.array(gate_by_label[label])
            logger.info(f"{label:15} | Mean: {weights.mean():.3f} | Std: {weights.std():.3f}")
    logger.info("=" * 60)


def visualize_embedding_space(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    h_fused: torch.Tensor,
    labels: List[str],
    save_path: str = None
):
    """
    Visualize embedding spaces using t-SNE or PCA.
    
    Shows how embeddings are distributed before and after fusion.
    
    Args:
        img_emb: Image embeddings (B, D_img)
        txt_emb: Text embeddings (B, D_txt)
        h_fused: Fused embeddings (B, d_fused)
        labels: Ground truth labels
        save_path: Path to save figure
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # Convert to numpy
    img_emb_np = img_emb.cpu().numpy()
    txt_emb_np = txt_emb.cpu().numpy()
    h_fused_np = h_fused.cpu().numpy()
    
    # Reduce to 2D using PCA first (for speed), then t-SNE
    logger.info("Computing t-SNE projections...")
    
    # PCA to 50D first (speeds up t-SNE)
    if img_emb_np.shape[1] > 50:
        pca_img = PCA(n_components=50).fit_transform(img_emb_np)
    else:
        pca_img = img_emb_np
    
    if txt_emb_np.shape[1] > 50:
        pca_txt = PCA(n_components=50).fit_transform(txt_emb_np)
    else:
        pca_txt = txt_emb_np
    
    if h_fused_np.shape[1] > 50:
        pca_fused = PCA(n_components=50).fit_transform(h_fused_np)
    else:
        pca_fused = h_fused_np
    
    # t-SNE to 2D
    tsne_img = TSNE(n_components=2, random_state=42).fit_transform(pca_img)
    tsne_txt = TSNE(n_components=2, random_state=42).fit_transform(pca_txt)
    tsne_fused = TSNE(n_components=2, random_state=42).fit_transform(pca_fused)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Color mapping
    label_names = ['Safe', 'Aggressive', 'Sexual', 'Superstition']
    color_map = {
        'Safe': 'green',
        'Aggressive': 'red',
        'Sexual': 'purple',
        'Superstition': 'blue'
    }
    colors = [color_map.get(label, 'gray') for label in labels]
    
    # Plot 1: Image embeddings
    ax = axes[0]
    for label in label_names:
        mask = [l == label for l in labels]
        if any(mask):
            ax.scatter(tsne_img[mask, 0], tsne_img[mask, 1], 
                      c=color_map[label], label=label, alpha=0.6, s=50)
    ax.set_title('Image Embedding Space', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Text embeddings
    ax = axes[1]
    for label in label_names:
        mask = [l == label for l in labels]
        if any(mask):
            ax.scatter(tsne_txt[mask, 0], tsne_txt[mask, 1], 
                      c=color_map[label], label=label, alpha=0.6, s=50)
    ax.set_title('Text Embedding Space', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Fused embeddings
    ax = axes[2]
    for label in label_names:
        mask = [l == label for l in labels]
        if any(mask):
            ax.scatter(tsne_fused[mask, 0], tsne_fused[mask, 1], 
                      c=color_map[label], label=label, alpha=0.6, s=50)
    ax.set_title('Fused Embedding Space', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved: {save_path}")
    
    plt.show()


def analyze_prediction_confidence(
    model,
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    labels: List[str],
    save_path: str = None
):
    """
    Analyze prediction confidence distribution.
    
    Args:
        model: Fusion model
        img_emb: Image embeddings
        txt_emb: Text embeddings
        labels: Ground truth labels
        save_path: Path to save figure
    """
    model.eval()
    
    with torch.no_grad():
        logits_stage1, logits_stage2, h_fused = model(img_emb, txt_emb)
        
        # Get probabilities
        probs_stage1 = torch.softmax(logits_stage1, dim=1)
        probs_stage2 = torch.softmax(logits_stage2, dim=1)
        
        # Get confidence (max probability)
        conf_stage1, pred_stage1 = probs_stage1.max(dim=1)
        conf_stage2, pred_stage2 = probs_stage2.max(dim=1)
    
    conf_stage1 = conf_stage1.cpu().numpy()
    conf_stage2 = conf_stage2.cpu().numpy()
    pred_stage1 = pred_stage1.cpu().numpy()
    
    # Ground truth binary labels
    y_stage1 = np.array([0 if label == 'Safe' else 1 for label in labels])
    
    # Correctness
    correct_stage1 = (pred_stage1 == y_stage1)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Confidence distribution for correct vs incorrect
    ax = axes[0]
    
    conf_correct = conf_stage1[correct_stage1]
    conf_incorrect = conf_stage1[~correct_stage1]
    
    ax.hist(conf_correct, bins=20, alpha=0.6, color='green', label='Correct', density=True)
    ax.hist(conf_incorrect, bins=20, alpha=0.6, color='red', label='Incorrect', density=True)
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Stage 1 Confidence Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Confidence by class
    ax = axes[1]
    
    label_names = ['Safe', 'Aggressive', 'Sexual', 'Superstition']
    conf_by_label = {label: [] for label in label_names}
    
    for i, label in enumerate(labels):
        if label in conf_by_label:
            conf_by_label[label].append(conf_stage1[i])
    
    data_to_plot = [conf_by_label[label] for label in label_names if conf_by_label[label]]
    labels_to_plot = [label for label in label_names if conf_by_label[label]]
    
    box = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
    
    colors = ['green', 'red', 'orange', 'purple', 'blue']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('Confidence by Class', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved: {save_path}")
    
    plt.show()
    
    # Statistics
    logger.info("=" * 60)
    logger.info("CONFIDENCE STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Correct predictions:   Mean={conf_correct.mean():.3f}, Std={conf_correct.std():.3f}")
    logger.info(f"Incorrect predictions: Mean={conf_incorrect.mean():.3f}, Std={conf_incorrect.std():.3f}")
    logger.info("=" * 60)


def example_visualization():
    """Example of how to use visualization functions."""
    logger.info("=" * 80)
    logger.info("FUSION VISUALIZATION EXAMPLE")
    logger.info("=" * 80)
    
    # Generate dummy data
    batch_size = 100
    d_img = 1280
    d_txt = 384
    
    img_emb = torch.randn(batch_size, d_img)
    txt_emb = torch.randn(batch_size, d_txt)
    
    # Generate dummy labels
    label_names = ['Safe', 'Aggressive', 'Sexual', 'Superstition']
    labels = np.random.choice(label_names, size=batch_size).tolist()
    
    # Create model
    from common.models.advanced_fusion import GatedFusionClassifier
    
    model = GatedFusionClassifier(d_img=d_img, d_txt=d_txt, d_fused=256)
    
    # Get fused embeddings
    with torch.no_grad():
        _, _, h_fused = model(img_emb, txt_emb)
    
    # Visualizations
    logger.info("\n1. Visualizing gate weights...")
    visualize_gate_weights(model, img_emb, txt_emb, labels, save_path='gate_weights.png')
    
    logger.info("\n2. Visualizing embedding spaces...")
    visualize_embedding_space(img_emb, txt_emb, h_fused, labels, save_path='embedding_space.png')
    
    logger.info("\n3. Analyzing prediction confidence...")
    analyze_prediction_confidence(model, img_emb, txt_emb, labels, save_path='confidence.png')
    
    logger.info("\n✅ Visualization complete!")


if __name__ == '__main__':
    example_visualization()

