"""
Advanced Fusion Modules for Multimodal Video Classification.

This module implements two fusion strategies:
1. GatedFusionClassifier: Lightweight gated fusion for local training (16GB RAM)
2. AttentionFusionClassifier: Attention-based fusion for GPU training (Colab)

Direct 4-class classification: Safe, Aggressive, Sexual, Superstition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

# Label definitions
LABELS = ['Safe', 'Aggressive', 'Sexual', 'Superstition']
NUM_CLASSES = 4


class GatedFusionClassifier(nn.Module):
    """
    Lightweight Gated Multimodal Unit (GMU) Fusion for local training.
    
    This uses a gating mechanism to dynamically weight image vs text modalities
    based on the input. The gate learns when to trust image more vs text more.
    
    Architecture:
        1. Project img_emb and txt_emb to common dimension d_fused
        2. Apply tanh activation to get hidden representations
        3. Compute gate z from concatenated projections
        4. Fused = z * h_img + (1 - z) * h_txt
        5. Direct 4-class classification
    
    Memory efficient: ~10-20MB parameters for typical configs.
    """
    
    def __init__(
        self,
        d_img: int,
        d_txt: int,
        d_fused: int = 256,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        use_layernorm: bool = True
    ):
        """
        Args:
            d_img: Dimension of image embeddings
            d_txt: Dimension of text embeddings
            d_fused: Dimension of fused representation (default: 256)
            num_classes: Number of output classes (default: 4)
            dropout: Dropout probability (default: 0.3)
            use_layernorm: Whether to use layer normalization (default: True)
                          LayerNorm is preferred over BatchNorm for small batch sizes
        """
        super().__init__()
        
        self.d_img = d_img
        self.d_txt = d_txt
        self.d_fused = d_fused
        self.num_classes = num_classes
        
        # === PROJECTION LAYERS ===
        # Project image embeddings to common dimension
        self.img_proj = nn.Sequential(
            nn.Linear(d_img, d_fused),
            nn.LayerNorm(d_fused) if use_layernorm else nn.BatchNorm1d(d_fused),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Project text embeddings to common dimension
        self.txt_proj = nn.Sequential(
            nn.Linear(d_txt, d_fused),
            nn.LayerNorm(d_fused) if use_layernorm else nn.BatchNorm1d(d_fused),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # === GATING MECHANISM ===
        # Gate network: takes concatenated projections and outputs gate weight
        self.gate = nn.Sequential(
            nn.Linear(d_fused * 2, d_fused),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_fused, d_fused),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # === CLASSIFIER HEAD ===
        # Direct 4-class classification: Safe, Aggressive, Sexual, Superstition
        self.classifier = nn.Sequential(
            nn.Linear(d_fused, d_fused // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_fused // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        img_emb: torch.Tensor, 
        txt_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with gated fusion.
        
        Args:
            img_emb: Image embeddings (B, D_img)
            txt_emb: Text embeddings (B, D_txt)
        
        Returns:
            logits: Classification logits (B, num_classes)
            h_fused: Fused hidden representation (B, d_fused) for analysis
        """
        batch_size = img_emb.size(0)
        
        # Step 1: Project both modalities to common dimension with activation
        h_img = self.img_proj(img_emb)  # (B, d_fused)
        h_txt = self.txt_proj(txt_emb)  # (B, d_fused)
        
        # Step 2: Compute gate from concatenated projections
        # Gate decides how much to trust image vs text
        concat = torch.cat([h_img, h_txt], dim=1)  # (B, 2*d_fused)
        z = self.gate(concat)  # (B, d_fused), values in [0, 1]
        
        # Step 3: Gated fusion
        # z close to 1 = trust image more
        # z close to 0 = trust text more
        h_fused = z * h_img + (1 - z) * h_txt  # (B, d_fused)
        
        # Step 4: Classify
        logits = self.classifier(h_fused)  # (B, num_classes)
        
        return logits, h_fused
    
    def get_gate_weights(
        self, 
        img_emb: torch.Tensor, 
        txt_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Get gate weights for analysis (how much model trusts image vs text).
        
        Returns:
            gate_weights: (B, d_fused) gate values in [0, 1]
                         Average across d_fused to get scalar per sample
        """
        with torch.no_grad():
            # Project
            h_img = self.img_proj(img_emb)
            h_txt = self.txt_proj(txt_emb)
            # Compute gate
            concat = torch.cat([h_img, h_txt], dim=1)
            z = self.gate(concat)  # (B, d_fused)
            return z


class AttentionFusionClassifier(nn.Module):
    """
    Attention-Based Fusion for GPU training (Colab).
    
    This uses cross-attention mechanism where image and text attend to each other,
    allowing for more complex interaction between modalities.
    
    Architecture:
        1. Project img_emb and txt_emb to common dimension d_fused
        2. Add positional embeddings to distinguish modalities
        3. Cross-attention: Image attends to Text, Text attends to Image
        4. Combine with residual connections
        5. Feed-forward refinement
        6. Direct 4-class classification
    
    More powerful but heavier: ~30-50MB parameters for typical configs.
    """
    
    def __init__(
        self,
        d_img: int,
        d_txt: int,
        d_fused: int = 512,
        num_classes: int = NUM_CLASSES,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.3,
        ff_dim_multiplier: int = 2
    ):
        """
        Args:
            d_img: Dimension of image embeddings
            d_txt: Dimension of text embeddings
            d_fused: Dimension of fused representation (default: 512)
            num_classes: Number of output classes (default: 4)
            num_heads: Number of attention heads (default: 8)
            num_layers: Number of cross-attention layers (default: 2)
            dropout: Dropout probability (default: 0.3)
            ff_dim_multiplier: Feed-forward dimension multiplier (default: 2)
        """
        super().__init__()
        
        self.d_img = d_img
        self.d_txt = d_txt
        self.d_fused = d_fused
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        assert d_fused % num_heads == 0, "d_fused must be divisible by num_heads"
        
        # === PROJECTION LAYERS ===
        self.img_proj = nn.Linear(d_img, d_fused)
        self.txt_proj = nn.Linear(d_txt, d_fused)
        
        # === MODALITY EMBEDDINGS ===
        # Learnable embeddings to distinguish image vs text modality
        self.img_modality_emb = nn.Parameter(torch.randn(1, d_fused))
        self.txt_modality_emb = nn.Parameter(torch.randn(1, d_fused))
        
        # === CROSS-ATTENTION LAYERS ===
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=d_fused,
                num_heads=num_heads,
                dropout=dropout,
                ff_dim=d_fused * ff_dim_multiplier
            )
            for _ in range(num_layers)
        ])
        
        # === FUSION AGGREGATION ===
        # Aggregate image and text representations after cross-attention
        self.fusion_agg = nn.Sequential(
            nn.Linear(d_fused * 2, d_fused),
            nn.LayerNorm(d_fused),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # === CLASSIFIER HEAD ===
        # Direct 4-class classification: Safe, Aggressive, Sexual, Superstition
        self.classifier = nn.Sequential(
            nn.Linear(d_fused, d_fused // 2),
            nn.LayerNorm(d_fused // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_fused // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize modality embeddings
        nn.init.normal_(self.img_modality_emb, std=0.02)
        nn.init.normal_(self.txt_modality_emb, std=0.02)
    
    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention-based fusion.
        
        Args:
            img_emb: Image embeddings (B, D_img)
            txt_emb: Text embeddings (B, D_txt)
        
        Returns:
            logits: Classification logits (B, num_classes)
            h_fused: Fused hidden representation (B, d_fused) for analysis
        """
        batch_size = img_emb.size(0)
        
        # Step 1: Project to common dimension
        h_img = self.img_proj(img_emb)  # (B, d_fused)
        h_txt = self.txt_proj(txt_emb)  # (B, d_fused)
        
        # Step 2: Add modality embeddings
        h_img = h_img + self.img_modality_emb  # (B, d_fused)
        h_txt = h_txt + self.txt_modality_emb  # (B, d_fused)
        
        # Step 3: Prepare for cross-attention
        # Add sequence dimension: (B, 1, d_fused)
        h_img = h_img.unsqueeze(1)  # (B, 1, d_fused)
        h_txt = h_txt.unsqueeze(1)  # (B, 1, d_fused)
        
        # Step 4: Apply cross-attention layers
        for layer in self.cross_attention_layers:
            h_img, h_txt = layer(h_img, h_txt)
        
        # Step 5: Remove sequence dimension
        h_img = h_img.squeeze(1)  # (B, d_fused)
        h_txt = h_txt.squeeze(1)  # (B, d_fused)
        
        # Step 6: Aggregate both modalities
        h_concat = torch.cat([h_img, h_txt], dim=1)  # (B, 2*d_fused)
        h_fused = self.fusion_agg(h_concat)  # (B, d_fused)
        
        # Step 7: Classify
        logits = self.classifier(h_fused)  # (B, num_classes)
        
        return logits, h_fused


class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention Layer: Image attends to Text, Text attends to Image.
    
    This allows bidirectional interaction between modalities.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ff_dim: int = 1024
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Image-to-Text attention (Image queries, Text keys/values)
        self.img2txt_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.img2txt_norm = nn.LayerNorm(d_model)
        
        # Text-to-Image attention (Text queries, Image keys/values)
        self.txt2img_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.txt2img_norm = nn.LayerNorm(d_model)
        
        # Feed-forward networks
        self.img_ff = self._build_ff(d_model, ff_dim, dropout)
        self.txt_ff = self._build_ff(d_model, ff_dim, dropout)
        
        self.img_ff_norm = nn.LayerNorm(d_model)
        self.txt_ff_norm = nn.LayerNorm(d_model)
    
    def _build_ff(self, d_model: int, ff_dim: int, dropout: float):
        """Build feed-forward network."""
        return nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        h_img: torch.Tensor,
        h_txt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention forward pass.
        
        Args:
            h_img: Image hidden states (B, L_img, d_model)
            h_txt: Text hidden states (B, L_txt, d_model)
        
        Returns:
            h_img_out: Updated image hidden states (B, L_img, d_model)
            h_txt_out: Updated text hidden states (B, L_txt, d_model)
        """
        # Image attends to Text
        img_attn_out, _ = self.img2txt_attn(
            query=h_img,
            key=h_txt,
            value=h_txt
        )
        h_img = self.img2txt_norm(h_img + img_attn_out)  # Residual + LayerNorm
        
        # Text attends to Image
        txt_attn_out, _ = self.txt2img_attn(
            query=h_txt,
            key=h_img,
            value=h_img
        )
        h_txt = self.txt2img_norm(h_txt + txt_attn_out)  # Residual + LayerNorm
        
        # Feed-forward for Image
        img_ff_out = self.img_ff(h_img)
        h_img = self.img_ff_norm(h_img + img_ff_out)
        
        # Feed-forward for Text
        txt_ff_out = self.txt_ff(h_txt)
        h_txt = self.txt_ff_norm(h_txt + txt_ff_out)
        
        return h_img, h_txt


# ============================================================================
# LOSS FUNCTION FOR 4-CLASS CLASSIFICATION
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    This loss focuses training on hard examples by down-weighting easy examples.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor (default: 1.0)
            gamma: Focusing parameter (default: 2.0). Higher gamma = more focus on hard examples
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) - logits
            targets: (B,) - class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.1,
    class_weights: torch.Tensor = None,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0
) -> Dict[str, torch.Tensor]:
    """
    Compute cross-entropy or focal loss for 4-class classification.
    
    Args:
        logits: (B, 4) - Classification logits
        labels: (B,) - Class labels (0-3)
        label_smoothing: Label smoothing factor (default: 0.1) - helps prevent overfitting
        class_weights: Optional class weights for imbalanced data
        use_focal_loss: Whether to use focal loss instead of CE
        focal_gamma: Gamma parameter for focal loss (default: 2.0)
    
    Returns:
        Dict containing:
            - loss: Total loss
            - accuracy: Batch accuracy
    """
    if use_focal_loss:
        # Focal loss with label smoothing
        # First apply label smoothing to cross entropy, then focal weighting
        num_classes = logits.size(-1)
        if label_smoothing > 0:
            # Smooth labels: (1 - smoothing) for correct class, smoothing/(num_classes-1) for others
            smooth_labels = torch.zeros_like(logits).scatter_(
                1, labels.unsqueeze(1), 1.0
            )
            smooth_labels = smooth_labels * (1 - label_smoothing) + label_smoothing / num_classes
            # Compute cross entropy with soft labels
            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = -(smooth_labels * log_probs).sum(dim=-1)  # (B,)
        else:
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Apply focal weighting
        probs = torch.softmax(logits, dim=-1)
        p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # Probability of correct class
        focal_weight = (1 - p_t) ** focal_gamma
        loss = (focal_weight * ce_loss).mean()
    else:
        # Standard cross-entropy with optional class weights and label smoothing
        loss = F.cross_entropy(
            logits, 
            labels,
            weight=class_weights,
            label_smoothing=label_smoothing
        )
    
    # Compute accuracy
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == labels).float().mean()
    
    return {
        'loss': loss,
        'accuracy': accuracy.item()
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example of how to use the fusion modules."""
    
    print("=" * 80)
    print("EXAMPLE USAGE OF ADVANCED FUSION MODULES")
    print("=" * 80)
    
    # Hyperparameters
    batch_size = 16
    d_img = 1280  # EfficientNet-B0 embedding dimension
    d_txt = 384   # SentenceTransformer embedding dimension
    
    # Create dummy data
    img_emb = torch.randn(batch_size, d_img)
    txt_emb = torch.randn(batch_size, d_txt)
    
    # Labels for two-stage training
    y_stage1 = torch.randint(0, 2, (batch_size,))  # 0: Safe, 1: Unsafe
    y_stage2 = torch.randint(0, 4, (batch_size,))  # 0-3 for harmful types
    mask_stage2 = y_stage1 == 1  # Only Unsafe samples
    
    print(f"\nBatch size: {batch_size}")
    print(f"Image embedding dim: {d_img}")
    print(f"Text embedding dim: {d_txt}")
    print(f"Unsafe samples: {mask_stage2.sum().item()}/{batch_size}")
    
    # ========================================================================
    # 1. GATED FUSION (Lightweight for local)
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. GATED FUSION CLASSIFIER (Lightweight)")
    print("=" * 80)
    
    gated_model = GatedFusionClassifier(
        d_img=d_img,
        d_txt=d_txt,
        d_fused=256,
        dropout=0.3
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in gated_model.parameters())
    print(f"Number of parameters: {num_params:,} (~{num_params/1e6:.2f}M)")
    
    # Forward pass
    logits_stage1, logits_stage2, h_fused = gated_model(img_emb, txt_emb)
    
    print(f"Output shapes:")
    print(f"  - logits_stage1: {logits_stage1.shape}  (Binary: Safe vs Unsafe)")
    print(f"  - logits_stage2: {logits_stage2.shape}  (Multiclass: 4 harmful types)")
    print(f"  - h_fused: {h_fused.shape}  (Fused representation)")
    
    # Compute loss
    loss_dict = compute_two_stage_loss(
        logits_stage1, logits_stage2,
        y_stage1, y_stage2, mask_stage2
    )
    
    print(f"\nLoss breakdown:")
    print(f"  - Total loss: {loss_dict['loss']:.4f}")
    print(f"  - Stage 1 loss: {loss_dict['loss_stage1']:.4f}")
    print(f"  - Stage 2 loss: {loss_dict['loss_stage2']:.4f}")
    
    # Analyze gate weights
    gate_weights = gated_model.get_gate_weights(img_emb, txt_emb)
    avg_gate = gate_weights.mean(dim=1)  # Average across dimensions
    print(f"\nGate weights (avg): {avg_gate[:5].tolist()}")
    print(f"  → Values close to 1.0 = Trust image more")
    print(f"  → Values close to 0.0 = Trust text more")
    
    # ========================================================================
    # 2. ATTENTION FUSION (Powerful for GPU)
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. ATTENTION FUSION CLASSIFIER (Powerful)")
    print("=" * 80)
    
    attn_model = AttentionFusionClassifier(
        d_img=d_img,
        d_txt=d_txt,
        d_fused=512,
        num_heads=8,
        num_layers=2,
        dropout=0.3
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in attn_model.parameters())
    print(f"Number of parameters: {num_params:,} (~{num_params/1e6:.2f}M)")
    
    # Forward pass
    logits_stage1, logits_stage2, h_fused = attn_model(img_emb, txt_emb)
    
    print(f"Output shapes:")
    print(f"  - logits_stage1: {logits_stage1.shape}  (Binary: Safe vs Unsafe)")
    print(f"  - logits_stage2: {logits_stage2.shape}  (Multiclass: 4 harmful types)")
    print(f"  - h_fused: {h_fused.shape}  (Fused representation)")
    
    # Compute loss
    loss_dict = compute_two_stage_loss(
        logits_stage1, logits_stage2,
        y_stage1, y_stage2, mask_stage2
    )
    
    print(f"\nLoss breakdown:")
    print(f"  - Total loss: {loss_dict['loss']:.4f}")
    print(f"  - Stage 1 loss: {loss_dict['loss_stage1']:.4f}")
    print(f"  - Stage 2 loss: {loss_dict['loss_stage2']:.4f}")
    
    # ========================================================================
    # 3. COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    gated_params = sum(p.numel() for p in gated_model.parameters())
    attn_params = sum(p.numel() for p in attn_model.parameters())
    
    print(f"{'Model':<25} {'Parameters':<15} {'Recommended For'}")
    print("-" * 80)
    print(f"{'GatedFusionClassifier':<25} {gated_params:>10,} {'Local (16GB RAM, CPU/GPU)'}")
    print(f"{'AttentionFusionClassifier':<25} {attn_params:>10,} {'Colab (GPU)'}")
    print("-" * 80)
    print(f"Ratio: {attn_params/gated_params:.2f}x more parameters in Attention model")
    
    print("\n" + "=" * 80)
    print("TRAINING TIPS")
    print("=" * 80)
    print("""
1. For GatedFusionClassifier:
   - Learning rate: 1e-4 to 5e-4
   - Batch size: 16-32
   - Suitable for CPU or low-end GPU
   - Fast convergence (~10-15 epochs)

2. For AttentionFusionClassifier:
   - Learning rate: 1e-4 to 3e-4 (lower than gated)
   - Batch size: 8-16 (more memory intensive)
   - Requires GPU (preferably 8GB+ VRAM)
   - May need more epochs (~15-25)

3. Two-Stage Training:
   - Stage1 weight: 1.0 (balanced with stage2)
   - Stage2 weight: 1.0
   - Adjust if one stage is harder to learn
   - Monitor both losses separately

4. Label Smoothing:
   - Use 0.0 for small datasets (<1000 samples)
   - Use 0.1 for large datasets (>10000 samples)
   - Helps prevent overconfidence
    """)


if __name__ == '__main__':
    example_usage()

