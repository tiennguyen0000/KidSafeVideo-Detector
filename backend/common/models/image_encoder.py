"""Image encoder models for both Ultra-Light and Balanced modes."""
import torch
import torch.nn as nn
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np


class ImageEncoder(nn.Module):
    """Base image encoder class."""
    
    def __init__(self, mode: str, num_classes: int = 5):
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
    
    def forward(self, images):
        raise NotImplementedError


class UltraLightImageEncoder(ImageEncoder):
    """Ultra-Light image encoder using pretrained EfficientNet-B0 (frozen backbone).
    
    EfficientNet-B0 outputs 1280d embeddings, which are reduced to 512d via dim_reducer.
    
    IMPORTANT: dim_reducer uses DETERMINISTIC initialization to ensure
    consistent embeddings across different encoder instances (e.g., Spark workers).
    """
    
    def __init__(self, num_classes: int = 5, embedding_dim: int = 512, hidden_dim: int = 256):
        super().__init__(mode='ultra_light', num_classes=num_classes)
        
        # Load PRETRAINED EfficientNet-B0 (outputs 1280d)
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Replace classifier with identity
        self.encoder._fc = nn.Identity()
        
        # Freeze pretrained backbone
        self.freeze_encoder()
        
        # Dimension reducer: 1280 → 512 (same as balanced mode)
        # Use DETERMINISTIC initialization for consistent embeddings across workers
        self.dim_reducer = nn.Sequential(
            nn.Linear(1280, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )
        self._init_dim_reducer_deterministic()
        
        # FREEZE dim_reducer to ensure consistent embeddings
        for param in self.dim_reducer.parameters():
            param.requires_grad = False
        
        self.embedding_dim = embedding_dim  # Output dim after reducer
        
        # Trainable custom classifier (optional - only if num_classes provided)
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = None
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _init_dim_reducer_deterministic(self):
        """Initialize dim_reducer with DETERMINISTIC weights.
        
        This ensures all encoder instances (e.g., across Spark workers)
        produce identical embeddings for the same input.
        
        Uses orthogonal initialization with fixed seed for reproducibility.
        """
        # Use fixed seed for reproducibility across all workers
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)  # Fixed seed
        
        # Initialize Linear layer with orthogonal initialization
        linear = self.dim_reducer[0]
        nn.init.orthogonal_(linear.weight)
        nn.init.zeros_(linear.bias)
        
        # Initialize LayerNorm
        layer_norm = self.dim_reducer[1]
        nn.init.ones_(layer_norm.weight)
        nn.init.zeros_(layer_norm.bias)
        
        # Restore RNG state
        torch.set_rng_state(rng_state)
    
    def freeze_encoder(self):
        """Freeze pretrained EfficientNet backbone."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self, num_blocks: int = 2):
        """Unfreeze last N blocks for fine-tuning (optional)."""
        # Unfreeze last N blocks of EfficientNet
        total_blocks = len(self.encoder._blocks)
        for block in self.encoder._blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
    
    def _get_device(self):
        """Get device from encoder (not classifier which may be None)."""
        return next(self.encoder.parameters()).device
    
    def preprocess_images(self, images):
        """
        Preprocess images for the model.
        Args:
            images: List of PIL Images or numpy arrays or tensor
        Returns:
            Tensor of shape (batch_size, 3, 224, 224)
        """
        if isinstance(images, torch.Tensor):
            return images
        
        if not isinstance(images, list):
            images = [images]
        
        processed = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif isinstance(img, str):
                img = Image.open(img).convert('RGB')
            
            processed.append(self.transform(img))
        
        return torch.stack(processed)
    
    def forward(self, images):
        """
        Args:
            images: Batch of images (batch_size, 3, 224, 224) or list of images
        Returns:
            logits: (batch_size, num_classes) if classifier exists, else embeddings
        """
        if not isinstance(images, torch.Tensor):
            images = self.preprocess_images(images)
        
        device = self._get_device()
        images = images.to(device)
        
        # Get raw embeddings from encoder (1280d)
        raw_embeddings = self.encoder(images)
        
        # Apply dimension reducer (1280d → 512d)
        embeddings = self.dim_reducer(raw_embeddings)
        
        # Classify if classifier exists
        if self.classifier is not None:
            logits = self.classifier(embeddings)
            return logits
        else:
            # Return embeddings directly if no classifier
            return embeddings
    
    def get_embeddings(self, images):
        """Get image embeddings without classification."""
        if not isinstance(images, torch.Tensor):
            images = self.preprocess_images(images)
        
        device = self._get_device()
        images = images.to(device)
        
        with torch.no_grad():
            raw_embeddings = self.encoder(images)
            embeddings = self.dim_reducer(raw_embeddings)
        
        return embeddings


class BalancedImageEncoder(ImageEncoder):
    """Balanced image encoder using pretrained ResNet50 (frozen backbone).
    
    ResNet50 outputs 2048d embeddings directly (no dimension reduction).
    This preserves maximum information from the pretrained model.
    
    The fusion module will handle the dimension mismatch with text (768d).
    """
    
    def __init__(self, num_classes: int = 5, embedding_dim: int = 2048, hidden_dim: int = 512):
        super().__init__(mode='balanced', num_classes=num_classes)
        
        # Load PRETRAINED ResNet50 (outputs 2048d)
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder.fc = nn.Identity()
        
        # Freeze ResNet backbone completely
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # No dim_reducer - output raw 2048d for maximum information
        # The fusion module will project this to the fused dimension
        self.embedding_dim = 2048  # Raw ResNet50 output
        
        # Optional dim_reducer for backward compatibility (only if embedding_dim != 2048)
        if embedding_dim != 2048:
            self.dim_reducer = nn.Sequential(
                nn.Linear(2048, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU()
            )
            self._init_dim_reducer_deterministic(embedding_dim)
            for param in self.dim_reducer.parameters():
                param.requires_grad = False
            self.embedding_dim = embedding_dim
        else:
            self.dim_reducer = None
        
        # Trainable classifier (optional - only if num_classes provided)
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(self.embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = None
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _init_dim_reducer_deterministic(self, embedding_dim: int):
        """Initialize dim_reducer with DETERMINISTIC weights.
        
        This ensures all encoder instances (e.g., across Spark workers)
        produce identical embeddings for the same input.
        
        Uses orthogonal initialization with fixed seed for reproducibility.
        """
        # Use fixed seed for reproducibility across all workers
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)  # Fixed seed
        
        # Initialize Linear layer with orthogonal initialization
        linear = self.dim_reducer[0]
        nn.init.orthogonal_(linear.weight)
        nn.init.zeros_(linear.bias)
        
        # Initialize LayerNorm
        layer_norm = self.dim_reducer[1]
        nn.init.ones_(layer_norm.weight)
        nn.init.zeros_(layer_norm.bias)
        
        # Restore RNG state
        torch.set_rng_state(rng_state)
    
    def freeze_encoder(self):
        """Freeze pretrained ResNet backbone."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self, num_layers: int = 2):
        """Unfreeze last N ResNet layers for fine-tuning (optional)."""
        for layer in list(self.encoder.children())[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def _get_device(self):
        """Get device from encoder (not classifier which may be None)."""
        return next(self.encoder.parameters()).device
    
    def preprocess_images(self, images):
        """Preprocess images for the model."""
        if isinstance(images, torch.Tensor):
            return images
        
        if not isinstance(images, list):
            images = [images]
        
        processed = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif isinstance(img, str):
                img = Image.open(img).convert('RGB')
            
            processed.append(self.preprocess(img))
        
        return torch.stack(processed)
    
    def forward(self, images):
        """
        Args:
            images: Batch of images or list of images
        Returns:
            logits: (batch_size, num_classes) if classifier exists, else embeddings
        """
        if not isinstance(images, torch.Tensor):
            images = self.preprocess_images(images)
        
        device = self._get_device()
        images = images.to(device)
        
        # Get embeddings from ResNet50 (2048d)
        with torch.no_grad():
            embeddings = self.encoder(images)
        
        # Apply dim_reducer if exists (backward compatibility)
        if self.dim_reducer is not None:
            embeddings = self.dim_reducer(embeddings)
        
        # Classify if classifier exists
        if self.classifier is not None:
            logits = self.classifier(embeddings)
            return logits
        else:
            return embeddings
    
    def get_embeddings(self, images):
        """Get image embeddings (2048d or reduced dim) without classification."""
        if not isinstance(images, torch.Tensor):
            images = self.preprocess_images(images)
        
        device = self._get_device()
        images = images.to(device)
        
        # Get embeddings from ResNet50 (2048d)
        with torch.no_grad():
            embeddings = self.encoder(images)
            
            # Apply dim_reducer if exists
            if self.dim_reducer is not None:
                embeddings = self.dim_reducer(embeddings)
        
        return embeddings


def get_image_encoder(mode: str, num_classes: int = 5, config: dict = None):
    """Factory function to get image encoder based on mode."""
    if mode == 'ultra_light':
        # Note: EfficientNet-B0 outputs 1280d, use raw output for cache compatibility
        embedding_dim = config.get('image_embedding_dim', 1280) if config else 1280
        hidden_dim = config.get('hidden_dim', 256) if config else 256
        return UltraLightImageEncoder(num_classes, embedding_dim, hidden_dim)
    elif mode == 'balanced':
        # Note: ResNet50 outputs 2048d, reduced to embedding_dim (default 512) via dim_reducer
        embedding_dim = config.get('image_embedding_dim', 512) if config else 512
        hidden_dim = config.get('hidden_dim', 512) if config else 512
        return BalancedImageEncoder(num_classes, embedding_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown mode: {mode}")
