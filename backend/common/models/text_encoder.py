"""Text encoder models for both Ultra-Light and Balanced modes."""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


class TextEncoder(nn.Module):
    """Base text encoder class."""
    
    def __init__(self, mode: str, num_classes: int = 5):
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
    
    def forward(self, text):
        raise NotImplementedError


class UltraLightTextEncoder(TextEncoder):
    """
    Ultra-Light text encoder using pretrained SentenceTransformer models.
    
    Supports:
    - sentence-transformers/* models (use SentenceTransformer directly)
    - VoVanPhuc/sup-SimCSE-* (Vietnamese SimCSE, use SentenceTransformer)
    - vinai/phobert-base (NOT recommended - not sentence embedding model)
    """
    
    def __init__(
        self, 
        num_classes: int = 5, 
        embedding_dim: int = 384, 
        hidden_dim: int = 256,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    ):
        super().__init__(mode='ultra_light', num_classes=num_classes)
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Detect model type
        # SimCSE models and sentence-transformers can use SentenceTransformer
        self._use_sentence_transformer = (
            'sentence-transformers' in model_name.lower() or
            'simcse' in model_name.lower() or
            'paraphrase' in model_name.lower()
        )
        
        # Load appropriate encoder
        if self._use_sentence_transformer:
            # SentenceTransformer models (best for sentence embeddings)
            self.encoder = SentenceTransformer(model_name)
            self.tokenizer = None
        else:
            # Fallback to AutoModel (for raw transformers like PhoBERT)
            # WARNING: Raw PhoBERT is NOT optimized for sentence embeddings!
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze pretrained encoder (only train classifier)
        self.freeze_encoder()
        
        # Trainable classifier head (optional)
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = None
    
    def freeze_encoder(self):
        """Freeze pretrained encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self, num_layers: int = 2):
        """Unfreeze encoder for fine-tuning (optional)."""
        if not self._use_sentence_transformer:
            # Unfreeze last N transformer layers for AutoModel
            for layer in self.encoder.encoder.layer[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            # Unfreeze for SentenceTransformer
            for param in self.encoder.parameters():
                param.requires_grad = True
    
    def forward(self, texts):
        """
        Args:
            texts: List of strings or single string
        Returns:
            logits: (batch_size, num_classes) if classifier exists, else embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Get embeddings based on model type
        if self._use_sentence_transformer:
            with torch.no_grad():
                embeddings = self.encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        else:
            embeddings = self._encode_automodel(texts)
        
        # Classify if classifier exists
        if self.classifier is not None:
            logits = self.classifier(embeddings)
            return logits
        else:
            return embeddings
    
    def _encode_automodel(self, texts):
        """Encode texts using AutoModel (e.g., raw PhoBERT) with mean pooling."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        # Move to same device as model
        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.encoder(**inputs)
        
        # Use MEAN pooling instead of CLS token (better for sentence embeddings)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        # Mean pooling: sum token embeddings weighted by attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return embeddings
    
    def get_embeddings(self, texts):
        """Get text embeddings without classification."""
        if isinstance(texts, str):
            texts = [texts]
        
        if self._use_sentence_transformer:
            with torch.no_grad():
                embeddings = self.encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            return embeddings
        else:
            with torch.no_grad():
                return self._encode_automodel(texts)


class BalancedTextEncoder(TextEncoder):
    """Balanced text encoder using pretrained PhoBERT (frozen backbone)."""
    
    def __init__(self, num_classes: int = 5, embedding_dim: int = 768, hidden_dim: int = 512):
        super().__init__(mode='balanced', num_classes=num_classes)
        
        # Load PRETRAINED PhoBERT
        self.model_name = 'vinai/phobert-base'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        
        self.embedding_dim = embedding_dim
        
        # Freeze pretrained encoder (only train classifier)
        self.freeze_encoder()
        
        # Trainable classifier head (optional - only if num_classes provided)
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = None
    
    def freeze_encoder(self):
        """Freeze pretrained PhoBERT backbone."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self, num_layers: int = 2):
        """Unfreeze last N layers for fine-tuning (optional)."""
        # Unfreeze last N transformer layers
        for layer in self.encoder.encoder.layer[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def forward(self, texts):
        """
        Args:
            texts: List of strings or single string
        Returns:
            logits: (batch_size, num_classes) if classifier exists, else embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        # Move to same device as model
        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        outputs = self.encoder(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Classify if classifier exists
        if self.classifier is not None:
            logits = self.classifier(embeddings)
            return logits
        else:
            # Return embeddings directly if no classifier
            return embeddings
    
    def get_embeddings(self, texts):
        """Get text embeddings without classification."""
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings


def get_text_encoder(mode: str, num_classes: int = 5, config: dict = None):
    """Factory function to get text encoder based on mode."""
    if mode == 'ultra_light':
        embedding_dim = config.get('text_embedding_dim', 384) if config else 384
        hidden_dim = config.get('hidden_dim', 256) if config else 256
        model_name = config.get('text_encoder', 'sentence-transformers/all-MiniLM-L6-v2') if config else 'sentence-transformers/all-MiniLM-L6-v2'
        return UltraLightTextEncoder(num_classes, embedding_dim, hidden_dim, model_name)
    elif mode == 'balanced':
        embedding_dim = config.get('text_embedding_dim', 768) if config else 768
        hidden_dim = config.get('hidden_dim', 512) if config else 512
        return BalancedTextEncoder(num_classes, embedding_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown mode: {mode}")
