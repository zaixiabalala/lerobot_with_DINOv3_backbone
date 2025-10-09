"""DINOv3 Encoder wrapper for ACT policy."""

import torch
import torch.nn as nn

from lerobot.policies.act.dino.config import *
from transformers import AutoModel, AutoImageProcessor

class DINOEncoder(nn.Module):
    """
    DINOv3 encoder that wraps the pretrained model and provides
    an interface compatible with ACT's ResNet backbone.
    
    Returns output in the same format as IntermediateLayerGetter:
    {"feature_map": tensor of shape (B, C, H, W)}
    """
    
    def __init__(self, model_name: str, output_dim: int = 512, freeze: bool = True, model_dir: str = DINOV3_LOCATION):
        """
        Args:
            model_name: Name of the DINOv3 model (e.g., "dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16")
            output_dim: Output dimension after projection
            freeze: Whether to freeze the DINO backbone weights
            model_dir: Directory of the DINOv3 model
        """
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.freeze = freeze
        self.model_dir = model_dir
        
        assert model_name in MODEL_TO_NUM_LAYERS, f"Model name {model_name} not in {MODEL_TO_NUM_LAYERS}"

        # Load pretrained DINOv3 model
        if "dinov3" in model_name:
            # Load from torch hub or local path
            # Adjust the repo path according to your DINOv3 setup
            try:
                # Try loading from local path first (if you have cloned the repo)
                self.dino = AutoModel.from_pretrained(self.model_dir)
           
            except Exception as e:
                raise RuntimeError(
                        f"Failed to load DINOv3 model '{model_name}'. "
                        f"Please ensure the model is available. Error: {e}"
                    )
        
        # Freeze DINO backbone if requested
        if freeze:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()
        
        self.n_layers = MODEL_TO_NUM_LAYERS[self.model_name]
        # Projection layer to match output dimension
        # Use Conv2d to match ResNet's output format
        self.projection = nn.Conv2d(MODEL_TO_HIDDEN_DIM[self.model_name], output_dim, kernel_size=1)
        
        # Store feature dimension for compatibility with ACT code
        self.fc = nn.Module()  # Dummy module for compatibility
        self.fc.in_features = output_dim
        
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input images of shape (B, C, H, W)
            
        Returns:
            Dictionary with "feature_map" key containing tensor of shape (B, output_dim, h, w)
        """
        B, C, H, W = x.shape
        
        # Forward through DINO with appropriate gradient tracking
        if self.freeze:
            with torch.no_grad():
                outputs = self.dino(pixel_values=x, output_hidden_states=True)
                hidden_states = outputs.last_hidden_state
                patch_features = hidden_states[:, 5:, :]  # (B*T, num_patches, dinov3_dim)
        else:
            raise NotImplementedError("DINOv3 encoder does not support training")
        
        patch = getattr(self.dino.config, "patch_size", 16)
        H = x.shape[-2] // patch
        W = x.shape[-1] // patch
        assert H * W == patch_features.shape[1], "grid size mismatch"

        B, N, C = patch_features.shape  # C 应该是 768
        feat_map = patch_features.transpose(1, 2).reshape(B, C, H, W)     # (B, 768, H, W)
        
        # Project to target dimension
        feature_map = self.projection(feat_map)  # (B, num_patches, output_dim)
        
        # Return in the same format as IntermediateLayerGetter
        return {"feature_map": feature_map}
    
    def train(self, mode: bool = True):
        """Override train mode to keep DINO frozen if requested."""
        super().train(mode)
        if self.freeze:
            self.dino.eval()
        return self
