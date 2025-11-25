from __future__ import annotations

from dataclasses import dataclass

import torch

from config import CONFIG


@dataclass(slots=True)
class SparseAutoencoder:
    """
    Wrapper for Sparse Autoencoder from Gemma Scope.
    
    Uses pretrained SAEs from: https://huggingface.co/google/gemma-scope
    """
    encoder_weight: torch.Tensor  # [n_features, d_model]
    decoder_weight: torch.Tensor  # [n_features, d_model]
    bias: torch.Tensor  # [n_features]

    @classmethod
    def load(cls, hidden_size: int | None = None) -> "SparseAutoencoder":
        """
        Load pretrained SAE from Gemma Scope using sae-lens library.
        
        Falls back to random initialization if sae-lens is not installed
        or if loading fails (for development/testing only).
        
        Args:
            hidden_size: Model hidden size (auto-detected if None)
        """
        try:
            from sae_lens import SAE
            
            print(f"Loading SAE from Gemma Scope...")
            print(f"  Release: {CONFIG.model.sae_release}")
            print(f"  SAE ID: {CONFIG.model.sae_id}")
            
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release=CONFIG.model.sae_release,
                sae_id=CONFIG.model.sae_id,
                device="cpu",
            )
            
            print(f"✓ SAE loaded successfully!")
            print(f"  Features: {sae.cfg.d_sae}")
            print(f"  Hidden size: {sae.cfg.d_in}")
            print(f"  Sparsity: {sparsity}")
            
            # Verify dimension match
            if hidden_size is not None and sae.cfg.d_in != hidden_size:
                print(f"⚠️  WARNING: SAE hidden size ({sae.cfg.d_in}) != model hidden size ({hidden_size})")
                print(f"    This SAE was trained on a different model variant.")
                print(f"    Falling back to random SAE with correct dimensions.")
                return cls._create_random_sae(hidden_size=hidden_size)
            
            return cls(
                encoder_weight=sae.W_enc.detach().cpu(),
                decoder_weight=sae.W_dec.detach().cpu().t(),  # Transpose to [n_features, d_model]
                bias=sae.b_enc.detach().cpu() if hasattr(sae, "b_enc") else torch.zeros(sae.cfg.d_sae),
            )
            
        except ImportError:
            print("⚠️  sae-lens not installed.")
            print("    Run: pip install sae-lens")
            print("    Falling back to random SAE (FOR TESTING ONLY)")
            return cls._create_random_sae(hidden_size=hidden_size)
            
        except Exception as e:
            print(f"⚠️  Failed to load SAE from Gemma Scope: {e}")
            print("    Falling back to random SAE (FOR TESTING ONLY)")
            return cls._create_random_sae(hidden_size=hidden_size)
    
    @classmethod
    def _create_random_sae(cls, hidden_size: int | None = None, n_features: int = 16384) -> "SparseAutoencoder":
        """Create random SAE for testing pipeline without real checkpoint."""
        if hidden_size is None:
            hidden_size = 2048  # Default for Gemma-2B
        print(f"    Creating random SAE: {hidden_size} → {n_features} features")
        print(f"    ⚠️  THIS IS A RANDOM SAE - NOT USEFUL FOR REAL EXPERIMENTS!")
        encoder_weight = torch.randn(n_features, hidden_size) * 0.02
        decoder_weight = torch.randn(n_features, hidden_size) * 0.02
        bias = torch.zeros(n_features)
        return cls(encoder_weight=encoder_weight, decoder_weight=decoder_weight, bias=bias)

    @torch.inference_mode()
    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Encode residual activations to sparse SAE codes.
        
        Args:
            residual: [batch, d_model] or [batch, seq, d_model]
        
        Returns:
            codes: [batch, n_features] or [batch, seq, n_features]
        """
        original_shape = residual.shape
        original_dtype = residual.dtype
        
        # Convert to float32 for SAE encoding (SAE weights are in float32)
        residual = residual.float()
        
        if residual.dim() == 3:
            residual = residual.reshape(-1, residual.size(-1))
        
        projected = torch.nn.functional.linear(residual, self.encoder_weight, self.bias)
        codes = torch.nn.functional.relu(projected)
        
        if len(original_shape) == 3:
            codes = codes.reshape(original_shape[0], original_shape[1], -1)
        
        return codes

    def decode_feature(self, feature_index: int) -> torch.Tensor:
        """Get decoder direction for a specific feature."""
        return self.decoder_weight[feature_index]

    def build_direction(self, indices: list[int], weights: list[float]) -> torch.Tensor:
        """
        Build steering direction from weighted combination of features.
        
        Args:
            indices: Feature indices to combine
            weights: Weight for each feature
        
        Returns:
            direction: [d_model] steering vector
        """
        direction = torch.zeros_like(self.decoder_weight[0])
        for idx, weight in zip(indices, weights):
            direction += weight * self.decode_feature(idx)
        return direction
