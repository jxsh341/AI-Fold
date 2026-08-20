"""AI-Fold v0.1 Latent Diffusion Head

Implements the conditional latent diffusion for generating
future trajectory latents Z_future = [z_1, ..., z_T].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

from aifold.config import ModelConfig


@dataclass
class DiffusionConfig:
    """Configuration for diffusion process."""
    num_steps: int = 32
    sigma_data: float = 1.0
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0  # For EDM-style schedule


class FourierEmbedding(nn.Module):
    """Fixed Fourier feature embedding for noise levels."""
    
    def __init__(self, dim: int = 256):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        # Fixed frequencies (not learned)
        freqs = torch.exp(torch.arange(0, dim // 2) * (-torch.log(torch.tensor(10000.0)) / (dim // 2)))
        self.register_buffer('freqs', freqs)
    
    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """sigma: [..., 1] or [...]
        Returns: [..., dim]"""
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        
        # Log sigma
        log_sigma = torch.log(sigma + 1e-8) * 0.25
        
        # Fourier features
        args = log_sigma * self.freqs
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embedding


class AdaLN(nn.Module):
    """Adaptive LayerNorm (AdaLN-Zero style) for conditioning."""
    
    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        
        # AdaLN parameters: scale, bias, gate
        self.to_scale = nn.Linear(d_cond, d_model)
        self.to_bias = nn.Linear(d_cond, d_model)
        self.to_gate = nn.Linear(d_cond, d_model)
        
        # Zero init for gate (AdaLN-Zero)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.zeros_(self.to_gate.bias)
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_bias.weight)
        nn.init.zeros_(self.to_bias.bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: [..., T, d_model], cond: [..., d_cond]"""
        # Handle cond dimensions - if cond has one fewer dim than x, add time dim
        if cond.dim() == x.dim() - 1:
            cond = cond.unsqueeze(-2)  # [..., 1, d_cond]
        
        x = self.norm(x)
        scale = self.to_scale(cond)
        bias = self.to_bias(cond)
        gate = torch.sigmoid(self.to_gate(cond))
        
        x = (1 + scale) * x + bias
        x = x * gate
        return x


class DiffusionSelfAttention(nn.Module):
    """Self-attention with pair bias for diffusion transformer."""
    
    def __init__(self, d_model: int, d_P: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0
        
        self.adaln = AdaLN(d_model, d_model)  # Condition on single
        self.pair_to_bias = nn.Linear(d_P, num_heads)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(
        self,
        x: torch.Tensor,              # [B, M, T, d_Z] or [B, T, d_Z]
        single_cond: torch.Tensor,    # [B, M, d_H] or [B, d_H]
        pair_cond: torch.Tensor,      # [B, M, N, N, d_P] or [B, N, N, d_P]
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Self-attention over time dimension with pair bias."""
        
        # Handle both [B, M, T, d_Z] and [B, T, d_Z]
        has_samples = x.dim() == 4
        if not has_samples:
            x = x.unsqueeze(1)  # [B, 1, T, d_Z]
            single_cond = single_cond.unsqueeze(1) if single_cond.dim() == 2 else single_cond
            pair_cond = pair_cond.unsqueeze(1) if pair_cond.dim() == 4 else pair_cond
        
        B, M, T, d_Z = x.shape
        residual = x
        
        # Flatten batch and sample dimensions
        x_flat = x.reshape(B * M, T, d_Z)
        cond_flat = single_cond.reshape(B * M, -1)
        
        # AdaLN
        x_flat = self.adaln(x_flat, cond_flat)
        
        # Pair bias from pair conditioning
        # pair_cond: [B, M, N, N, d_P] -> [B*M, num_heads, N, N]
        pair_bias = self.pair_to_bias(pair_cond.reshape(B * M, *pair_cond.shape[2:]))
        pair_bias = pair_bias.permute(0, 3, 1, 2)  # [B*M, H, N, N]
        
        # Self-attention
        qkv = self.qkv(x_flat).reshape(B * M, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # [B*M, T, H, D]
        
        q = q.transpose(1, 2)  # [B*M, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Add pair bias (broadcast over time)
        # pair_bias: [B*M, H, N, N] -> need to match [T, T]
        # For now, we use pair_bias[:, :, :T, :T] assuming N >= T
        if T <= pair_bias.shape[-1]:
            attn = attn + pair_bias[:, :, :T, :T]
        
        if mask is not None:
            # mask: [B, T] -> [B, M, T] -> [B*M, T]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).expand(-1, M, -1)  # [B, M, T]
            mask_flat = mask.reshape(B * M, T)
            attn = attn.masked_fill(~mask_flat.view(B * M, 1, T, 1).bool(), -1e9)
            attn = attn.masked_fill(~mask_flat.view(B * M, 1, 1, T).bool(), -1e9)
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B*M, H, T, D]
        out = out.transpose(1, 2).reshape(B * M, T, d_Z)
        
        out = self.proj(out)
        out = self.dropout(out)
        out = out.reshape(B, M, T, d_Z)
        
        if not has_samples:
            out = out.squeeze(1)
        
        return residual + out


class DiffusionBlock(nn.Module):
    """One diffusion transformer block."""
    
    def __init__(self, d_model: int, d_H: int, d_P: int, num_heads: int):
        super().__init__()
        self.attn = DiffusionSelfAttention(d_model, d_P, num_heads)
        self.ffn_adaln = AdaLN(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        single_cond: torch.Tensor,
        pair_cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.attn(x, single_cond, pair_cond, mask)
        
        # FFN with AdaLN
        residual = x
        has_samples = x.dim() == 4
        if not has_samples:
            x = x.unsqueeze(1)
            single_cond = single_cond.unsqueeze(1) if single_cond.dim() == 2 else single_cond
        
        B, M, T, d_Z = x.shape
        x_flat = x.reshape(B * M, T, d_Z)
        cond_flat = single_cond.reshape(B * M, -1)
        
        x_flat = self.ffn_adaln(x_flat, cond_flat)
        x_flat = self.ffn(x_flat)
        x_flat = x_flat.reshape(B, M, T, d_Z)
        
        if not has_samples:
            x_flat = x_flat.squeeze(1)
        
        return residual + x_flat


class DiffusionTransformer(nn.Module):
    """Conditional diffusion transformer for latent trajectory generation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_Z = config.d_Z
        self.d_H = config.d_H
        self.d_P = config.d_P
        
        # Noise level embedding
        self.noise_embed = nn.Sequential(
            FourierEmbedding(256),
            nn.Linear(256, config.d_Z),
            nn.GELU(),
            nn.Linear(config.d_Z, config.d_Z),
        )
        
        # Horizon embedding
        self.horizon_embed = nn.Embedding(8, config.d_Z)  # T ∈ {1..8}
        
        # Goal/type embedding
        self.goal_embed = nn.Embedding(16, config.d_Z)  # 16 goal types
        
        # Single conditioning projection
        self.single_proj = nn.Sequential(
            nn.Linear(config.d_H, config.d_Z),
            nn.LayerNorm(config.d_Z),
        )
        
        # Pair conditioning projection
        self.pair_proj = nn.Sequential(
            nn.Linear(config.d_P, config.d_P),
            nn.LayerNorm(config.d_P),
        )
        
        # Diffusion blocks
        self.blocks = nn.ModuleList([
            DiffusionBlock(config.d_Z, config.d_H, config.d_P, config.num_diffusion_heads)
            for _ in range(config.num_diffusion_blocks)
        ])
        
        # Output norm and projection
        self.output_norm = nn.LayerNorm(config.d_Z)
        self.output_proj = nn.Linear(config.d_Z, config.d_Z)
    
    def forward(
        self,
        z_noisy: torch.Tensor,              # [B, M, T, d_Z] or [B, T, d_Z]
        sigma: torch.Tensor,                # [B, M] or [B]
        single_cond: torch.Tensor,          # [B, N, d_H]
        pair_cond: torch.Tensor,            # [B, N, N, d_P]
        horizon: Optional[torch.Tensor] = None,
        goal_type: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoise z_noisy given sigma and trunk conditioning."""
        
        B = z_noisy.shape[0]
        has_samples = z_noisy.dim() == 4
        if not has_samples:
            z_noisy = z_noisy.unsqueeze(1)  # [B, 1, T, d_Z]
            M = 1
        else:
            M = z_noisy.shape[1]
        
        
        
        T = z_noisy.shape[2]
        
        # Build conditioning
        # Single cond: project trunk H
        single_cond_proj = self.single_proj(single_cond)  # [B, N, d_Z]
        # Pool to get per-sample conditioning
        single_cond_pool = single_cond_proj.mean(dim=1)  # [B, d_Z]
        single_cond_pool = single_cond_pool.unsqueeze(1).expand(-1, M, -1)  # [B, M, d_Z]
        
        # Pair cond: project trunk P
        pair_cond_proj = self.pair_proj(pair_cond)  # [B, N, N, d_P]
        pair_cond_pool = pair_cond_proj.unsqueeze(1).expand(-1, M, -1, -1, -1)  # [B, M, N, N, d_P]
        
        # Create time mask for diffusion (T time steps)
        if mask is not None and mask.shape[-1] != T:
            # Entity mask [B, N] provided but we need time mask [B, T]
            # Create causal time mask
            device = z_noisy.device
            time_mask = torch.ones(B, T, dtype=torch.bool, device=z_noisy.device)
            # Causal: can only attend to past and current
            for t in range(T):
                time_mask[:, t] = time_mask[:, t] & (torch.arange(T, device=device) <= t)
            mask = time_mask
        elif mask is None:
            # No mask provided, create full mask
            device = z_noisy.device
            mask = torch.ones(B, T, dtype=torch.bool, device=z_noisy.device)
        
        # Noise embedding
        sigma_flat = sigma.reshape(-1)
        noise_emb = self.noise_embed(sigma_flat)  # [B*M, d_Z]
        noise_emb = noise_emb.reshape(B, M, -1)
        
        # Single cond: project trunk H
        single_cond_proj = self.single_proj(single_cond)  # [B, N, d_Z]
        # Pool to get per-sample conditioning
        single_cond_pool = single_cond_proj.mean(dim=1)  # [B, d_Z]
        single_cond_pool = single_cond_pool.unsqueeze(1).expand(-1, M, -1)  # [B, M, d_Z]
        
        # Pair cond: project trunk P
        pair_cond_proj = self.pair_proj(pair_cond)  # [B, N, N, d_P]
        pair_cond_pool = pair_cond_proj.unsqueeze(1).expand(-1, M, -1, -1, -1)  # [B, M, N, N, d_P]
        
        # Noise embedding
        sigma_flat = sigma.reshape(-1)
        noise_emb = self.noise_embed(sigma_flat)  # [B*M, d_Z]
        noise_emb = noise_emb.reshape(B, M, -1)
        
        # Add noise and horizon embeddings to single conditioning
        single_cond_pool = single_cond_pool + noise_emb
        
        if horizon is not None:
            if horizon.dim() == 0:
                horizon = horizon.expand(B)
            horizon_emb = self.horizon_embed(horizon.clamp(0, 7))
            single_cond_pool = single_cond_pool + horizon_emb.unsqueeze(1)
        
        if goal_type is not None:
            if goal_type.dim() == 0:
                goal_type = goal_type.expand(B)
            goal_emb = self.goal_embed(goal_type.clamp(0, 15))
            single_cond_pool = single_cond_pool + goal_emb.unsqueeze(1)
        
        # Diffusion blocks
        x = z_noisy
        for block in self.blocks:
            x = block(x, single_cond_pool, pair_cond_pool, mask)
        
        # Output
        x = self.output_norm(x)
        x = self.output_proj(x)
        
        if not has_samples:
            x = x.squeeze(1)
        
        return x


class LatentDiffusionHead(nn.Module):
    """Latent diffusion head for generating M candidate future trajectories."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.diffusion_transformer = DiffusionTransformer(config)
        
        # Diffusion schedule (EDM-style)
        self.register_buffer('sigma_data', torch.tensor(config.sigma_data))
        self.num_steps = config.num_diffusion_steps
        
        # Precompute noise schedule
        self._init_noise_schedule(config.sigma_min, config.sigma_max, config.rho)
    
    def _init_noise_schedule(self, sigma_min: float, sigma_max: float, rho: float):
        """EDM noise schedule."""
        steps = torch.arange(self.num_steps + 1)
        sigmas = (sigma_max ** (1/rho) + steps / self.num_steps * 
                  (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        self.register_buffer('sigmas', sigmas)
    
    def forward(
        self,
        z_noisy: torch.Tensor,              # [B, M, T, d_Z] or [B, T, d_Z]
        sigma: torch.Tensor,                # [B, M] or [B]
        single_cond: torch.Tensor,          # [B, N, d_H]
        pair_cond: torch.Tensor,            # [B, N, N, d_P]
        horizon: Optional[torch.Tensor] = None,
        goal_type: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoise one step (used during sampling)."""
        return self.diffusion_transformer(
            z_noisy, sigma, single_cond, pair_cond, horizon, goal_type, mask
        )
    
    def _sample_step(
        self,
        z: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        single_cond: torch.Tensor,
        pair_cond: torch.Tensor,
        horizon: Optional[torch.Tensor],
        goal_type: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """One denoising step (2nd order Heun)."""
        
        # EDM-style denoising
        denoised = self.forward(z, sigma, single_cond, pair_cond, horizon, goal_type, mask)
        
# Score estimate: (z - denoised) / sigma
        # sigma: [B, M] -> [B, M, 1, 1] for broadcasting with [B, M, T, d_Z]
        sigma_expanded = sigma.view(*sigma.shape, 1, 1)
        score = (z - denoised) / sigma_expanded
        
        # Euler step
        dt = sigma_next - sigma
        dt_expanded = dt.view(*dt.shape, 1, 1)
        z_next = z + dt_expanded * score
        
        # 2nd order correction
        if (sigma_next > 0).all():
            denoised_next = self.forward(z_next, sigma_next, single_cond, pair_cond, horizon, goal_type, mask)
            sigma_next_expanded = sigma_next.view(*sigma_next.shape, 1, 1)
            score_next = (z_next - denoised_next) / sigma_next_expanded
            dt_expanded = dt.view(*dt.shape, 1, 1)
            z_next = z + 0.5 * dt_expanded * (score + score_next)
        
        # Add noise (for stochastic sampling)
        if (sigma_next > 0).all():
            noise = torch.randn_like(z, generator=generator)
            noise_scale = torch.sqrt(torch.abs(sigma_next**2 - sigma**2))
            noise_scale = noise_scale.view(*noise_scale.shape, 1, 1)
            z_next = z_next + noise_scale * noise
        
        return z_next
    
    def sample(
        self,
        single_cond: torch.Tensor,          # [B, N, d_H]
        pair_cond: torch.Tensor,            # [B, N, N, d_P]
        horizon: Optional[torch.Tensor] = None,
        goal_type: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate M candidate future trajectories via diffusion sampling."""
        
        B = single_cond.shape[0]
        M = num_samples or self.config.num_samples
        T = self.config.horizon_T
        d_Z = self.config.d_Z
        
        # Initial noise
        z = torch.randn(B, M, T, d_Z, device=single_cond.device, generator=generator)
        z = z * self.sigmas[0]  # Scale by initial sigma
        
        # Diffusion loop
        for i in range(self.num_steps):
            sigma = self.sigmas[i].expand(B, M)
            sigma_next = self.sigmas[i + 1].expand(B, M)
            
            z = self._sample_step(
                z, sigma, sigma_next,
                single_cond, pair_cond,
                horizon, goal_type, mask,
                generator
            )
        
        return z  # [B, M, T, d_Z]
    
    def training_step(
        self,
        z_target: torch.Tensor,             # [B, T, d_Z] ground truth
        single_cond: torch.Tensor,          # [B, N, d_H]
        pair_cond: torch.Tensor,            # [B, N, N, d_P]
        horizon: Optional[torch.Tensor] = None,
        goal_type: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute diffusion training loss."""
        
        B, T, d_Z = z_target.shape
        
        # Sample noise level
        t = torch.randint(0, self.num_steps, (B,), device=z_target.device)
        sigma = self.sigmas[t].view(B, 1, 1)  # [B, 1, 1] for [B, T, d_Z]
        
        # Add noise
        noise = torch.randn_like(z_target)
        z_noisy = z_target + sigma * noise
        
        # Denoise
        z_denoised = self.forward(
            z_noisy.unsqueeze(1),  # [B, 1, T, d_Z]
            self.sigmas[t],
            single_cond, pair_cond,
            horizon, goal_type, mask
        ).squeeze(1)
        
        # EDM loss: MSE on denoised output
        # Weight by (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        weight = (sigma.squeeze()**2 + self.sigma_data**2) / (sigma.squeeze() * self.sigma_data)**2
        loss = weight * F.mse_loss(z_denoised, z_target, reduction='none').mean(dim=(1, 2))
        loss = loss.mean()
        
        return loss, {
            'diffusion_loss': loss.item(),
            'sigma_mean': sigma.mean().item(),
        }