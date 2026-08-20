"""AI-Fold v0.1 Core Modules

This module implements the core architectural components:
- EntityEncoder: Encodes typed entities into H vectors
- PairConstructor: Builds initial P from H + explicit relations
- RelationalBlock: The core trunk block with pair-biased attention
- RelationalTrunk: Stack of RelationalBlocks with recycling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

from aifold.config import ModelConfig, EntityTypeConfig, RelationTypeConfig


class EntityEncoder(nn.Module):
    """Encode typed entities into initial H vectors [N, d_H].
    
    Inputs per entity:
    - type_id: int (0-15 for 16 entity types)
    - attributes: float vector [A] (per-type attributes)
    - content: optional text/code [L, d_C]
    - content_mask: bool [L]
    
    Output: H [N, d_H]
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_H = config.d_H
        self.num_types = EntityTypeConfig.NUM_TYPES
        
        # Type embedding
        self.type_embedding = nn.Embedding(self.num_types, config.d_H)
        
        # Attribute projection (from A to d_H)
        self.attribute_proj = nn.Linear(16, config.d_H)  # Assuming A=16
        
        # Attribute encoder (per-type, shared MLP)
        self.attribute_encoder = nn.Sequential(
            nn.Linear(config.d_H, config.d_H * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_H * 2, config.d_H),
        )
        
        # Content encoder (for text/code content)
        self.content_encoder = nn.Sequential(
            nn.Linear(config.d_H, config.d_H * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_H * 2, config.d_H),
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.LayerNorm(config.d_H, eps=config.layer_norm_eps),
            nn.Linear(config.d_H, config.d_H),
        )
        
        # Learnable attribute defaults per type
        self.register_buffer(
            "default_attributes", 
            torch.zeros(self.num_types, config.d_H)
        )
    
    def forward(
        self,
        type_ids: torch.Tensor,                    # [N] int
        attributes: Optional[torch.Tensor] = None,  # [N, A] float
        content: Optional[torch.Tensor] = None,     # [N, L, d_C] float
        content_mask: Optional[torch.Tensor] = None, # [N, L] bool
    ) -> torch.Tensor:
        """Encode entities to H vectors."""
        N = type_ids.shape[0]
        
        # Type embedding
        h = self.type_embedding(type_ids)  # [N, d_H]
        
        # Attributes
        if attributes is not None:
            attr_proj = self.attribute_proj(attributes)  # [N, d_H]
            attr_enc = self.attribute_encoder(attr_proj)
        else:
            # Use per-type learned defaults
            attr_enc = self.default_attributes[type_ids]
        h = h + attr_enc
        
        # Content
        if content is not None and content_mask is not None:
            # Mean pool over content length
            content_enc = self.content_encoder(content)  # [N, L, d_H]
            masked = content_enc * content_mask.unsqueeze(-1).float()
            lengths = content_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
            content_pooled = masked.sum(dim=1) / lengths
            h = h + content_pooled
        
        # Final projection
        h = self.projection(h)
        
        return h


class PairConstructor(nn.Module):
    """Build initial pair representation P from H + explicit relations.
    
    P_ij = L(H_i) + R(H_j) + E_type(rel) + E_temporal(Δt) + E_causal + E_structural
    
    Inputs:
    - H: [N, d_H] entity states
    - relation_types: [N, N] int relation type IDs
    - temporal_offsets: [N, N] int bucketed temporal offsets
    - causal_types: [N, N] int causal relation type IDs (optional)
    - structural_types: [N, N] int structural relation type IDs (optional)
    
    Output: P [N, N, d_P]
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_H = config.d_H
        self.d_P = config.d_P
        
        # Left/right projections for outer sum
        self.left_proj = nn.Linear(config.d_H, config.d_P)
        self.right_proj = nn.Linear(config.d_H, config.d_P)
        
        # Relation type embeddings
        self.relation_embed = nn.Embedding(
            RelationTypeConfig.NUM_TYPES, config.d_P
        )
        
        # Temporal offset embeddings (bucketed)
        self.temporal_embed = nn.Embedding(11, config.d_P)  # -5 to +5
        
        # Causal relation embeddings (optional, separate vocabulary)
        self.causal_embed = nn.Embedding(5, config.d_P)  # 4 causal + none
        
        # Structural relation embeddings (optional)
        self.structural_embed = nn.Embedding(6, config.d_P)  # 5 structural + none
    
    def forward(
        self,
        H: torch.Tensor,                          # [N, d_H]
        relation_types: torch.Tensor,             # [N, N] int
        temporal_offsets: torch.Tensor,           # [N, N] int (bucketed)
        causal_types: Optional[torch.Tensor] = None,
        structural_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N = H.shape[0]
        
        # Outer sum: L(H_i) + R(H_j)
        left = self.left_proj(H)[:, None, :]      # [N, 1, d_P]
        right = self.right_proj(H)[None, :, :]    # [1, N, d_P]
        P = left + right                          # [N, N, d_P]
        
        # Semantic relation embedding
        P = P + self.relation_embed(relation_types)
        
        # Temporal offset embedding
        P = P + self.temporal_embed(temporal_offsets.clamp(-5, 5) + 5)
        
        # Causal (if provided)
        if causal_types is not None:
            P = P + self.causal_embed(causal_types)
        
        # Structural (if provided)
        if structural_types is not None:
            P = P + self.structural_embed(structural_types)
        
        return P


class TransitionBlock(nn.Module):
    """Standard transition block (FFN with GLU) matching AF3 pattern."""
    
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff * 2)  # GLU
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = F.gelu(x) * torch.sigmoid(gate)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class AxialAttention(nn.Module):
    """Axial attention over pair representation (row or column)."""
    
    def __init__(self, d_model: int, num_heads: int, axis: Literal['row', 'col'] = 'row'):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.axis = axis
        
        assert d_model % num_heads == 0
        
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B, N, N, d_P] or [N, N, d_P], mask: [B, N] or [N] or [B, N, N]"""
        # Handle both batched and unbatched
        if x.dim() == 4:
            B, N = x.shape[0], x.shape[1]
            has_batch = True
        else:
            B = 1
            N = x.shape[0]
            has_batch = False
            x = x.unsqueeze(0)  # [1, N, N, d_P]
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        # Ensure mask is [B, N]
        if mask is not None and mask.dim() == 1:
            # Flattened [B*N] -> [B, N]
            mask = mask.view(B, N)
        
        residual = x
        x = self.norm(x)
        
        qkv = self.qkv(x).reshape(B, N, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(3)  # [B, N, N, H, D]
        
        if self.axis == 'row':
            # Attention over columns for each row: [B, N, H, N, D]
            q = q.permute(0, 1, 3, 2, 4)  # [B, N, H, N, D]
            k = k.permute(0, 1, 3, 2, 4)
            v = v.permute(0, 1, 3, 2, 4)
            
            attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            if mask is not None:
                # mask: [B, N] -> [B, N, 1, 1, N]
                mask_expanded = mask.view(B, N, 1, 1, 1).expand(-1, -1, -1, -1, N)
                attn = attn.masked_fill(~mask_expanded.bool(), -1e9)
            
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)  # [B, N, H, N, D]
            out = out.permute(0, 1, 3, 2, 4).reshape(B, N, N, self.d_model)
            
        else:  # column
            # Attention over rows for each column
            q = q.permute(0, 2, 3, 1, 4)  # [B, N, H, N, D]
            k = k.permute(0, 2, 3, 1, 4)
            v = v.permute(0, 2, 3, 1, 4)
            
            attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            if mask is not None:
                # mask: [B, N] -> [B, 1, 1, N, N]
                mask_expanded = mask.view(B, 1, 1, N, 1).expand(-1, -1, -1, -1, N)
                attn = attn.masked_fill(~mask_expanded.bool(), -1e9)
            
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            out = out.permute(0, 3, 1, 2, 4).reshape(B, N, N, self.d_model)
        
        out = self.proj(out)
        out = self.dropout(out)
        
        if not has_batch:
            out = out.squeeze(0)
        
        return residual + out


class SelfAttentionWithPairBias(nn.Module):
    """Entity self-attention with pair bias, matching AF3 pattern."""
    
    def __init__(self, d_H: int, d_P: int, num_heads: int):
        super().__init__()
        self.d_H = d_H
        self.d_P = d_P
        self.num_heads = num_heads
        self.head_dim = d_H // num_heads
        
        assert d_H % num_heads == 0
        
        self.norm = nn.LayerNorm(d_H)
        self.pair_to_bias = nn.Linear(d_P, num_heads)
        self.qkv = nn.Linear(d_H, d_H * 3)
        self.proj = nn.Linear(d_H, d_H)
        self.dropout = nn.Dropout(0.1)
    
    def forward(
        self,
        H: torch.Tensor,           # [B, N, d_H] or [N, d_H]
        P: torch.Tensor,           # [B, N, N, d_P] or [N, N, d_P]
        mask: Optional[torch.Tensor] = None,  # [B, N] or [N]
    ) -> torch.Tensor:
        # Handle both batched and unbatched
        if H.dim() == 3:
            B, N, _ = H.shape
            has_batch = True
        else:
            B = 1
            N = H.shape[0]
            has_batch = False
            H = H.unsqueeze(0)
            P = P.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        residual = H
        H = self.norm(H)
        
        # Pair bias: project pair to attention bias logits
        pair_bias = self.pair_to_bias(P)  # [B, N, N, H]
        pair_bias = pair_bias.permute(0, 3, 1, 2)  # [B, H, N, N]
        
        # Self-attention
        qkv = self.qkv(H).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # [B, N, H, D]
        
        q = q.permute(0, 2, 1, 3)  # [B, H, N, D]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn + pair_bias  # [B, H, N, N]
        
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.view(-1, N)  # Handle flattened mask
            # mask: [B, N] -> [B, 1, 1, N] for query, [B, 1, N, 1] for key
            attn = attn.masked_fill(~mask.view(B, 1, 1, N).bool(), -1e9)
            attn = attn.masked_fill(~mask.view(B, 1, N, 1).bool(), -1e9)
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, N, D]
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.d_H)
        
        out = self.proj(out)
        out = self.dropout(out)
        
        if not has_batch:
            out = out.squeeze(0)
        
        return residual + out


class TriRelBlock(nn.Module):
    """Optional TriRel block for Experiment H.
    
    P_ij += f(P_ik, P_kj) aggregated over k
    """
    
    def __init__(self, d_P: int):
        super().__init__()
        self.d_P = d_P
        # Bilinear combination: P_ik W P_kj
        self.W = nn.Parameter(torch.randn(d_P, d_P) * 0.02)
        self.norm = nn.LayerNorm(d_P)
        self.proj = nn.Linear(d_P, d_P)
    
    def forward(self, P: torch.Tensor) -> torch.Tensor:
        """P: [N, N, d_P]"""
        N = P.shape[0]
        P_norm = self.norm(P)
        
        # P_ik @ W @ P_kj^T aggregated over k
        # [N, N, d_P] @ [d_P, d_P] @ [N, N, d_P]^T
        # Efficient: (P @ W) @ P.transpose(-2, -1) but we need per-pair
        
        # For each i,j: sum_k P_ik W P_kj^T
        # This is: (P @ W) @ P.transpose(-2, -1) = [N, N, N] - too large
        # Use low-rank approximation or chunking
        
        # Simple approximation: mean over k of P_ik * P_kj
        P_left = P_norm[:, :, None, :]   # [N, N, 1, d_P]
        P_right = P_norm[None, :, :, :]  # [1, N, N, d_P]
        
        # Element-wise product + project
        combined = P_left * P_right      # [N, N, N, d_P]
        combined = combined.mean(dim=2)  # [N, N, d_P] average over k
        
        update = self.proj(combined)
        return P + update


class RelationalBlock(nn.Module):
    """One block of the relational trunk (AF3 PairFormer style)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_H = config.d_H
        self.d_P = config.d_P
        self.stochastic_depth_prob = config.stochastic_depth_prob
        
        # Pair updates
        self.pair_row_attn = AxialAttention(config.d_P, config.num_heads, axis='row')
        self.pair_col_attn = AxialAttention(config.d_P, config.num_heads, axis='col')
        self.pair_transition = TransitionBlock(config.d_P)
        
        # Entity updates (conditioned on pair)
        self.entity_attn = SelfAttentionWithPairBias(config.d_H, config.d_P, config.num_heads)
        self.entity_transition = TransitionBlock(config.d_H)
        
        # Optional TriRel (Experiment H)
        self.trirel = TriRelBlock(config.d_P) if config.use_trirel else None
    
    def forward(
        self,
        H: torch.Tensor,           # [N, d_H]
        P: torch.Tensor,           # [N, N, d_P]
        mask: Optional[torch.Tensor] = None,  # [N]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Stochastic depth: randomly skip block during training
        if self.training and self.stochastic_depth_prob > 0:
            if torch.rand(1).item() < self.stochastic_depth_prob:
                return H, P  # Skip this block entirely
        
        # Pair updates
        P = self.pair_row_attn(P, mask)
        P = self.pair_col_attn(P, mask)
        P = self.pair_transition(P)
        
        # Optional TriRel
        if self.trirel is not None:
            P = self.trirel(P)
        
        # Entity updates (with pair bias)
        H = self.entity_attn(H, P, mask)
        H = self.entity_transition(H)
        
        return H, P


class RelationalTrunk(nn.Module):
    """Full relational trunk with recycling."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            RelationalBlock(config) for _ in range(config.num_relational_blocks)
        ])
    
    def forward(
        self,
        H: torch.Tensor,           # [N, d_H]
        P: torch.Tensor,           # [N, N, d_P]
        mask: Optional[torch.Tensor] = None,  # [N]
        prev_H: Optional[torch.Tensor] = None,  # For recycling
        prev_P: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass through trunk."""
        
        # Add recycled states if provided (zero-init on first pass)
        if prev_H is not None:
            H = H + prev_H
        if prev_P is not None:
            P = P + prev_P
        
        for block in self.blocks:
            H, P = block(H, P, mask)
        
        return H, P
    
    def recycle(
        self,
        H: torch.Tensor,
        P: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run recycling loop."""
        prev_H = torch.zeros_like(H)
        prev_P = torch.zeros_like(P)
        
        for _ in range(self.config.num_recycles):
            H, P = self.forward(H, P, mask, prev_H, prev_P)
            prev_H = H
            prev_P = P
        
        return H, P