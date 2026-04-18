"""Vision tower shared by dense and MoE Qwen3-VL variants.

Matches HF Qwen3VLVisionModel:
- forward(hidden_states: (seq_len, C*T*H*W), grid_thw: (N_images, 3)) -> (pooler_output, last_hidden_state, deepstack_features)
- patch_embed: Conv3d (custom CUDA path)
- pos_embed: bilinear interpolation over a learned (num_positions, hidden) grid, based on (h, w) per image
- rotary_pos_emb: 2D grid coordinates with spatial_merge awareness
- blocks: 27× (LayerNorm + attention with full non-causal + LayerNorm + MLP)
- deepstack mergers: use_postshuffle_norm=True
- merger: use_postshuffle_norm=False
"""
from __future__ import annotations

import torch
from torch import nn

from cuda_qwen3_vl.configs import VisionConfig
from cuda_qwen3_vl.modules import (
    CudaVisionBlock,
    CudaVisionPatchEmbed,
    CudaVisionPatchMerger,
)


class CudaVisionTower(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.spatial_merge_size = cfg.spatial_merge_size
        self.patch_size = cfg.patch_size
        self.num_grid_per_side = int(cfg.num_position_embeddings ** 0.5)

        self.patch_embed = CudaVisionPatchEmbed(
            in_channels=cfg.in_channels,
            hidden_size=cfg.hidden_size,
            patch_size=cfg.patch_size,
            temporal_patch_size=cfg.temporal_patch_size,
        )

        # HF uses nn.Embedding directly at `pos_embed` (not wrapped).
        # We keep the CudaEmbedding module but expose its `.weight` at `pos_embed.emb.weight`
        # via the loader name remap. (kept for compat with load_hf_weights.)
        from cuda_qwen3_vl.modules.embedding import CudaEmbedding
        self.pos_embed = _PosEmbedWrap(cfg.num_position_embeddings, cfg.hidden_size)

        # Rotary (match HF exactly): Qwen3VLVisionRotaryEmbedding(dim = head_dim // 2) has
        # inv_freq = 1/theta^(arange(0,dim,2)/dim). Its forward(max_hw) returns (max_hw, dim//2)
        # via outer(seq, inv_freq). Then rot_pos_emb gathers a (total, 2, dim//2) tensor and
        # flattens to (total, dim). Finally the vision model cats(rot, rot, -1) → (total, 2*dim) = (total, head_dim).
        head_dim = cfg.hidden_size // cfg.num_heads
        self.head_dim = head_dim
        self.rotary_dim = head_dim // 2           # HF's init dim for rotary
        half = self.rotary_dim // 2               # inv_freq length
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        assert inv_freq.numel() == half, f"inv_freq len mismatch {inv_freq.numel()} vs {half}"
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.blocks = nn.ModuleList([
            CudaVisionBlock(cfg.hidden_size, cfg.num_heads, cfg.intermediate_size, eps=cfg.rms_norm_eps)
            for _ in range(cfg.num_layers)
        ])
        self.merger = CudaVisionPatchMerger(
            hidden_size=cfg.hidden_size,
            out_hidden_size=cfg.out_hidden_size,
            spatial_merge_size=cfg.spatial_merge_size,
            use_postshuffle_norm=False,
        )
        self.deepstack_mergers = nn.ModuleList([
            CudaVisionPatchMerger(
                hidden_size=cfg.hidden_size,
                out_hidden_size=cfg.out_hidden_size,
                spatial_merge_size=cfg.spatial_merge_size,
                use_postshuffle_norm=True,
            )
            for _ in cfg.deepstack_layers
        ])

    # ---- Helpers mirroring HF ----

    def _freq_table(self, max_hw: int, device: torch.device) -> torch.Tensor:
        """Return (max_hw, rotary_dim // 2) table of outer(seq, inv_freq). Matches HF."""
        seq = torch.arange(max_hw, device=device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq.to(device))

    def _rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Match HF's rot_pos_emb: returns (total_tokens, rotary_dim) where rotary_dim = head_dim // 2.

        The returned tensor has halves [row_freqs, col_freqs] (each of length rotary_dim // 2)
        gathered from the 1D freq_table by 2D grid coordinates. Vision model then concats
        with itself to reach (total_tokens, head_dim) for cos/sin.
        """
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()
        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self._freq_table(max_hw, grid_thw.device)  # (max_hw, rot_half)

        total = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total, 2), dtype=torch.long, device=grid_thw.device)

        offset = 0
        for t, h, w in grid_thw_list:
            merged_h, merged_w = h // merge_size, w // merge_size
            block_rows = torch.arange(merged_h, device=grid_thw.device)
            block_cols = torch.arange(merged_w, device=grid_thw.device)
            intra_row = torch.arange(merge_size, device=grid_thw.device)
            intra_col = torch.arange(merge_size, device=grid_thw.device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if t > 1:
                coords = coords.repeat(t, 1)
            n = coords.shape[0]
            pos_ids[offset:offset + n] = coords
            offset += n

        embeddings = freq_table[pos_ids]       # (total, 2, rot_half)
        embeddings = embeddings.flatten(1)     # (total, rot_half * 2) == (total, head_dim)
        return embeddings

    def _fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Match HF's fast_pos_embed_interpolate: bilinear lookup from a 48×48 learned grid,
        then spatial-merge permute to tokenization order. Returns (total_tokens, hidden)."""
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()
        grid_ts = [r[0] for r in grid_thw_list]
        grid_hs = [r[1] for r in grid_thw_list]
        grid_ws = [r[2] for r in grid_thw_list]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for _t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_fl = h_idxs.int(); w_fl = w_idxs.int()
            h_cl = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_cl = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_fl; dw = w_idxs - w_fl
            base_h = h_fl * self.num_grid_per_side
            base_h_c = h_cl * self.num_grid_per_side
            ids = [
                (base_h[None].T + w_fl[None]).flatten(),
                (base_h[None].T + w_cl[None]).flatten(),
                (base_h_c[None].T + w_fl[None]).flatten(),
                (base_h_c[None].T + w_cl[None]).flatten(),
            ]
            ws = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(ids[i].tolist())
                weight_list[i].extend(ws[i].tolist())

        idx_t = torch.tensor(idx_list, dtype=torch.long, device=device)
        w_t = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos = self.pos_embed.emb(idx_t).to(device) * w_t[:, :, None]
        patches = pos[0] + pos[1] + pos[2] + pos[3]                # (sum(h*w), hidden)
        patches = patches.split([h * w for h, w in zip(grid_hs, grid_ws)])

        out = []
        for emb, t, h, w in zip(patches, grid_ts, grid_hs, grid_ws):
            emb = emb.repeat(t, 1)
            emb = (
                emb.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            out.append(emb)
        return torch.cat(out)

    # ---- Main forward (matches HF Qwen3VLVisionModel.forward) ----

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        hidden_states: (seq_len, C * T_patch * H_patch * W_patch)   -- pre-flattened patches from HF processor
        grid_thw: (N_images, 3) with (t, h, w) per image in patch units

        Returns:
          pooler_output        (N_merged_tokens, out_hidden)   -- merger applied
          last_hidden_state    (seq_len, hidden)               -- pre-merger hidden states
          deepstack_features   list[(N_merged_tokens, out_hidden)]  -- one per deepstack layer
        """
        x = self.patch_embed(hidden_states)                       # (seq_len, hidden)
        pos = self._fast_pos_embed_interpolate(grid_thw).to(x.dtype)
        x = x + pos

        rot = self._rot_pos_emb(grid_thw)                         # (seq_len, rotary_dim) where rotary_dim = head_dim // 2
        emb = torch.cat((rot, rot), dim=-1).to(x.dtype)            # (seq_len, head_dim)
        cos = emb.cos().unsqueeze(0)                               # (1, seq_len, head_dim)
        sin = emb.sin().unsqueeze(0)

        # Add batch dim: our blocks take (B, S, H) input.
        x = x.unsqueeze(0)
        deepstack_outs: list[torch.Tensor] = []
        deepstack_idx = {layer_idx: i for i, layer_idx in enumerate(self.cfg.deepstack_layers)}

        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin)
            if i in deepstack_idx:
                ds_i = deepstack_idx[i]
                deepstack_outs.append(self.deepstack_mergers[ds_i](x.squeeze(0)))

        last_hidden_state = x.squeeze(0)
        pooler_output = self.merger(last_hidden_state)
        return pooler_output, last_hidden_state, deepstack_outs


class _PosEmbedWrap(nn.Module):
    """Keeps our `pos_embed.emb.weight` state_dict key (so loader mapping matches)."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        from cuda_qwen3_vl.modules.embedding import CudaEmbedding
        self.emb = CudaEmbedding(num_embeddings, embedding_dim)

    @property
    def weight(self) -> torch.Tensor:
        return self.emb.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids)
