"""
models.py

Model definitions for the proposed spatial transcriptomics framework.

Public-release note:
    This file provides the model architecture interfaces and reusable baseline
    components. Several core cross-interaction modules are intentionally kept as
    lightweight placeholders. Full implementation will be released after formal
    publication / acceptance.
"""

from __future__ import annotations

from typing import Optional, Dict, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution

# =========================================================
# Main proposed model
# =========================================================

class CIGST(nn.Module):
    """
    CIGST: Cross-Interaction Graph Learning for Spatial Transcriptomics.

    The public skeleton shows the high-level data flow:
        1. spatial graph branch
        2. feature / image branch
        3. cross-modal interaction module
        4. spot-level fused embedding
        5. ZINB reconstruction head

    Core interaction details are intentionally omitted in this public version.
    """

    def __init__(
        self,
        nfeat: int,
        nhid1: int,
        nhid2: int,
        dropout: float,
        use_image: bool = False,
        image_in_channels: int = 3,
        mode: str = "graph_only",
    ):
        super().__init__()

        self.use_image = use_image
        self.mode = mode
        self.embed_dim = nhid2

        # Dual graph branches
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)

        # Optional image branch
        self.CNN = CNNEncoder(in_channels=image_in_channels, out_dim=nhid2)

        # Proposed interaction modules, exposed as interfaces only
        self.enc_spatial = CLSEncoder(dim=nhid2, nhead=4, dropout=dropout)
        self.enc_feature = CLSEncoder(dim=nhid2, nhead=4, dropout=dropout)
        self.cross_s_to_e = CrossDecoder(dim=nhid2, nhead=4, dropout=dropout)
        self.cross_e_to_s = CrossDecoder(dim=nhid2, nhead=4, dropout=dropout)
        self.fusion = LowRankBilinearFusion(
            dim_x=nhid2,
            dim_y=nhid2,
            dim_out=nhid2,
            rank=4,
            dropout=dropout,
        )

        # Public heads
        self.spot_head = nn.Linear(nhid2 * 2, nhid2)
        self.global_head = nn.Linear(nhid2 * 2, nhid2)
        self.ZINB = ZINBDecoder(nfeat, nhid1, nhid2)
        self.dropout = dropout

    def _get_feature_branch(
        self,
        x: torch.Tensor,
        fadj: Optional[torch.Tensor],
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Select expression / image / hybrid branch representation."""
        if self.mode == "graph_only":
            if fadj is None:
                raise ValueError("mode='graph_only' requires fadj.")
            z_e = self.FGCN(x, fadj)
        elif self.mode == "graph_image":
            if image is None:
                raise ValueError("mode='graph_image' requires image input.")
            z_e = self.CNN(image)
        elif self.mode == "graph_feature_image":
            if fadj is None or image is None:
                raise ValueError("mode='graph_feature_image' requires both fadj and image.")
            z_feat = self.FGCN(x, fadj)
            z_img = self.CNN(image)
            z_e = 0.5 * (z_feat + z_img)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return z_e

    def forward(
        self,
        x: torch.Tensor,
        sadj: torch.Tensor,
        fadj: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        """
        Forward pass.

        Args:
            x: gene expression feature matrix, shape (N, nfeat)
            sadj: spatial graph adjacency
            fadj: feature graph adjacency
            image: optional image patch tensor, shape (N, C, H, W)
            return_dict: whether to return a dictionary

        Returns:
            Dictionary or tuple containing fused embedding and ZINB parameters.
        """
        z_s = self.SGCN(x, sadj)
        z_e = self._get_feature_branch(x, fadj, image=image)

        # The following interaction process is intentionally abstracted.
        z_f, global_fused = self._cross_interaction_fusion(z_s, z_e)

        pi, disp, mean = self.ZINB(z_f)

        if return_dict:
            return {
                "emb": z_f,
                "z_s": z_s,
                "z_e": z_e,
                "z_f": z_f,
                "global_fused": global_fused,
                "pi": pi,
                "disp": disp,
                "mean": mean,
            }

        return z_f, pi, disp, mean, z_s, z_e, global_fused

    def _cross_interaction_fusion(
        self,
        z_s: torch.Tensor,
        z_e: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal interaction and low-rank fusion interface.

        This function is the core part of CIGST. The detailed implementation is
        omitted in this public skeleton.
        """
        raise NotImplementedError(
            "_cross_interaction_fusion is part of the proposed method and is omitted in the public skeleton."
        )

# =========================================================
# Basic graph encoder
# =========================================================

class GCN(nn.Module):
    """Two-layer graph convolutional encoder."""

    def __init__(self, nfeat: int, nhid: int, out: int, dropout: float):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


# =========================================================
# ZINB decoder
# =========================================================

class ZINBDecoder(nn.Module):
    """
    ZINB decoder for reconstructing gene expression distribution parameters.

    Args:
        nfeat: number of genes / input features
        nhid1: hidden dimension
        input_dim: embedding dimension
    """

    def __init__(self, nfeat: int, nhid1: int, input_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, nhid1),
            nn.BatchNorm1d(nhid1),
            nn.ReLU(),
        )
        self.pi = nn.Linear(nhid1, nfeat)
        self.disp = nn.Linear(nhid1, nfeat)
        self.mean = nn.Linear(nhid1, nfeat)

    @staticmethod
    def disp_act(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.softplus(x), 1e-4, 1e4)

    @staticmethod
    def mean_act(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.disp_act(self.disp(x))
        mean = self.mean_act(self.mean(x))
        return pi, disp, mean


# =========================================================
# Optional image encoder
# =========================================================

class CNNEncoder(nn.Module):
    """
    Lightweight optional image encoder.

    Input:
        image: Tensor with shape (N, C, H, W)
    Output:
        Tensor with shape (N, out_dim)
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.features(image)
        x = x.flatten(1)
        x = self.proj(x)
        return x


# =========================================================
# Core proposed modules: public interfaces only
# =========================================================

class CLSEncoder(nn.Module):
    """
    CLS-token self-attention encoder.

    Public skeleton only. The full implementation is omitted in this release.
    """

    def __init__(self, dim: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.dropout = dropout

    def forward(self, x_tokens: torch.Tensor):
        """
        Args:
            x_tokens: spot-level tokens, shape (N, E)

        Returns:
            cls_enc: global summary token, shape (E,)
            patch_enc: contextualized spot tokens, shape (N, E)
        """
        raise NotImplementedError(
            "CLSEncoder is part of the proposed method and is omitted in the public skeleton."
        )


class CrossDecoder(nn.Module):
    """
    Cross-attention decoder between two modality-specific token sequences.

    Public skeleton only. The full implementation is omitted in this release.
    """

    def __init__(self, dim: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.dropout = dropout

    def forward(self, q_patches: torch.Tensor, kv_patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_patches: query-side spot tokens, shape (Nq, E)
            kv_patches: key/value-side spot tokens, shape (Nk, E)

        Returns:
            Cross-modal summary representation, shape (E,)
        """
        raise NotImplementedError(
            "CrossDecoder is part of the proposed method and is omitted in the public skeleton."
        )


class LowRankBilinearFusion(nn.Module):
    """
    Low-rank bilinear fusion module.

    Public skeleton only. The full implementation is omitted in this release.
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_out: int,
        rank: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_out = dim_out
        self.rank = rank
        self.dropout = dropout

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: first modality representation, shape (E,) or (N, E)
            y: second modality representation, shape (E,) or (N, E)

        Returns:
            fused representation, shape (dim_out,) or (N, dim_out)
        """
        raise NotImplementedError(
            "LowRankBilinearFusion is part of the proposed method and is omitted in the public skeleton."
        )

