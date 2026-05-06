"""
utils.py

Utility functions for spatial transcriptomics representation learning.

Note:
    This file keeps general preprocessing, graph construction, clustering,
    and numerical-stability utilities. Core algorithmic components are exposed
    only as function interfaces/placeholders to avoid disclosing full method
    details before formal release.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph

try:
    import scanpy as sc
except ImportError:
    sc = None

try:
    import networkx as nx
    import community as community_louvain
except ImportError:
    nx = None
    community_louvain = None

EPS = 1e-15


# =========================================================
# Basic tensor / matrix utilities
# =========================================================

def _nan2zero(x: torch.Tensor) -> torch.Tensor:
    """Replace NaN values with zeros."""
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x: torch.Tensor) -> torch.Tensor:
    """Replace NaN values with positive infinity."""
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """Convert a SciPy sparse matrix to a PyTorch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def sparse_to_tuple(sparse_mx: Union[sp.spmatrix, list]) -> Union[tuple, list]:
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx: sp.spmatrix) -> tuple:
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        return [to_tuple(mx) for mx in sparse_mx]
    return to_tuple(sparse_mx)


def normalize_sparse_matrix(mx: sp.spmatrix) -> sp.spmatrix:
    """Row-normalize a sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)


def degree_power(A: Union[np.ndarray, sp.spmatrix], power: float) -> Union[np.ndarray, sp.spmatrix]:
    """Compute D^power for an adjacency matrix A."""
    degrees = np.power(np.array(A.sum(1)), power).flatten()
    degrees[np.isinf(degrees)] = 0.0
    if sp.issparse(A):
        return sp.diags(degrees)
    return np.diag(degrees)


def norm_adj(A: Union[np.ndarray, sp.spmatrix]) -> Union[np.ndarray, sp.spmatrix]:
    """Symmetrically normalize adjacency matrix: D^{-1/2} A D^{-1/2}."""
    normalized_D = degree_power(A, -0.5)
    return normalized_D.dot(A).dot(normalized_D)


def dopca(data: Union[np.ndarray, sp.spmatrix], dim: int = 50) -> np.ndarray:
    """Apply PCA to dense or sparse input features."""
    if sp.issparse(data):
        data = data.toarray()
    return PCA(n_components=dim).fit_transform(data)


def PCA_process(X: Union[np.ndarray, sp.spmatrix], nps: int) -> np.ndarray:
    """Run PCA and print the retained variance ratio."""
    if sp.issparse(X):
        X = X.toarray()
    print("Shape of data to PCA:", X.shape)
    pca = PCA(n_components=nps)
    X_PC = pca.fit_transform(X)
    print("Shape of data output by PCA:", X_PC.shape)
    print("PCA recover:", pca.explained_variance_ratio_.sum())
    return X_PC


# =========================================================
# Initialization and attention-mask utilities
# =========================================================

def init_max_weights(module_or_model: nn.Module) -> None:
    """Initialize Linear, Bilinear, Conv2d and Embedding layers."""
    for m in module_or_model.modules():
        if isinstance(m, (nn.Linear, nn.Bilinear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)


def build_attn_mask(
    adj: Union[torch.Tensor, sp.spmatrix, np.ndarray],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert adjacency matrix to a boolean attention mask for MultiheadAttention.

    True means the corresponding position is masked. Self-loops and neighbors
    are allowed, while non-neighbor positions are masked.
    """
    if isinstance(adj, torch.Tensor):
        adj_tensor = adj.to_dense() if adj.is_sparse else adj.clone()
    elif sp.issparse(adj):
        adj_tensor = torch.tensor(adj.toarray(), dtype=torch.float32)
    elif isinstance(adj, np.ndarray):
        adj_tensor = torch.from_numpy(adj.astype(np.float32))
    else:
        raise TypeError(f"Unsupported adj type: {type(adj)}")

    if device is None:
        device = adj_tensor.device

    adj_tensor = (adj_tensor > 0).to(torch.bool).to(device)
    n_nodes = adj_tensor.size(0)
    eye = torch.eye(n_nodes, dtype=torch.bool, device=device)
    allow = adj_tensor | eye
    return ~allow


# =========================================================
# Graph construction utilities
# =========================================================

def spatial_construct_graph(
    positions: np.ndarray,
    k: int = 15,
) -> Tuple[sp.coo_matrix, torch.Tensor, torch.Tensor]:
    """
    Construct a spatial graph from spot coordinates using an adaptive radius.

    Returns:
        sadj: symmetric SciPy sparse adjacency matrix
        graph_nei: dense neighbor mask tensor
        graph_neg: dense non-neighbor mask tensor
    """
    print("start spatial construct graph")
    distances = euclidean_distances(positions)

    tmp = 0
    min_k = 2
    for step, interval in [(100, range(100, 1000, 100)),
                           (10, None),
                           (5, None)]:
        if interval is None:
            interval = range(tmp - step * 10, 1000, step)
        for threshold in interval:
            A_tmp = np.where(distances > threshold, 0, 1)
            if min_k < np.min(np.sum(A_tmp, axis=1)) and k < np.max(np.sum(A_tmp, axis=1)):
                tmp = threshold
                if step == 5:
                    distances = A_tmp
                break

    A = distances if set(np.unique(distances)).issubset({0, 1}) else np.where(distances > tmp, 0, 1)
    row, col = np.diag_indices_from(A)
    A[row, col] = 0

    graph_nei = torch.from_numpy(A.astype(np.float32))
    graph_neg = torch.ones(positions.shape[0], positions.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    return sadj, graph_nei, graph_neg


def spatial_construct_graph_from_adata(
    adata,
    radius: int = 150,
) -> Tuple[sp.coo_matrix, torch.Tensor, torch.Tensor]:
    """Construct a spatial graph from adata.obsm['spatial'] using a fixed radius."""
    from sklearn.neighbors import NearestNeighbors

    coor = pd.DataFrame(adata.obsm["spatial"])
    coor.index = adata.obs.index
    coor.columns = ["imagerow", "imagecol"]

    A = np.zeros((coor.shape[0], coor.shape[0]), dtype=np.float32)
    nbrs = NearestNeighbors(radius=radius).fit(coor)
    _, indices = nbrs.radius_neighbors(coor, return_distance=True)

    for i in range(indices.shape[0]):
        A[[i] * indices[i].shape[0], indices[i]] = 1

    print("The graph contains %d edges, %d cells." % (A.sum(), adata.n_obs))
    print("%.4f neighbors per cell on average." % (A.sum() / adata.n_obs))

    graph_nei = torch.from_numpy(A)
    graph_neg = torch.ones(coor.shape[0], coor.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    return sadj, graph_nei, graph_neg


def features_construct_graph(
    features: Union[np.ndarray, sp.spmatrix, torch.Tensor],
    k: int = 15,
    pca: Optional[int] = None,
    mode: str = "connectivity",
    metric: str = "cosine",
) -> sp.coo_matrix:
    """Construct a feature kNN graph and return a symmetric SciPy sparse adjacency."""
    print("start features construct graph")

    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if sp.issparse(features):
        features = features.toarray()
    if pca is not None:
        features = dopca(features, dim=pca)

    A = kneighbors_graph(
        features,
        k + 1,
        mode=mode,
        metric=metric,
        include_self=True,
    ).toarray()

    row, col = np.diag_indices_from(A)
    A[row, col] = 0

    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    return fadj


def get_adj(
    data: Union[np.ndarray, sp.spmatrix],
    pca: Optional[int] = None,
    k: int = 25,
    mode: str = "connectivity",
    metric: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct kNN adjacency matrix and its normalized version."""
    if pca is not None:
        data = dopca(data, dim=pca)
    A = kneighbors_graph(data, k, mode=mode, metric=metric, include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    return adj, adj_n


# =========================================================
# Generic loss utilities
# =========================================================

def cosine_similarity(emb: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity with numerical-stability handling."""
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T) + EPS)
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat


def regularization_loss(
    emb: torch.Tensor,
    graph_nei: torch.Tensor,
    graph_neg: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Graph regularization loss for neighbor similarity and non-neighbor separation."""
    mat = torch.sigmoid(cosine_similarity(emb))
    mat = torch.clamp(mat, eps, 1.0 - eps)
    neigh_loss = torch.mul(graph_nei, torch.log(mat)).mean()
    neg_loss = torch.mul(graph_neg, torch.log(1.0 - mat)).mean()
    return -(neigh_loss + neg_loss) / 2


def consistency_loss(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """Distributional consistency loss between two embeddings."""
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    return torch.mean((cov1 - cov2) ** 2)


def xin1_loss(emb1: torch.Tensor, xin1: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss for the first branch."""
    assert emb1.shape == xin1.shape
    return F.mse_loss(emb1, xin1)


def xin2_loss(emb2: torch.Tensor, xin2: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss for the second branch."""
    assert emb2.shape == xin2.shape
    return F.mse_loss(emb2, xin2)


# =========================================================
# Core method placeholders
# =========================================================
# The following functions are intentionally left as interfaces only.
# Release the implementation after acceptance or when making the code fully public.


def dicr_loss(com1: torch.Tensor, com2: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Dual information correlation reduction loss.

    Placeholder for the proposed cross-view decorrelation objective.
    """
    raise NotImplementedError(
        "dicr_loss is part of the proposed method and is omitted in this public skeleton."
    )


def contrastive_interaction_loss(
    z_f: torch.Tensor,
    z_s: torch.Tensor,
    z_e: torch.Tensor,
    tau: float = 0.2,
    *args,
    **kwargs,
) -> torch.Tensor:
    """
    Cross-modal contrastive interaction loss.

    Placeholder for the proposed instance-level alignment objective.
    """
    raise NotImplementedError(
        "contrastive_interaction_loss is part of the proposed method and is omitted in this public skeleton."
    )


def interaction_constraint_loss(
    z_f: torch.Tensor,
    z_s: torch.Tensor,
    z_e: torch.Tensor,
    graph_nei: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
) -> torch.Tensor:
    """
    Interaction-aware relation constraint loss.

    Placeholder for the proposed interaction relation modeling objective.
    """
    raise NotImplementedError(
        "interaction_constraint_loss is part of the proposed method and is omitted in this public skeleton."
    )


def interaction_constraint_loss_v2(
    z_f: torch.Tensor,
    z_s: torch.Tensor,
    z_e: torch.Tensor,
    graph_nei: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
) -> torch.Tensor:
    """
    Alternative interaction constraint interface.

    Placeholder for ablation or extended version.
    """
    raise NotImplementedError(
        "interaction_constraint_loss_v2 is part of the proposed method and is omitted in this public skeleton."
    )


# =========================================================
# ZINB / NB reconstruction losses
# =========================================================

class NB(object):
    """Negative binomial reconstruction loss."""

    def __init__(self, theta: Optional[torch.Tensor] = None, scale_factor: float = 1.0):
        super().__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor, mean: bool = True) -> torch.Tensor:
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6, device=self.theta.device))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
            y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps))
        )
        final = _nan2inf(t1 + t2)
        return torch.mean(final) if mean else final


class ZINB(NB):
    """Zero-inflated negative binomial reconstruction loss."""

    def __init__(self, pi: torch.Tensor, ridge_lambda: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor, mean: bool = True) -> torch.Tensor:
        scale_factor = self.scale_factor
        eps = self.eps

        pi = self.pi.clamp(1e-4, 1 - 1e-4)
        theta = torch.minimum(self.theta, torch.tensor(1e6, device=self.theta.device)).clamp(1e-4, 1e6)
        y_pred = y_pred.clamp(1e-5, 1e6) * scale_factor

        t1 = torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + eps)
        t2 = (theta + y_true) * torch.log1p(y_pred / (theta + eps)) + (
            y_true * (torch.log(theta + eps) - torch.log(y_pred + eps))
        )
        nb_case = (t1 + t2) - torch.log(1.0 - pi + eps)

        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)

        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        result = result + self.ridge_lambda * torch.square(pi)
        result = _nan2inf(result)
        return torch.mean(result) if mean else result


# =========================================================
# Clustering utilities
# =========================================================

def mclust_R(
    adata,
    num_cluster: int,
    modelNames: str = "EEE",
    used_obsm: str = "emb",
    random_seed: int = 2020,
):
    """Cluster embeddings using the R package mclust."""
    np.random.seed(random_seed)

    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri

    robjects.r.library("mclust")
    rpy2.robjects.numpy2ri.activate()
    robjects.r["set.seed"](random_seed)
    rmclust = robjects.r["Mclust"]

    res = rmclust(
        rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]),
        num_cluster,
        modelNames,
    )
    mclust_res = np.array(res[-2])

    adata.obs["mclust"] = mclust_res
    adata.obs["mclust"] = adata.obs["mclust"].astype("int").astype("category")
    return adata


def computeCentroids(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute cluster centroids."""
    n_clusters = len(np.unique(labels))
    return np.array([np.mean(data[labels == i], axis=0) for i in range(n_clusters)])


class LouvainClustering:
    """Simple Louvain clustering wrapper."""

    def __init__(self, level: float):
        self.level = level

    def updateLabels(self, level: float) -> None:
        level = int((len(self.dendrogram) - 1) * level)
        partition = community_louvain.partition_at_level(self.dendrogram, level)
        self.labels = np.array(list(partition.values()))

    def update(self, inputs: np.ndarray, adj_mat: np.ndarray) -> None:
        if nx is None or community_louvain is None:
            raise ImportError("Please install networkx and python-louvain first.")
        self.graph = nx.from_numpy_array(adj_mat)
        self.dendrogram = community_louvain.generate_dendrogram(self.graph)
        self.updateLabels(self.level)
        self.centroids = computeCentroids(inputs, self.labels)


def res_search_fixed_clus(
    cluster_type: str,
    adata,
    fixed_clus_count: int,
    increment: float = 0.01,
) -> Tuple[np.ndarray, int]:
    """Search Leiden/Louvain resolution to obtain a target number of clusters."""
    if sc is None:
        raise ImportError("scanpy is required for resolution search.")

    cluster_labels = None
    flag = 1

    if cluster_type == "leiden":
        for res in sorted(list(np.arange(0.14, 2.5, increment))):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs["leiden"]).leiden.unique())
            if count_unique == fixed_clus_count:
                cluster_labels = np.array(adata.obs["leiden"])
                flag = 0
                break
            if count_unique > fixed_clus_count:
                cluster_labels = np.array(adata.obs["leiden"])
                flag = 1
                break

    elif cluster_type == "louvain":
        for res in sorted(list(np.arange(0.14, 2.5, increment))):
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs["louvain"]).louvain.unique())
            if count_unique == fixed_clus_count:
                cluster_labels = np.array(adata.obs["louvain"])
                flag = 0
                break
            if count_unique > fixed_clus_count:
                cluster_labels = np.array(adata.obs["louvain"])
                flag = 1
                break
    else:
        raise ValueError("cluster_type must be either 'leiden' or 'louvain'.")

    if cluster_labels is None:
        warnings.warn("Target cluster number was not found. Returning the last available labels.")
        cluster_labels = np.array(adata.obs[cluster_type])

    return cluster_labels, flag


# =========================================================
# Visualization colors
# =========================================================

class Colors:
    """Default color palette for spatial-domain visualization."""

    colors = [
        "#1f77b4", "#ff7f0e", "#279e68", "#d62728", "#633194",
        "#8c564b", "#F73BAD", "#ad494a", "#F6E800", "#01F7F7",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#8c6d31",
    ]
