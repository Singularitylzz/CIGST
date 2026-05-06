"""
Training and evaluation script for CIGST.

"""

from __future__ import division, print_function

import argparse
import os
import random
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.optim as optim
from sklearn import metrics
from sklearn.cluster import KMeans

from config import Config
from models import CIGST
from utils import (
    ZINB,
    dicr_loss,
    interaction_constraint_loss,
    normalize_sparse_matrix,
    regularization_loss,
    sparse_mx_to_torch_sparse_tensor,
)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =========================================================
# Data loading
# =========================================================

def load_data(dataset: str, data_root: str) -> Tuple:
    """
    Load preprocessed DLPFC data.

    Expected h5ad fields:
        adata.X                  : expression features
        adata.obs['ground']      : manual labels for evaluation
        adata.obsm['fadj']       : feature graph adjacency
        adata.obsm['sadj']       : spatial graph adjacency
        adata.obsm['graph_nei']  : neighbor mask
        adata.obsm['graph_neg']  : negative-pair mask
    """
    print("load data:", dataset)
    path = os.path.join(data_root, dataset, "CIGST.h5ad")

    if not os.path.exists(path):
        # Backward compatibility with earlier preprocessing filename.
        fallback_path = os.path.join(data_root, dataset, "MAFN.h5ad")
        if os.path.exists(fallback_path):
            path = fallback_path
        else:
            raise FileNotFoundError(f"Cannot find data file: {path}")

    adata = sc.read_h5ad(path)

    features = torch.FloatTensor(adata.X)
    labels = adata.obs["ground"]

    fadj = adata.obsm["fadj"]
    sadj = adata.obsm["sadj"]

    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

    graph_nei = torch.FloatTensor(adata.obsm["graph_nei"])
    graph_neg = torch.FloatTensor(adata.obsm["graph_neg"])

    print("done")
    return adata, features, labels, nfadj, nsadj, graph_nei, graph_neg


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int, cuda: bool = False) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================================================
# One training step
# =========================================================

def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    sadj: torch.Tensor,
    fadj: torch.Tensor,
    graph_nei: torch.Tensor,
    graph_neg: torch.Tensor,
    config,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one optimization step."""
    model.train()
    optimizer.zero_grad()

    outputs = model(features, sadj, fadj, image=None, return_dict=True)

    emb = outputs["emb"]
    z_s = outputs["z_s"]
    z_e = outputs["z_e"]
    z_f = outputs["z_f"]
    pi = outputs["pi"]
    disp = outputs["disp"]
    mean = outputs["mean"]

    # Reconstruction objective
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0.0).loss(features, mean, mean=True)

    # Spatial regularization objective
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)

    # Proposed cross-view and interaction objectives.
    # Implementations are provided through utils.py interfaces.
    dcir_loss = dicr_loss(z_s, z_e)
    inter_loss = interaction_constraint_loss(z_f, z_s, z_e, graph_nei)

    total_loss = (
        config.alpha * zinb_loss
        + config.gamma * reg_loss
        + config.eta * dcir_loss
        + config.lambda_inter * inter_loss
    )

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    emb_np = np.nan_to_num(
        emb.detach().cpu().numpy(),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )
    mean_np = np.nan_to_num(
        mean.detach().cpu().numpy(),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )

    return emb_np, mean_np, zinb_loss, reg_loss, dcir_loss, inter_loss, total_loss


# =========================================================
# Evaluation
# =========================================================

def evaluate_clustering(
    emb: np.ndarray,
    labels,
    n_clusters: int,
    seed: int,
) -> Tuple[np.ndarray, float, float]:
    """Evaluate learned embedding with KMeans clustering."""
    emb = np.nan_to_num(emb, nan=0.0, posinf=1e6, neginf=-1e6)

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=50,
        random_state=seed,
    ).fit(emb)

    pred = kmeans.labels_
    ari = metrics.adjusted_rand_score(labels, pred)
    nmi = metrics.normalized_mutual_info_score(labels, pred)
    return pred, ari, nmi


# =========================================================
# Visualization and saving
# =========================================================

def save_manual_annotation(adata, dataset: str, savepath: str) -> None:
    """Save manual annotation spatial plot."""
    title = f"Manual annotation (slice #{dataset})"

    color_key = "ground_truth" if "ground_truth" in adata.obs.columns else "ground"
    sc.pl.spatial(adata, img_key="hires", color=[color_key], title=title, show=False)
    plt.savefig(os.path.join(savepath, "Manual_Annotation.jpg"), bbox_inches="tight", dpi=600)
    plt.close()


def save_cigst_results(
    adata,
    dataset: str,
    savepath: str,
    idx_max: np.ndarray,
    emb_max: np.ndarray,
    mean_max: np.ndarray,
    ari_max: float,
) -> None:
    """Save spatial plot, UMAP/PAGA plot, embeddings, labels and h5ad output."""
    title = f"CIGST: ARI={ari_max:.2f}"

    adata.obs["idx"] = idx_max.astype(str)
    adata.obsm["emb"] = emb_max
    adata.obsm["mean"] = mean_max

    sc.pl.spatial(adata, img_key="hires", color=["idx"], title=title, show=False)
    plt.savefig(os.path.join(savepath, "CIGST.jpg"), bbox_inches="tight", dpi=600)
    plt.close()

    sc.pp.neighbors(adata, use_rep="mean")
    sc.tl.umap(adata)
    sc.tl.paga(adata, groups="idx")
    sc.pl.paga_compare(
        adata,
        legend_fontsize=10,
        frameon=False,
        size=20,
        title=title,
        legend_fontoutline=2,
        show=False,
    )
    plt.savefig(os.path.join(savepath, "CIGST_umap_mean.jpg"), bbox_inches="tight", dpi=600)
    plt.close()

    pd.DataFrame(emb_max).to_csv(os.path.join(savepath, "CIGST_emb.csv"), index=False)
    pd.DataFrame(idx_max).to_csv(os.path.join(savepath, "CIGST_idx.csv"), index=False)

    adata.layers["X"] = adata.X
    adata.layers["mean"] = mean_max
    adata.write(os.path.join(savepath, "CIGST.h5ad"))


# =========================================================
# Main experiment
# =========================================================

def run_single_dataset(args, dataset: str) -> None:
    """Run CIGST on one DLPFC slice."""
    config = Config(args.config)

    adata, features, labels, fadj, sadj, graph_nei, graph_neg = load_data(dataset, args.data_root)
    print(adata)

    savepath = os.path.join(args.save_root, dataset)
    os.makedirs(savepath, exist_ok=True)

    plt.rcParams["figure.figsize"] = (3, 3)
    save_manual_annotation(adata, dataset, savepath)

    _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
    ground = torch.LongTensor(ground)
    config.n = len(ground)
    config.class_num = len(ground.unique())

    config.epochs = args.epochs if args.epochs is not None else config.epochs + 1
    config.lambda_inter = args.lambda_inter
    config.eta = args.eta

    cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    set_seed(config.seed, cuda=cuda)

    features = features.to(device)
    sadj = sadj.to(device)
    fadj = fadj.to(device)
    graph_nei = graph_nei.to(device)
    graph_neg = graph_neg.to(device)

    print(
        dataset,
        "lr=", config.lr,
        "alpha=", config.alpha,
        "gamma=", config.gamma,
        "eta=", config.eta,
        "lambda_inter=", config.lambda_inter,
    )

    model = CIGST(
        nfeat=config.fdim,
        nhid1=config.nhid1,
        nhid2=config.nhid2,
        dropout=config.dropout,
        use_image=False,
        mode="graph_only",
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    epoch_max = 0
    ari_max = -1.0
    nmi_max = -1.0
    idx_max = None
    mean_max = None
    emb_max = None

    for epoch in range(config.epochs):
        emb, mean, zinb_loss, reg_loss, dcir_loss, inter_loss, total_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            features=features,
            sadj=sadj,
            fadj=fadj,
            graph_nei=graph_nei,
            graph_neg=graph_neg,
            config=config,
        )

        idx, ari_res, nmi_res = evaluate_clustering(
            emb=emb,
            labels=labels,
            n_clusters=config.class_num,
            seed=config.seed,
        )

        print(
            dataset,
            "epoch:", epoch,
            "zinb_loss={:.4f}".format(float(zinb_loss.detach().cpu())),
            "reg_loss={:.4f}".format(float(reg_loss.detach().cpu())),
            "dcir_loss={:.4f}".format(float(dcir_loss.detach().cpu())),
            "inter_loss={:.4f}".format(float(inter_loss.detach().cpu())),
            "total_loss={:.4f}".format(float(total_loss.detach().cpu())),
            "ARI={:.4f}".format(ari_res),
            "NMI={:.4f}".format(nmi_res),
        )

        if ari_res > ari_max:
            ari_max = ari_res
            nmi_max = nmi_res
            epoch_max = epoch
            idx_max = idx
            mean_max = mean
            emb_max = emb

    print(dataset, "best_ari=", ari_max, "best_nmi=", nmi_max, "best_epoch=", epoch_max)

    save_cigst_results(
        adata=adata,
        dataset=dataset,
        savepath=savepath,
        idx_max=idx_max,
        emb_max=emb_max,
        mean_max=mean_max,
        ari_max=ari_max,
    )


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train CIGST on spatial transcriptomics datasets.")

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["151676"],
        help="Dataset slice IDs to run.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../generate_data/DLPFC",
        help="Root directory of preprocessed h5ad files.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="./result/DLPFC",
        help="Directory for saving results.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/DLPFC.ini",
        help="Path to config file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )
    parser.add_argument(
        "--lambda_inter",
        type=float,
        default=1.0,
        help="Weight for interaction constraint loss.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="Weight for DICR loss.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA visible device ID.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    for dataset in args.datasets:
        run_single_dataset(args, dataset)
