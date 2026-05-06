"""
train_hbc_cigst.py

Training and evaluation script for CIGST on the HBC / breast cancer Visium dataset.

This script supports two settings:
    1. HBC data with manual annotations provided by metadata.tsv
    2. HBC data without valid ground truth labels, where the best epoch is selected by loss

Public-release note:
    The script keeps the experimental pipeline and calls the proposed objectives
    through public interfaces in utils.py. Core interaction implementations are
    not exposed in this training script.
"""

from __future__ import division, print_function

import argparse
import os
import random
from typing import Optional, Tuple

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
    interaction_constraint_loss_v2,
    normalize_sparse_matrix,
    regularization_loss,
    sparse_mx_to_torch_sparse_tensor,
)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =========================================================
# Metadata utilities
# =========================================================

def add_hbc_ground_truth(adata, meta_path: Optional[str] = None):
    """
    Merge HBC metadata annotations into adata.obs['ground_truth'].

    The metadata file is expected to contain one barcode column and one manual
    annotation column. If the annotation column name is not recognized, the last
    column is used by default.
    """
    if meta_path is None or not os.path.exists(meta_path):
        print(f"[Warning] Cannot find metadata file: {meta_path}")
        return adata

    meta = pd.read_csv(meta_path, sep="\t")

    print("\n[metadata.tsv columns]")
    print(meta.columns.tolist())
    print(meta.head())

    if "barcode" not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: "barcode"})

    meta["barcode"] = meta["barcode"].astype(str)

    possible_annotation_cols = [
        "ground_truth",
        "ground",
        "annotation",
        "Annotation",
        "manual_annotation",
        "Manual annotation",
        "label",
        "Label",
        "cluster",
        "region",
        "Region",
        "fine_annot_type",
        "annot_type",
        "pathology",
        "Pathology",
    ]

    anno_col = None
    for col in possible_annotation_cols:
        if col in meta.columns:
            anno_col = col
            break

    if anno_col is None:
        anno_col = meta.columns[-1]
        print(f"[Warning] Annotation column not recognized. Use last column: {anno_col}")
    else:
        print(f"[Info] Use annotation column: {anno_col}")

    meta = meta.set_index("barcode")

    adata.obs["ground_truth"] = adata.obs_names.map(meta[anno_col])
    adata.obs["ground"] = adata.obs["ground_truth"]

    print("\n[Ground truth check]")
    print("n_obs:", adata.n_obs)
    print("missing labels:", adata.obs["ground_truth"].isna().sum())
    print(adata.obs["ground_truth"].value_counts(dropna=False))

    return adata


def get_label_info(labels: pd.Series):
    """
    Check whether valid ground truth labels are available.

    Returns:
        label_array: all labels converted to string
        valid_mask: boolean mask for valid labels
        valid_labels: labels after removing invalid entries
        has_gt: whether labels contain more than one valid class
    """
    label_array = np.array(labels.astype(str), dtype=str)
    invalid_label_set = {"unknown", "nan", "None", "NA", "NaN"}
    valid_mask = np.array([x not in invalid_label_set for x in label_array])
    valid_labels = label_array[valid_mask]
    has_gt = len(np.unique(valid_labels)) > 1

    print("\n[Label check]")
    print("has_gt:", has_gt)
    print("unique labels:", np.unique(valid_labels))
    print("n_classes:", len(np.unique(valid_labels)))

    return label_array, valid_mask, valid_labels, has_gt


# =========================================================
# Data loading
# =========================================================

def load_data(
    dataset: str,
    data_root: str,
    meta_path: Optional[str] = None,
    output_name: str = "CIGST.h5ad",
) -> Tuple:
    """Load generated HBC h5ad data and optional manual annotations."""
    print("load data:")

    path = os.path.join(data_root, dataset, output_name)
    print("h5ad path:", path)

    if not os.path.exists(path):
        fallback_path = os.path.join(data_root, dataset, "MAFN.h5ad")
        if os.path.exists(fallback_path):
            path = fallback_path
            print("[Info] Use fallback h5ad path:", path)
        else:
            raise FileNotFoundError(f"Cannot find h5ad file: {path}")

    adata = sc.read_h5ad(path)

    if meta_path is not None:
        adata = add_hbc_ground_truth(adata, meta_path)

    features = torch.FloatTensor(adata.X)

    if "ground_truth" in adata.obs.columns:
        labels = adata.obs["ground_truth"]
    elif "ground" in adata.obs.columns:
        labels = adata.obs["ground"]
        adata.obs["ground_truth"] = labels
    else:
        labels = pd.Series(["unknown"] * adata.n_obs, index=adata.obs.index)
        adata.obs["ground_truth"] = labels

    labels = labels.astype(str)

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
# Training and evaluation
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
):
    """Run one training epoch."""
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

    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0.0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    dcir_loss = dicr_loss(z_s, z_e)

    # HBC uses the alternative interaction constraint interface by default.
    inter_loss = interaction_constraint_loss_v2(
        z_f,
        z_s,
        z_e,
        graph_nei,
        w_align=1.0,
        w_nei=0.1,
    )

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


def run_kmeans(emb: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """Run KMeans clustering on learned embeddings."""
    emb = np.nan_to_num(emb, nan=0.0, posinf=1e6, neginf=-1e6)
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=50,
        random_state=seed,
    ).fit(emb)
    return kmeans.labels_


def evaluate_if_available(
    labels_all: np.ndarray,
    pred: np.ndarray,
    valid_mask: np.ndarray,
    has_gt: bool,
):
    """Compute ARI/NMI only when valid ground truth labels exist."""
    if not has_gt:
        return -1.0, -1.0

    ari = metrics.adjusted_rand_score(labels_all[valid_mask], pred[valid_mask])
    nmi = metrics.normalized_mutual_info_score(labels_all[valid_mask], pred[valid_mask])
    return ari, nmi


# =========================================================
# Visualization and saving
# =========================================================

def save_manual_annotation(adata, dataset: str, savepath: str, has_gt: bool) -> None:
    """Save manual annotation plot when labels are available."""
    if not has_gt:
        print("[Warning] No valid ground truth found. Skip Manual_Annotation plot.")
        return

    title = f"Manual annotation ({dataset})"
    sc.pl.spatial(
        adata,
        img_key="hires",
        color=["ground_truth"],
        title=title,
        show=False,
    )
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
    nmi_max: float,
    has_gt: bool,
) -> None:
    """Save spatial visualization, UMAP/PAGA, embeddings and h5ad outputs."""
    if has_gt:
        title = f"CIGST: ARI={ari_max:.2f}, NMI={nmi_max:.2f}"
    else:
        title = f"CIGST: {dataset}"

    adata.obs["idx"] = idx_max.astype(str)
    adata.obsm["emb"] = emb_max
    adata.obsm["mean"] = mean_max

    sc.pl.spatial(
        adata,
        img_key="hires",
        color=["idx"],
        title=title,
        show=False,
    )
    plt.savefig(os.path.join(savepath, "CIGST.jpg"), bbox_inches="tight", dpi=600)
    plt.close()

    sc.pp.neighbors(adata, use_rep="emb")
    sc.tl.umap(adata)

    try:
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
        plt.savefig(os.path.join(savepath, "CIGST_umap_emb.jpg"), bbox_inches="tight", dpi=600)
        plt.close()
    except Exception as exc:
        print("[Warning] PAGA failed:", exc)

    pd.DataFrame(emb_max).to_csv(os.path.join(savepath, "CIGST_emb.csv"), index=False)
    pd.DataFrame(idx_max).to_csv(os.path.join(savepath, "CIGST_idx.csv"), index=False)

    adata.layers["X"] = adata.X
    adata.layers["mean"] = mean_max
    adata.write(os.path.join(savepath, "CIGST.h5ad"))

    print("Results saved to:", savepath)


# =========================================================
# Main experiment
# =========================================================

def run_single_dataset(args, dataset: str) -> None:
    """Run CIGST on one HBC dataset."""
    config = Config(args.config)

    print("\n==============================")
    print(dataset)
    print("==============================")

    adata, features, labels, fadj, sadj, graph_nei, graph_neg = load_data(
        dataset=dataset,
        data_root=args.data_root,
        meta_path=args.meta_path,
        output_name=args.input_name,
    )
    print(adata)

    savepath = os.path.join(args.save_root, dataset)
    os.makedirs(savepath, exist_ok=True)

    label_array, valid_mask, valid_labels, has_gt = get_label_info(labels)
    save_manual_annotation(adata, dataset, savepath, has_gt)

    cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    config.class_num = len(np.unique(valid_labels)) if has_gt else args.default_clusters
    config.n = adata.n_obs
    config.epochs = args.epochs if args.epochs is not None else config.epochs + 1
    config.lambda_inter = args.lambda_inter
    config.eta = args.eta

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
        "beta=", config.beta,
        "gamma=", config.gamma,
        "eta=", config.eta,
        "lambda_inter=", config.lambda_inter,
        "class_num=", config.class_num,
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
    best_loss = 1e10
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

        idx = run_kmeans(
            emb=emb,
            n_clusters=config.class_num,
            seed=config.seed,
        )

        ari_res, nmi_res = evaluate_if_available(
            labels_all=label_array,
            pred=idx,
            valid_mask=valid_mask,
            has_gt=has_gt,
        )

        if has_gt:
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
                idx_max = idx.copy()
                mean_max = mean.copy()
                emb_max = emb.copy()
        else:
            print(
                dataset,
                "epoch:", epoch,
                "zinb_loss={:.4f}".format(float(zinb_loss.detach().cpu())),
                "reg_loss={:.4f}".format(float(reg_loss.detach().cpu())),
                "dcir_loss={:.4f}".format(float(dcir_loss.detach().cpu())),
                "inter_loss={:.4f}".format(float(inter_loss.detach().cpu())),
                "total_loss={:.4f}".format(float(total_loss.detach().cpu())),
            )

            if float(total_loss.detach().cpu()) < best_loss:
                best_loss = float(total_loss.detach().cpu())
                epoch_max = epoch
                idx_max = idx.copy()
                mean_max = mean.copy()
                emb_max = emb.copy()

    print(dataset, "best_ari=", ari_max, "best_nmi=", nmi_max, "best_epoch=", epoch_max)

    if idx_max is None:
        idx_max = idx.copy()
        mean_max = mean.copy()
        emb_max = emb.copy()

    save_cigst_results(
        adata=adata,
        dataset=dataset,
        savepath=savepath,
        idx_max=idx_max,
        emb_max=emb_max,
        mean_max=mean_max,
        ari_max=ari_max,
        nmi_max=nmi_max,
        has_gt=has_gt,
    )


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train CIGST on HBC spatial transcriptomics data.")

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["V1_Breast_Cancer_Block_A_Section_1"],
        help="HBC dataset names.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../generate_data/HBC",
        help="Root directory of generated HBC h5ad files.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="./result/HBC",
        help="Directory for saving HBC results.",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default=None,
        help="Path to HBC metadata.tsv. If omitted, labels are read from h5ad if available.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/HBC.ini",
        help="Path to HBC config file.",
    )
    parser.add_argument(
        "--input_name",
        type=str,
        default="CIGST.h5ad",
        help="Input h5ad filename under each generated dataset folder.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs.",
    )
    parser.add_argument(
        "--lambda_inter",
        type=float,
        default=0.05,
        help="Weight for interaction constraint loss.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="Weight for DICR loss.",
    )
    parser.add_argument(
        "--default_clusters",
        type=int,
        default=7,
        help="Cluster number used when no valid ground truth exists.",
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
