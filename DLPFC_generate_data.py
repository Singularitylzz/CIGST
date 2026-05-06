"""
generate_data_cigst.py

Preprocess 10x Visium spatial transcriptomics data and generate h5ad files
for CIGST training.

This script keeps the standard data-preparation pipeline:
    1. read Visium count matrix and spatial image information
    2. align manual annotations with valid spots
    3. preprocess expression matrix and select HVGs
    4. construct feature graph and spatial graph
    5. save the processed AnnData object

Expected input structure for each DLPFC slice:
    data_root/
        151676/
            metadata.tsv
            151676_filtered_feature_bc_matrix.h5
            spatial/
            tissue_hires_image.png / tissue_lowres_image.png

Output:
    save_root/
        151676/
            CIGST.h5ad
"""

from __future__ import division, print_function

import argparse
import os
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from config import Config
from utils import features_construct_graph, spatial_construct_graph_from_adata


# =========================================================
# Preprocessing
# =========================================================

def preprocess_adata(
    adata,
    highly_genes: int = 3000,
    min_cells: int = 100,
    target_sum: float = 1e4,
    scale_max_value: float = 10.0,
):
    """
    Preprocess expression matrix and select highly variable genes.

    Steps:
        1. filter low-expression genes
        2. remove empty spots
        3. library-size normalization
        4. log1p transformation
        5. HVG selection
        6. scaling
    """
    print("start preprocessing & HVG selection")

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        n_top_genes=highly_genes,
    )

    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, zero_center=False, max_value=scale_max_value)

    return adata


# Backward-compatible alias
normalize = preprocess_adata


# =========================================================
# Label processing
# =========================================================

def read_dlpfc_labels(labels_path: str):
    """
    Read DLPFC manual annotations and remove unlabeled spots.

    Returns:
        labels: textual layer annotations after removing NA labels
        ground: numeric labels in string format
        na_index: indices of NA labels to be removed from Visium data
    """
    labels_df = pd.read_table(labels_path, sep="\t")

    if "layer_guess_reordered" not in labels_df.columns:
        raise KeyError("metadata.tsv must contain 'layer_guess_reordered'.")

    labels = labels_df["layer_guess_reordered"].copy()
    na_index = np.where(labels.isnull())[0]
    labels = labels.drop(labels.index[na_index])

    ground = labels.copy()
    label_mapping = {
        "WM": "0",
        "Layer1": "1",
        "Layer2": "2",
        "Layer3": "3",
        "Layer4": "4",
        "Layer5": "5",
        "Layer6": "6",
    }
    ground.replace(label_mapping, inplace=True)

    return labels, ground, na_index


# =========================================================
# Main data generation function
# =========================================================

def load_st_file(
    dataset: str,
    data_root: str,
    highly_genes: int,
    k: int,
    radius: int,
    count_file: Optional[str] = None,
):
    """
    Load and preprocess one Visium spatial transcriptomics slice.

    Args:
        dataset: slice ID, e.g. '151676'
        data_root: root directory containing DLPFC slice folders
        highly_genes: number of highly variable genes
        k: number of neighbors for feature graph
        radius: radius for spatial graph construction
        count_file: optional count matrix filename. If None, use
            '{dataset}_filtered_feature_bc_matrix.h5'

    Returns:
        Processed AnnData object.
    """
    path = os.path.join(data_root, dataset)
    labels_path = os.path.join(path, "metadata.tsv")

    if count_file is None:
        count_file = f"{dataset}_filtered_feature_bc_matrix.h5"

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Cannot find metadata file: {labels_path}")

    count_path = os.path.join(path, count_file)
    if not os.path.exists(count_path):
        raise FileNotFoundError(f"Cannot find count matrix: {count_path}")

    # ========= labels =========
    labels, ground, na_index = read_dlpfc_labels(labels_path)

    # ========= read Visium =========
    adata_raw = sc.read_visium(
        path,
        count_file=count_file,
        load_images=True,
    )
    adata_raw.var_names_make_unique()

    obs_names = np.array(adata_raw.obs.index)
    positions = adata_raw.obsm["spatial"]

    # Remove spots without manual annotations.
    data = np.delete(adata_raw.X.toarray(), na_index, axis=0)
    obs_names = np.delete(obs_names, na_index, axis=0)
    positions = np.delete(positions, na_index, axis=0)

    # ========= construct AnnData =========
    adata = ad.AnnData(
        pd.DataFrame(
            data,
            index=obs_names,
            columns=np.array(adata_raw.var.index),
            dtype=np.float32,
        )
    )

    adata.var_names_make_unique()

    # Manual annotations for evaluation only.
    adata.obs["ground_truth"] = labels.values
    adata.obs["ground"] = ground.values

    # Spatial coordinates and image metadata.
    adata.obsm["spatial"] = positions
    adata.obs["array_row"] = adata_raw.obs.loc[obs_names, "array_row"].values
    adata.obs["array_col"] = adata_raw.obs.loc[obs_names, "array_col"].values
    adata.uns["spatial"] = adata_raw.uns["spatial"]

    # Gene metadata.
    for key in ["gene_ids", "feature_types", "genome"]:
        if key in adata_raw.var.columns:
            adata.var[key] = adata_raw.var.loc[adata.var_names, key]

    adata.var_names_make_unique()

    # ========= preprocessing =========
    adata = preprocess_adata(adata, highly_genes=highly_genes)

    # ========= graph construction =========
    print("construct feature graph")
    fadj = features_construct_graph(adata.X, k=k)

    print("construct spatial graph")
    sadj, graph_nei, graph_neg = spatial_construct_graph_from_adata(
        adata,
        radius=radius,
    )

    adata.obsm["fadj"] = fadj
    adata.obsm["sadj"] = sadj
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()

    return adata


# Backward-compatible alias
load_ST_file = load_st_file


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate preprocessed data for CIGST.")

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["151676"],
        help="DLPFC slice IDs to preprocess.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/DLPFC",
        help="Root directory of raw DLPFC Visium data.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="../generate_data/DLPFC",
        help="Root directory for saving generated h5ad files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/DLPFC.ini",
        help="Path to config file.",
    )
    parser.add_argument(
        "--highly_genes",
        type=int,
        default=None,
        help="Override number of highly variable genes.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override k for feature graph construction.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=None,
        help="Override radius for spatial graph construction.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="CIGST.h5ad",
        help="Output h5ad filename.",
    )

    return parser.parse_args()


# =========================================================
# Main
# =========================================================

def main():
    args = parse_args()
    config = Config(args.config)

    highly_genes = args.highly_genes if args.highly_genes is not None else config.fdim
    k = args.k if args.k is not None else config.k
    radius = args.radius if args.radius is not None else config.radius

    os.makedirs(args.save_root, exist_ok=True)

    for dataset in args.datasets:
        print("=" * 60)
        print("processing dataset:", dataset)

        savepath = os.path.join(args.save_root, dataset)
        os.makedirs(savepath, exist_ok=True)

        adata = load_st_file(
            dataset=dataset,
            data_root=args.data_root,
            highly_genes=highly_genes,
            k=k,
            radius=radius,
        )

        output_path = os.path.join(savepath, args.output_name)
        print("saving:", output_path)
        adata.write(output_path)

        print("done")
        print("shape:", adata.X.shape)


if __name__ == "__main__":
    main()
