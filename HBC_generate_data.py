"""
generate_data_hbc_cigst.py

Generate preprocessed h5ad files for CIGST on the HBC / breast cancer Visium dataset.

This script supports:
    1. HBC data with metadata.tsv manual annotations
    2. HBC data without metadata.tsv, running in unsupervised mode

Expected input structure:
    data_root/
        metadata.tsv                                      # optional
        V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5
        spatial/
        tissue_hires_image.png / tissue_lowres_image.png

Output structure:
    save_root/
        V1_Breast_Cancer_Block_A_Section_1/
            CIGST.h5ad
"""

from __future__ import division, print_function

import argparse
import os
from typing import Optional, Tuple

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
    highly_genes: int = 10000,
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
        5. highly variable gene selection
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
# Metadata / label utilities
# =========================================================

def infer_label_column(metadata: pd.DataFrame) -> Optional[str]:
    """Infer the manual annotation column from metadata.tsv."""
    possible_cols = [
        "ground_truth",
        "ground",
        "annotation",
        "Annotation",
        "manual_annotation",
        "Manual annotation",
        "label",
        "Label",
        "cluster",
        "Cluster",
        "region",
        "Region",
        "fine_annot_type",
        "annot_type",
        "pathology",
        "Pathology",
        "layer_guess_reordered",
        "layer_guess",
    ]

    for col in possible_cols:
        if col in metadata.columns:
            return col

    return None


def read_hbc_metadata(meta_path: str) -> Tuple[Optional[pd.Series], Optional[pd.Series], np.ndarray]:
    """
    Read HBC metadata.tsv if available.

    Returns:
        labels: textual annotations after removing NA labels
        ground: categorical numeric labels in string format
        na_index: row indices of NA labels
    """
    if not os.path.exists(meta_path):
        print("metadata.tsv not found. Running in unsupervised mode.")
        return None, None, np.array([], dtype=int)

    print("metadata.tsv found, trying to load labels...")
    meta = pd.read_table(meta_path, sep="\t")
    print("metadata columns:", meta.columns.tolist())

    label_col = infer_label_column(meta)

    if label_col is None:
        print("No valid label column found. Running in unsupervised mode.")
        return None, None, np.array([], dtype=int)

    labels = meta[label_col].copy()
    na_index = np.where(labels.isnull())[0]
    labels = labels.drop(labels.index[na_index])
    ground = labels.astype("category").cat.codes.astype(str)

    print(f"Using label column: {label_col}")
    print("number of valid labels:", len(labels))
    print("number of classes:", len(np.unique(labels.astype(str))))

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
    meta_file: str = "metadata.tsv",
):
    """
    Load, preprocess and graph-construct one HBC Visium dataset.

    Args:
        dataset: dataset name, e.g. 'V1_Breast_Cancer_Block_A_Section_1'
        data_root: root directory of raw HBC data
        highly_genes: number of highly variable genes
        k: k for feature graph construction
        radius: radius for spatial graph construction
        count_file: optional count matrix filename. If None, use
            '{dataset}_filtered_feature_bc_matrix.h5'
        meta_file: metadata filename under data_root

    Returns:
        Processed AnnData object.
    """
    if count_file is None:
        count_file = f"{dataset}_filtered_feature_bc_matrix.h5"

    count_path = os.path.join(data_root, count_file)
    meta_path = os.path.join(data_root, meta_file)

    if not os.path.exists(count_path):
        raise FileNotFoundError(f"Cannot find count matrix: {count_path}")

    # ========= read Visium =========
    adata_raw = sc.read_visium(
        path=data_root,
        count_file=count_file,
        load_images=True,
    )
    adata_raw.var_names_make_unique()

    obs_names = np.array(adata_raw.obs.index)
    positions = adata_raw.obsm["spatial"]

    # ========= optional labels =========
    labels, ground, na_index = read_hbc_metadata(meta_path)
    use_labels = labels is not None and ground is not None

    if use_labels:
        data = np.delete(adata_raw.X.toarray(), na_index, axis=0)
        obs_names = np.delete(obs_names, na_index, axis=0)
        positions = np.delete(positions, na_index, axis=0)
    else:
        data = adata_raw.X.toarray()

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

    if use_labels:
        adata.obs["ground_truth"] = labels.values
        adata.obs["ground"] = ground.values
    else:
        adata.obs["ground_truth"] = ["unknown"] * adata.n_obs
        adata.obs["ground"] = ["unknown"] * adata.n_obs

    # Spatial coordinates and Visium metadata.
    adata.obsm["spatial"] = positions
    adata.obs["array_row"] = adata_raw.obs.loc[obs_names, "array_row"].values
    adata.obs["array_col"] = adata_raw.obs.loc[obs_names, "array_col"].values
    adata.uns["spatial"] = adata_raw.uns["spatial"]

    # Gene metadata.
    for col in ["gene_ids", "feature_types", "genome"]:
        if col in adata_raw.var.columns:
            adata.var[col] = adata_raw.var.loc[adata.var_names, col].values

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
    adata.obsm["graph_nei"] = graph_nei.numpy() if hasattr(graph_nei, "numpy") else graph_nei
    adata.obsm["graph_neg"] = graph_neg.numpy() if hasattr(graph_neg, "numpy") else graph_neg

    return adata


# Backward-compatible alias
load_ST_file = load_st_file


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate preprocessed HBC data for CIGST.")

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["V1_Breast_Cancer_Block_A_Section_1"],
        help="HBC dataset names.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/HBC",
        help="Root directory of raw HBC Visium data.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="../generate_data/HBC",
        help="Root directory for generated h5ad files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/HBC.ini",
        help="Path to HBC config file.",
    )
    parser.add_argument(
        "--count_file",
        type=str,
        default=None,
        help="Optional count matrix filename. If omitted, use '{dataset}_filtered_feature_bc_matrix.h5'.",
    )
    parser.add_argument(
        "--meta_file",
        type=str,
        default="metadata.tsv",
        help="Metadata filename under data_root.",
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
            count_file=args.count_file,
            meta_file=args.meta_file,
        )

        output_path = os.path.join(savepath, args.output_name)
        print("saving:", output_path)
        adata.write(output_path)

        print("done")
        print("shape:", adata.X.shape)
        print("output:", output_path)


if __name__ == "__main__":
    main()
