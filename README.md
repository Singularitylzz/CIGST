

# CIGST
This repository provides the official implementation framework of **CIGST** for spatial domain identification in spatial transcriptomics data.
CIGST integrates spatial graph representation learning, feature graph representation learning, and cross-modal interaction modeling to learn discriminative spot-level embeddings for spatial domain clustering.
---
## Repository Structure
```text
.
├── DLPFC_generate_data.py      # Generate preprocessed DLPFC h5ad files
├── DLPFC_test.py               # Train and evaluate CIGST on DLPFC slices
├── HBC_generate_data.py        # Generate preprocessed HBC h5ad files
├── HBC_test.py                 # Train and evaluate CIGST on HBC data
├── models.py                   # Model architecture and CIGST framework
├── utils.py                    # Graph construction, losses, clustering and utility functions
└── README.md

⸻

Datasets

1. DorsoLateral PreFrontal Cortex (DLPFC)

The DLPFC dataset is from the spatialLIBD project:

* Dataset / code source: LieberInstitute/spatialLIBD￼

The dataset contains 10x Visium spatial transcriptomics slices of the human dorsolateral prefrontal cortex with manual annotations of cortical layers and white matter.

Commonly used slices include:

151507, 151508, 151509, 151510,
151669, 151670, 151671, 151672,
151673, 151674, 151675, 151676

2. Human Breast Cancer (HBC)

The HBC dataset is from 10x Genomics:

* Dataset source: Human Breast Cancer (Block A Section 1) - 10x Genomics￼

⸻

Data Preparation

DLPFC

Run:

python DLPFC_generate_data.py

The generated files will be saved under:

../generate_data/DLPFC/

Each processed slice will be saved as an .h5ad file.

HBC

Run:

python HBC_generate_data.py

The generated files will be saved under:

../generate_data/HBC/

If metadata.tsv is available, the script will automatically load manual annotations. Otherwise, it will run in unsupervised mode.

⸻

Training and Evaluation

DLPFC

Run:

python DLPFC_test.py

The script trains CIGST on the selected DLPFC slice and evaluates clustering performance using ARI and NMI.

Results will be saved under:

./result/DLPFC/

HBC

Run:

python HBC_test.py

If ground-truth annotations are available, ARI and NMI will be reported. Otherwise, the best epoch is selected according to the training loss.

Results will be saved under:

./result/HBC/

⸻

Main Outputs

For each dataset, the scripts save:

CIGST.jpg              # Spatial clustering visualization
CIGST_umap_emb.jpg     # UMAP / PAGA visualization
CIGST_emb.csv          # Learned spot-level embeddings
CIGST_idx.csv          # Predicted cluster labels
CIGST.h5ad             # AnnData object with results

For datasets with manual annotations, the following file is also saved:

Manual_Annotation.jpg

⸻

Requirements

The code mainly depends on:

python
numpy
pandas
scipy
scikit-learn
torch
scanpy
anndata
matplotlib
networkx
python-louvain

Optional dependency:

rpy2
mclust



