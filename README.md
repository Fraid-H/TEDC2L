# TEDC2L: Cell-Lineage-Specific Gene Regulatory Network Inference & Driver Regulator Identification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TEDC2L** is a graph-based deep learning framework designed to infer cell-lineage-specific Gene Regulatory Networks (GRNs) and identify driver regulators. By incorporating multimodal enhancements (such as gene text embeddings, graph-text contrastive learning, and edge-level gating mechanisms) alongside advanced control principles (MDS, MFVS, MCES, and Energy-Optimal control), TEDC2L provides a robust pipeline for cell lineage analysis.

---

## Overview

![TEDC2L Architecture](docs/overview.png) *(Optional: Place your framework workflow figure here)*

- **Text-Embedding Enhanced GRN Inference:** Seamlessly integrates gene text embeddings via cosine similarity, KNN network augmentation, and graph-text contrastive alignment loss.
- **Edge-Level Gating Mechanism:** Uses a flexible gating parameter ($\\beta$) to balance node-level differential expression priors and edge-level text similarities.
- **Enhanced Driver Control Identifications:** Supports four complementary network control methods:
  1. **MDS** (Minimum Dominating Set)
  2. **MFVS** (Minimum Feedback Vertex Set)
  3. **MCES** (Minimum Controllability Edge Set)
  4. **Energy-Optimal Control**
- **Regulon-like Gene Modules (RGMs):** Evaluates downstream cell-level module activity profiles using `AUCell` scoring.

---

## Repository Structure

```text
TEDC2L/
├── TEDC2L.py                 # Main execution script and argument parser
├── TEDC2L_result_object.py    # Results container class with analytical methods
├── cell_lineage_GRN.py       # PyTorch Geometric GNN architecture & Encoder
├── driver_regulators.py      # Network control optimization solvers (MDS/MFVS/MCES/Energy)
├── visualize_results.py      # Publication-quality plotting and visualization script
├── utils.py                  # Data loading, text similarity processing, and helper functions
├── example_data/             # Example input data files
└── prior_data/               # Prior interaction network files
```

---

## Installation

### Prerequisites

Ensure you have Python 3.8+ and a CUDA-compatible PyTorch environment installed.

### Step-by-Step Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/TEDC2L.git
   cd TEDC2L
   ```

2. **Install core dependencies:**
   ```bash
   pip install torch torch-geometric pandas numpy scikit-learn scanpy networkx matplotlib seaborn matplotlib-venn tqdm
   ```

3. **Install `pyscenic` for RGM activity scoring:**
   ```bash
   pip install pyscenic
   ```

---

## Quick Start

You can run the full execution pipeline directly via `TEDC2L.py`:

```bash
python TEDC2L.py \
    --input_expData "../example_data/CRC_EMTAB8107_data_DE_C9.csv" \
    --input_priorNet "../prior_data/network-combined.csv" \
    --text_emb_path "../example_data/gene_embs_CRC_EMTAB8107_data_DE_C9.csv" \
    --out_dir "../output" \
    --cuda 0
```

### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--input_expData` | `str` | Path to the single-cell expression or differential expression matrix (`.csv`). |
| `--input_priorNet` | `str` | Path to the prior regulatory network file (`.csv`). |
| `--text_emb_path` | `str` | Path to the pretrained gene text embeddings (`.csv`). |
| `--out_dir` | `str` | Output directory to save inferred networks and control identification results. |
| `--cuda` | `int` | CUDA device ID (default: `0`). |

---

## Visualization

Once the execution completes, visualize all outputs (including Venn diagrams, UMAPs, driver influence scores, and heatmaps) using `visualize_results.py`:

```bash
python visualize_results.py --out_dir "../output"
```

---

## Output Files

After execution, the following result files will be generated in your specified `--out_dir`:

| Output File | Description |
| :--- | :--- |
| `cell_lineage_GRN.csv` | Inferred lineage-specific gene regulatory network. |
| `driver_regulators.csv` | Combined driver regulator rankings and identified driver flags. |
| `gene_embs.csv` | Learned low-dimensional gene embeddings. |
| `AUCell_mtx.csv` | RGM activity score matrix across cells. |
| `MDS_drivers.csv`, `MFVS_drivers.csv`, ... | Detailed driver lists per control method (MDS, MFVS, MCES, Energy-optimal). |
| `method_comparison_summary.csv` | Experiment parameters and driver overlap summaries. |
| `experimental_config.csv` | Logged parameters and execution settings. |

---

## Citation

If you use **TEDC2L** in your research, please cite our paper:

```bibtex
@article{TEDC2L2026,
  title={Cell-Lineage-Specific Gene Regulatory Network Inference & Driver Regulator Identification},
  author={Your Name and Co-authors},
  journal={Journal Name / BioRxiv},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```text
MIT License

Copyright (c) 2026 TEDC2L Developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
