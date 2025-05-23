# HGEMMD: Hypergraph-Enhanced Multimodal Dynamics for Patient Classification via Intra- and Inter-Sample Fusion
Multimodal learning is increasingly pivotal in biomedical research, where multi-omics technologies enable comprehensive characterization of diverse molecular layers. Despite their potential, effective integration of heterogeneous omics data remains challenging due to high dimensionality, modality inconsistency, and complex inter-sample dependencies. Existing methods primarily focus on *intra-sample* feature fusion, often overlooking **high-order structural relationships** across samples, which limits their ability to capture system-level interactions. To address these challenges, we propose **HyperGraph-Enhanced MultiModal Dynamics (HGEMMD)** — a novel framework that **simultaneously models intra-sample modality fusion and inter-sample high-order associations** for robust multi-omics integration and classification.

HGEMMD features the following core innovations:

- ✅ A **multimodal dynamics module** that alleviates data sparsity and modality heterogeneity.
- ✅ A **modality-aware hypergraph**, where each hyperedge connects semantically or functionally related samples, enabling the modeling of **non-pairwise dependencies**.
- ✅ A **relational consistency learning strategy** that aligns sample-level relational patterns before and after hypergraph propagation, preserving local semantics while ensuring global structural coherence.

Extensive experiments on benchmark multi-omics datasets demonstrate that **HGEMMD** consistently outperforms state-of-the-art approaches, validating its effectiveness in **robust and trustworthy multimodal integration**.

## 📚 Data Sources and Implementation Notes

- The data used in the paper can be obtained through the following links [https://github.com/txWang/MOGONET](https://github.com/txWang/MOGONET). 
- The code is implemented on the code provided by MMDynamics [https://github.com/TencentAILabHealthcare/mmdynamics](https://github.com/TencentAILabHealthcare/mmdynamics).
- If you have any questions, please contact me via the following email [ljwstruggle@gmail.com](ljwstruggle@gmail.com).

## 🚀 Usage
To reproduce the results, simply run:

```bash
python HGEMMD.py
```

## 📦 Requirements
* scikit-learn
* pytorch

## ⚠️ Disclaimer
This code is intended for **academic research only** and is **not approved for clinical use**.

## 📜 License
All rights reserved. Redistribution or commercial use is prohibited without prior written permission.