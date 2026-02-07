# MirAI: A Next-Gen Pathway for Early Alzheimer's Screening


**MirAI** is an integrated, non-invasive AI framework designed to identify individuals at elevated risk of Alzheimer’s Disease (AD) before significant cognitive decline occurs. By utilizing a **multi-stage, escalation-based architecture**, MirAI bridges the gap between scalable community screening and precision neurological diagnostics.

## 🚀 Project Overview

Current Alzheimer's diagnostics (PET scans, CSF analysis) are invasive, expensive, and unscalable for population-level screening. MirAI solves this by deploying a cost-effective, three-tier triage system:

1.  **Tier 1 (Clinical Screening):** Filters population using non-invasive demographics & functional scores.
2.  **Tier 2 (Genetic Stratification):** Refines risk for flagged individuals using *APOE* genotype.
3.  **Tier 3 (Biomarker Triage):** Provides definitive risk assessment using high-precision plasma biomarkers (pTau-217), strictly for high-risk cases.

## 🧠 System Architecture

MirAI uses a **Cascaded Inference Model** where the output probability of one stage serves as a "prior" feature for the next. This prevents data leakage and mimics real-world clinical workflows.

| Stage | Input Modality | Algorithm | Key Features | Performance (AUC) |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | **Clinical** | XGBoost | Age, Gender, Education, FAQ, ECog (Subjective Memory) | **0.908** |
| **Stage 2** | **Genetic** | XGBoost | *Stage 1 Risk Score* + *APOE* $\epsilon$4 Allele Count | **0.919** |
| **Stage 3** | **Biomarker** | XGBoost | *Stage 2 Risk Score* + Plasma pTau-217, A$\beta$42/40, NfL | **0.94+** |

![MirAI Architecture Badge](https://img.shields.io/badge/Architecture-Cascaded_XGBoost-blue)
![Python Version](https://img.shields.io/badge/Python-3.9%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
