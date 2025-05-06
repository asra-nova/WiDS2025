# Unraveling the Mysteries of the Female Brain: Sex Patterns in ADHD

**Authors**  
Reza Mansouri, Shakiba Mashini, Amir Sabbaghziarani and Geethanjali Nagaboina

Georgia State University


This repository contains the source code, data processing scripts, and model implementations accompanying our study titled **"Unraveling the Mysteries of the Female Brain: Sex Patterns in ADHD"**.

## Overview

Our work addresses the diagnostic disparities in Attention Deficit Hyperactivity Disorder (ADHD) by developing machine learning models that predict both ADHD diagnosis and biological sex from functional neuroimaging and behavioral data. We leverage a multi-modal dataset consisting of:

- Functional MRI (fMRI) connectome matrices
- Socio-demographic metadata
- Emotional and parenting profiles

The project integrates statistical learning, deep learning, and graph neural networks to explore sex-specific neural signatures of ADHD and improve early diagnostic accuracy—especially for underrepresented female cases.

## Highlights

- Multi-outcome prediction: Simultaneous classification of ADHD diagnosis and sex  
- Modalities: fMRI connectivity + tabular metadata  
- Models: Logistic Regression, Random Forest, MLPs, GNNs (GINConv, MVS-GCN)  
- Fusion techniques: Tensor Fusion and probability-level ensembling  
- Evaluation metric: Weighted F1* emphasizing underdiagnosed female ADHD cases  

## Installation

This project requires Python 3.9+. Recommended setup:

```bash
git clone https://github.com/asra-nova/WiDS2025.git
cd WiDS2025
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Results Summary

| Model                        | Validation F1* | Leaderboard F1* |
|-----------------------------|----------------|------------------|
| Logistic Regression         | **0.870**      | **0.730**        |
| MLP Tensor Fusion           | 0.733          | 0.472            |
| MLP Weighted Ensemble       | 0.711          | 0.461            |
| Ridge Classifier (Baseline) | 0.677          | 0.627            |
| SVM                         | 0.655          | 0.624            |
| MLP Merged                  | 0.645          | 0.423            |
| Random Forest               | 0.556          | 0.464            |

> Note: Leaderboard F1* reflects generalization to unseen i.i.d. test data.


## Acknowledgements

We thank the Healthy Brain Network (HBN) for providing the data used in this study, and the supporting faculty at Georgia State University for their guidance.

---

© 2024 Georgia State University. All rights reserved.
