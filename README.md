# Feature-selection-for-sysem-imbalance-forecasting
This project explores and compares different **feature selection methods** for improving **system imbalance forecasting** models. The goal is to identify the most relevant predictors for short-term electricity imbalance in Belgium.

**Author: Jannes De Sutter**  
MSc Mechanical Engineering at KU Leuven  
Academic year 2024–2025

## Abstract

In the context of liberalized electricity markets and an increasing share of intermittent renewable energy sources, accurate forecasting of system imbalance has become critical for maintaining grid stability and supporting operational decisions. This thesis presents a structured evaluation of feature selection methods, addressing a gap in the literature on short-term imbalance forecasting in Belgium.

The methodology relies on constructing lagged versions of input variables across multiple time resolutions, followed by a comparative analysis of filter, wrapper and embedded feature selection techniques. Two forecasting models are considered: a multivariate linear regression (LR) and a shallow multilayer perceptron (MLP), both evaluated using a time-series-aware cross-validation approach. The results indicate that appropriate feature selection can improve forecasting performance by up to 6% relative to the benchmark model currently used by the Belgian transmission system operator, when applied to the same predictive model. In addition, feature selection contributes to a significant reduction in model complexity by identifying compact and informative input subsets, while also highlighting the computational costs associated with the selection process.

Beyond quantitative improvements, this work provides insight into the temporal and contextual relevance of individual features. Several variables are consistently selected across methods and models, showing high importance when included. Wind and solar generation features stand out, einforcing the link between renewable production and system imbalance. Furthermore, recurring patterns in cross-border nominations, particularly with France and the Netherlands, suggest their notable impact on imbalance dynamics. 

Two final feature sets are proposed for operational use, one for each model, and the full implementation is made publicly available, offering a foundation for future adaptation and extension as the electricity landscape evolves and additional data sources become available.

Below is a short guide to the folders:

- **`A_Filter/`** – Filter methods (e.g. correlation, mutual information).  
- **`B_Wrapper/`** – Wrapper methods using backtesting (e.g. Sequential Feature Selection).  
- **`C_Embedded/`** – Embedded methods (e.g. Lasso, model-based importance).  
- **`data/`** – Preprocessed datasets with lagged features from various time resolutions.  
- **`models/`** – Saved models and results.  
- **`logs_vsc/`** – Experiment logs and training metrics.  
- **`source/`** – Core utilities for preprocessing, training, and evaluation.  
- **`MastersThesis_DeSutter_Jannes.pdf`** – Full thesis document.  
- **`README.md`** – Project overview and folder guide.
