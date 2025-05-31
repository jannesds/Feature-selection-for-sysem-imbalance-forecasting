# Feature-selection-for-sysem-imbalance-forecasting
This project explores and compares different **feature selection methods** for improving **system imbalance forecasting** models. The goal is to identify the most relevant predictors for short-term electricity imbalance in Belgium.
## Abstract

In the context of liberalized electricity markets and increasing reliance on intermittent renewable energy sources, accurate forecasting of system imbalance has become essential for maintaining grid stability and enabling sound operational decisions. This thesis introduces a structured evaluation of feature selection methods, filling a notable gap in the literature on short-term imbalance forecasting in Belgium.

A structured methodology is developed, combining lag-based feature engineering across multiple time resolutions with a comparative assessment of filter, wrapper, and embedded feature selection methods. Two forecasting models are employed: a multivariate linear regression and a shallow multilayer perceptron (MLP), both evaluated under a time-series-aware cross-validation strategy.

The results demonstrate that appropriate feature selection improves forecasting performance by up to **6%** compared to the benchmark model used by the Belgian transmission system operator, when applied to the same model. Additionally, feature selection significantly reduces model complexity by identifying compact and informative input subsets, while providing insights into the temporal and contextual relevance of key predictors.

Beyond quantitative improvements, this work contributes to model interpretability by highlighting the types of features most consistently selected across methods and models. Two final feature sets are proposed for operational use, one for each model, and the full implementation is made publicly available, offering a foundation for future adaptation and extension as the electricity landscape evolves.

