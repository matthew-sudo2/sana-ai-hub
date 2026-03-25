# README_snippet.md

## What to Tell the Judges

### Dataset Quality Assessment Model

**Our Approach:**
We model dataset quality using statistical meta-features extracted from real-world datasets (finance, healthcare, survey, census, titanic) and an Isolation Forest anomaly detector for domain-independent quality scoring.

**Key Features:**
- **6 Statistical Meta-Features**: missing_ratio, duplicate_ratio, numeric_ratio, constant_columns, avg_variance, avg_skewness
- **Real-World Training Data**: 115 labeled variations across 5 diverse datasets
- **Unsupervised Learning**: Isolation Forest learns what "good quality" looks like without explicit labels
- **Interpretable**: Quality score 0-100 with explainable feature contributions
- **Fast**: ~5ms per dataset assessment

**Why This Matters:**
- Domain-independent: works on any tabular dataset
- Scalable: can be integrated into data pipelines
- Practical: identifies specific quality issues (missing values, duplicates, type errors, variance, distribution skew)

**Technical Stack:**
- scikit-learn (Isolation Forest)
- pandas, numpy, scipy
- Pathlib for cross-platform paths
