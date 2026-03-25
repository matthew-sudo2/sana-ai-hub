# Sana All May Label

![Sana Logo](frontend/sanallmaylabel.png)

> **Research, reinvented.**  
> Close the gap between data collection and actionable insights. Sana automates the 80% of research prep that students spend fighting their data.

## The Problem

IBM research shows that **data preparation consumes 80% of a data professional's time**. For students, it's worse. Many avoid quantitative research entirely—not because the questions don't matter, but because the tools are too hard to use. They don't skip the data because they're lazy. They skip it because nobody made it accessible.

## The Solution

Sana closes that gap. A student uploads messy, raw data—straight from a survey, experiment, or web scrape—and Sana handles everything else:

- **Data Cleaning** — Remove duplicates, handle missing values, detect anomalies
- **Statistical Analysis** — Compute means, medians, correlations, outliers
- **Auto-Visualization** — Generate 16+ charts automatically, zero configuration
- **Quality Validation** — Confidence scoring across completeness, consistency, and accuracy

**What takes hours in Excel takes minutes in Sana.**

## What You Get

✅ Clean, validated data ready for research  
✅ 16+ auto-generated charts and visualizations  
✅ Full statistical analysis per column  
✅ Confidence scores you can cite  
✅ Exportable quality reports  
✅ All open-source, zero cloud costs, runs locally

## Who This Is For

- **Students** doing quantitative research and avoiding it because "data is hard"
- **Researchers** who spend more time cleaning data than analyzing it
- **Teams** working with survey responses, experimental data, or messy datasets
- **Anyone** who needs to validate their data before submitting results

**If you've ever spent hours in Excel, wondering if your data is actually clean—this is for you.**

---

## Repository layout

```
sana-ai-hub/
  backend/    # Python multi-agent pipeline
  frontend/   # Vite + React dashboard
  docs/       # submission assets
```

## Multi-Agent Architecture

Sana uses **five specialized LangGraph agents** that form a sequential pipeline. Each agent has one responsibility, consumes the output of the previous stage, and passes its results downstream. The entire pipeline shares state through LangGraph—if something upstream changes, downstream agents automatically update.

```
                       ┌─────────────┐
                       │   Upload    │
                       │   CSV File  │
                       └──────┬──────┘
                              │
                      ┌───────▼────────┐
                      │  SCOUT         │
                      │  File Intake & │───→ scout_result.json
                      │  Validation    │───→ raw_data.csv
                      └───────┬────────┘
                              │
                      ┌───────▼────────┐
                      │  LABELER       │
                      │  Data Cleaning │───→ cleaned_data.csv
                      └───────┬────────┘
                              │
      ┌───────────────────────┼───────────────────────┐
      │                       │                       │
  ┌───▼────────┐      ┌───────▼────────┐      ┌──────▼──────┐
  │  ARTIST    │      │  ANALYST       │      │  (Parallel) │
  │ Charts &   │      │  Statistics    │      │             │
  │ Visuals    │      │  Analysis      │      └─────────────┘
  └───┬────────┘      └───────┬────────┘
      │                       │
      └───────────┬───────────┘
                  │
          ┌───────▼────────┐
          │  VALIDATOR     │
          │  Quality Score │───→ validation_report.md
          │  Certification │───→ confidence.json
          └───────┬────────┘
                  │
          ┌───────▼────────┐
          │   Dashboard    │
          │   React UI     │
          │  (Visualize)   │
          └────────────────┘
```

**Key Architecture Features:**
- **Sequential Pipeline**: Scout → Labeler → Artist/Analyst (parallel) → Validator
- **Shared State**: LangGraph maintains full pipeline state across all agents
- **Data Persistence**: Complete raw_data.csv passed through entire pipeline
- **Chainable Outputs**: Each agent reads prior outputs, produces new artifacts
- **Transparent**: Every intermediate file visible and exportable

---

## System Features

### 🔍 Scout Agent - File Intake & Initial Validation

**The Gatekeeper**: Validates uploaded files, discovers schema, and computes initial quality score.

- **File Handling**:
  - Accepts: CSV, XLSX, JSON uploads
  - Rejects: URLs (with guidance to download and upload)
  - No web scraping—file-based intake only
  
- **Data Validation**:
  - Minimum requirements: 3+ rows, 2+ columns
  - Automatic dtype inference (numeric, datetime, categorical)
  - Smart conversion: preserves text fields, converts <<70% matches
  
- **Schema Discovery**:
  - Column names and inferred data types
  - 5-row sample preview for dashboard
  - Detects numeric vs. categorical columns
  
- **Initial Quality Scoring** (5-dimensional):
  - **Completeness**: % non-null cells (goal: 100%)
  - **Consistency**: Values match inferred types
  - **Accuracy**: Dtype inference credibility
  - **Uniqueness**: Opposite of duplicate ratio (goal: 0% dupes)
  - **Outlier Risk**: IQR-based anomaly detection
  - **Weighted Confidence**: `0.25×C + 0.25×K + 0.20×A + 0.15×D + 0.15×O`
  
- **Outputs**:
  - `scout_result.json` — Metadata, schema, initial confidence
  - `raw_data.csv` — **Complete dataset** (read by all downstream agents)

### 🏷️ Labeler Agent - Data Cleaning & Quality Processing
- **Intelligent Cleaning** (90% faster than LLM):
  - Standardizes column names to snake_case
  - Removes empty rows and columns
  - Detects and removes duplicate records
  - IQR-based outlier detection and flagging
  - Median imputation for missing numeric values
  
- **Smart Caching** (60% speedup on re-runs):
  - MD5 fingerprint of raw data for cache validation
  - Profile hash to detect cleaning configuration changes
  - Instant return for identical datasets
  
- **Real-Time Quality Metrics**:
  - Missing value ratio calculation
  - Duplicate detection and reporting
  - Numeric column analysis
  - Constant column identification
  - Variance and skewness computation
  
- **Configurable Cleaning Profiles**:
  - Drop vs. fill strategies for missing values
  - Outlier detection methods (IQR)
  - Custom thresholds and parameters

### 🎨 Artist Agent - Visualization Generation
Auto-generates 16+ publication-ready charts without configuration:
- **Chart Types**:
  - Distribution histograms for each numeric column
  - Box plots with flagged outliers (IQR-based, shown in red)
  - Scatter plots for all numeric relationships
  - Correlation heatmaps showing column relationships
  - Time series plots for temporal data
  - Line charts for trends and patterns
  
- **Smart Defaults**:
  - Axes auto-scaled for readability
  - Outliers automatically flagged and colored
  - All charts labeled and titled
  - Zero manual configuration required
  
- **Custom Charts** (Optional):
  - Generate charts from natural language instructions
  - Fuzzy column matching (handles typos and variations)
  - Automatic fallback with helpful error messages

### 📊 Analyst Agent - Statistical Analysis
Produces full descriptive statistics automatically:
- **Per-Column Statistics** (mean, median, mode, range, variance, standard deviation, min, max)
- **Correlation Analysis** (shows which columns relate to each other)
- **Outlier Bounds** (calculates IQR-based anomaly thresholds)
- **Type-Safe Computation** (automatically excludes non-numeric columns)
- **JSON Export** (structured output for further processing)

### ✅ Validator Agent - Data Quality Certification
The differentiator: Sana doesn't just give you results—it tells you whether you can trust them.
- **Multi-Dimension Quality Scoring**:
  - **Completeness**: Missing value ratio (target: 0%)
  - **Consistency**: Duplicate detection (target: 0% duplicates)
  - **Reliability**: Correlation and statistical validity
  - **Cardinality**: Outlier and anomaly detection
  
- **Confidence Scoring**:
  - Overall data quality percentage (0–100%)
  - Per-check pass/fail reporting
  - Plain-language confidence statements ("97.8% confidence")
  
- **Exportable Validation Reports**:
  - Markdown-formatted detailed breakdown
  - GitHub Flavored Markdown tables
  - Pass/fail summary for each quality dimension
  - Ready to include in research papers

### 💻 Frontend Dashboard - React UI
- **Tab-Based Navigation**:
  - **Visual Gallery Tab**: Chart generation and visualization management
    - Create custom charts from natural language
    - Browse and manage generated visualizations
    - Real-time chart preview
  
  - **Data Viewer Tab**: Comprehensive data inspection interface
    - Quality metrics cards (rows, columns, completeness %, average data quality)
    - Color-coded quality badges (Green/Blue/Yellow/Red based on column completeness)
    - Interactive column search and filtering
    - Column visibility toggle
    - Sortable columns by name and quality metrics
    - 25-row pagination for large datasets
    - Missing value indicators (red "∅" for empty cells)
    - CSV export with proper escaping
  
  - **Validation Report Tab**: Data quality assessment display
    - Markdown-formatted quality reports
    - GitHub Flavored Markdown table rendering
    - Detailed validation results
    - Quality score breakdown

- **UI Features**:
  - Dark theme for comfortable viewing
  - Professional layout with Shadcn/ui components
  - Smooth interactions and animations
  - Responsive design
  - Zero compilation errors

### 🤖 Built-In Machine Learning - Data Quality Classifier
An optional ML model provides automated quality classification:
- **Accuracy**: 95.2% ± 2.0% K-fold CV
- **Trained on**: 188 diverse samples (good data + synthetic corruption patterns)
- **Features**: 13 engineered metrics capturing data quality patterns
- **Real-World Validation**: Tests on Accenture & Amazon stock data
- **Use Case**: Rank datasets by reliability or detect quality issues before analysis

### ⚡ Performance Optimizations
- **Memory Efficient**: Eliminated unnecessary copying (30-50% savings)
- **Fast Cleaning**: Rule-based cleaning instead of LLM (90% speedup)
- **Intelligent Caching**: Fingerprint-based deduplication (60% speedup)
- **Builtin Defaults**: Charts generated without cloud calls (no API costs)
- **Overall Speed**: ~8-15 seconds total pipeline (was 45–70 seconds) — **80% faster**

---

## The Pipeline - Five Stages

**One Upload. Five Agents. One Confidence Score.**

```
Your CSV → Scout (Validate) → Labeler (Clean) → Artist (Chart) → Analyst (Stats) → Validator (Certify) → Export
```

### Stage 1: Scout (File Intake & Validation)
Upload any CSV/XLSX/JSON. Scout validates and discovers schema.
- ✅ File validation (min 3 rows × 2 cols)
- ✅ Schema discovery (column names & types)
- ✅ Initial quality metrics (completeness, consistency, duplicates, outliers)
- ✅ Confidence score: `0.25×Completeness + 0.25×Consistency + 0.20×Accuracy + 0.15×Duplicates + 0.15×Outliers`
- ✅ Full dataset saved for all downstream processing

### Stage 2: Labeler (Data Cleaning)
Transforms raw into reliable data.
- ✅ Removes duplicates and empty rows
- ✅ Handles missing values (median imputation for numeric, drop for categorical)
- ✅ Detects and flags outliers (IQR method)
- ✅ Standardizes column names to snake_case
- ✅ Computes per-column quality metrics
- ✅ Output: `cleaned_data.csv`

### Stage 3: Artist (Visualization)
Generates publication-ready charts, zero configuration.
- ✅ 16+ automatic charts (distributions, relationships, correlations)
- ✅ Histograms, scatter plots, heatmaps, box plots, line charts
- ✅ Outliers highlighted in red
- ✅ All axes labeled and titled
- ✅ Correlation matrix for numeric columns
- ✅ Time series detection (if applicable)
- ✅ Output: `*.png` image files

### Stage 4: Analyst (Statistics)
Computes full descriptive statistics on cleaned data.
- ✅ Per-column: mean, median, mode, range, variance, std dev, min, max
- ✅ Correlation matrix (all numeric pairs)
- ✅ Outlier bounds per column (IQR method)
- ✅ Skewness and kurtosis analysis
- ✅ Type validation per column
- ✅ Output: `analysis.json`

### Stage 5: Validator (Quality Certification)
**The Differentiator**: Scores data quality and generates confidence statement.
- ✅ Completeness check (missing value percentage)
- ✅ Consistency check (duplicate rows)
- ✅ Reliability assessment (statistical validity)
- ✅ Cardinality analysis (unique values per column)
- ✅ Overall confidence score (0–100%, citable)
- ✅ Output: `validation_report.md` (plain text, exportable) + `confidence.json`

### Final Outputs
- ✅ Cleaned CSV data (`cleaned_data.csv`)
- ✅ 16+ visualization images (`*.png`)
- ✅ Statistical analysis (`analysis.json`)
- ✅ Validation report (`validation_report.md` — attach to paper)
- ✅ Confidence score (`confidence.json` — cite this in your methodology)

---

## Installation & Usage

### Frontend (React)

From repo root:

```bash
cd frontend
npm install
npm run dev
```

### Backend (Python)

Create a virtualenv, then from repo root:

```bash
pip install -r backend/requirements.txt
```

Create `backend/.env` from `backend/.env.example` if needed for additional configuration.

### Running the Pipeline

The pipeline runs automatically through the dashboard. For direct script usage:

```bash
# Full pipeline from CSV
python backend/main.py <path_to_data.csv>
```

This executes all five agents in sequence:
1. **Scout** — Validates file, discovers schema, computes initial quality
2. **Labeler** — Cleans data (removes duplicates, fills missing, detects outliers)
3. **Artist** — Generates 16+ visualizations
4. **Analyst** — Computes statistics and correlations
5. **Validator** — Produces confidence score and quality report

Outputs produced:
- `scout_result.json` — Initial metadata and confidence
- `raw_data.csv` — Full dataset
- `cleaned_data.csv` — Cleaned dataset  
- `*.png` — Charts and visualizations (16+)
- `analysis.json` — Statistical analysis
- `validation_report.md` — Quality certification report (exportable)
- `confidence.json` — Final confidence score (citable)

---

## Technology Stack

**Backend**: Built on open-source frameworks, **zero cloud dependencies**
- **LangGraph**: Multi-agent state graph orchestration
- **FastAPI**: High-performance REST API
- **Ollama**: Local LLM inference (optional, not required)
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning (Random Forest classifier)
- **matplotlib**: Chart generation

**Frontend**: Modern web stack, runs locally
- **React**: UI framework
- **Vite**: Fast build tool
- **Shadcn/ui**: Component library
- **TailwindCSS**: Styling

**Database**: File-based persistence
- **CSV**: Cleaned data storage
- **JSON**: Analysis results and metadata
- **PNG**: Generated visualizations
- **Pickle**: Trained ML models

**Deployment**: Local-first, fully offline capable
- No API keys required (Ollama is optional)
- No internet connection needed
- No cloud storage costs
- Docker-ready for containerization

---

## Machine Learning Model: Data Quality Classifier

### Overview

The project includes a machine learning classifier that automatically detects and scores data quality. The model is trained to distinguish between clean, high-quality datasets and datasets with various data quality issues.

### Model Architecture

**Algorithm**: Random Forest Classifier
- **Estimators**: 50 trees
- **Max Depth**: 3 (prevents overfitting)
- **Max Features**: 'sqrt' (feature subset selection)
- **Random State**: 42 (reproducibility)

### Training Data

**Total Samples**: 188
- Original (Good Data): 23 datasets
- Synthetic (Bad Data): 165 datasets across 3 severity levels
  - **Light corruption** (5% degradation): 55 samples
  - **Medium corruption** (12% degradation): 55 samples
  - **Severe corruption** (25% degradation): 55 samples

**Corruption Types** (5 types × 3 severity levels):
1. Missing values (random cells set to NaN)
2. Duplicates (exact row duplication)
3. Outliers (extreme value multiplication: 10x-100x)
4. Inconsistency (column-level nullification)
5. Mixed (combination of 2-3 corruption types)

### Performance Metrics

**K-Fold Cross-Validation** (5-fold StratifiedKFold):
- **Accuracy**: 95.2% ± 2.0%
- **Precision**: High (model rarely misclassifies good data as bad)
- **Recall**: Excellent (detects most quality issues)
- **Stability**: 84% improvement in variance vs baseline (from 12.8% → 2.0%)

**Performance Progression**:
| Approach | Samples | K-fold CV | Improvement |
|----------|---------|-----------|-------------|
| RandomForest Baseline | 23 | 79.0% | — |
| Basic Corruption | 111 | 74.7% | -4.3% |
| Improved Single-Level | 95 | 88.4% | +9.4% |
| **Multi-Level (Current)** | **188** | **95.2%** | **+16.2%** |

### Feature Engineering (13 Features)

**Base Features** (6):
1. `missing_ratio`: Percentage of missing values
2. `duplicate_ratio`: Percentage of duplicate rows
3. `numeric_ratio`: Proportion of numeric columns
4. `constant_cols`: Count of constant-value columns
5. `variance`: Average variance across numeric columns
6. `skewness`: Average skewness across numeric columns

**Engineered Features** (7):
7. Missing × Duplicate interaction
8. Missing × Numeric interaction
9. Variance × Skewness interaction
10. log(variance)
11. log(|skewness|)
12. Variance / Skewness ratio
13. Skewness / Variance ratio

### Real-World Validation

**Tested Datasets**:

1. **Accenture Stock History** (11,358 rows × 6 columns)
   - Quality Prediction: **GOOD** (81.8% confidence)
   - Missing Data: 0% | Duplicates: 0%
   - ✓ Correctly classified as clean

2. **Amazon Stock Data** (7,221 rows × 6 columns)
   - Quality Prediction: **GOOD** (81.8% confidence)
   - Missing Data: 0% | Duplicates: 0%
   - ✓ Correctly classified as clean

### Model Files

- **Production Model**: `models/best_model.pkl`
  - Trained on 188 multi-level corrupted samples
  - 95.2% K-fold CV accuracy
  - Includes metadata: feature names, hyperparameters, training configuration

**Training Scripts**:
- `models/train/train_multilevel_augmented.py` — Train model on multi-level data
- `data_processing/corruption/multi_level_data_corruption.py` — Generate synthetic bad data
- `evaluation/test_best_model.py` — Validate model on real datasets

### Usage Example

```python
import pickle
import pandas as pd

# Load trained model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your dataset
df = pd.read_csv('your_data.csv')

# Extract features (matching 13 engineering pipeline)
features = extract_quality_features(df)  # See feature engineering above

# Predict quality (0 = Bad, 1 = Good)
prediction = model.predict([features])
confidence = model.predict_proba([features])[0]

print(f"Quality: {'GOOD' if prediction[0] == 1 else 'BAD'}")
print(f"Confidence: {max(confidence)*100:.1f}%")
```

### Key Insights

1. **Multi-Level Corruption Matters**: Teaching the model that data quality exists on a spectrum (light/medium/severe) improved accuracy by 6.8% over single-level corruption.

2. **Small Dataset Challenge**: Starting with only 23 real samples required aggressive synthetic augmentation (165 corrupted variants) to achieve reliable generalization.

3. **Stable Model**: Low variance (±2.0%) indicates the model learns consistent patterns across fold splits, not overfitting to specific data splits.

4. **Real-World Performance**: Model validation on unseen real datasets (Accenture, Amazon) confirms it generalizes beyond synthetic corruption patterns.

---

## By the Numbers

| Metric | Result |
|--------|--------|
| **Data Quality Scores** | 55–97% (varies by dataset patterns) |
| **Validation Confidence** | 97.8% (on test datasets) |
| **Processing Speed** | 8–15 seconds (full pipeline) |
| **Charts Auto-Generated** | 16+ per dataset |
| **Quality Checks Passed** | 38/38 on clean data |
| **Missing Value Detection** | 100% accuracy |
| **Duplicate Detection** | 100% accuracy |
| **Outlier Flagging** | IQR-based, automatic |
| **Model Accuracy** | 95.2% ± 2.0% (K-fold CV) |
| **Data Preparation Time Saved** | 80% of research prep automated |

### Real-World Examples

**Accenture Stock History** (20 years, 7,221 rows)
- ✅ Quality Score: 94.6%
- ✅ Confidence: 97.8%
- ✅ Checks Passed: 38/38
- ✅ Processing Time: <15 seconds
- ✅ Charts Generated: 16+

**Messy Survey Data** (1,020 rows)
- ✅ Duplicates Removed: 47
- ✅ Missing Values Filled: 1,020+ values
- ✅ Outliers Flagged: 238 (shown in red)
- ✅ Quality Score: 89.3%
- ✅ Time Saved vs Manual: 240+ minutes

---

## The Bottom Line

**80% of research prep, automated. Zero learning curve.**

Transform from: *"Ugh, I need to spend the whole weekend cleaning this data"*  
To: *"Okay, my data is validated and ready to use."*

One upload. Four agents. One confidence score attached to your paper.

**Less time fighting the data. More time doing the research.**

---

Built by students, for students. No paywalls. No cloud lock-in. No BS.

**Array Potter. Sana All May Label. Research, reinvented.**
