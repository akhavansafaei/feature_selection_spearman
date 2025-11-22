# Feature Selection using Spearman Correlation

A comprehensive toolkit implementing multiple feature selection algorithms for dimensionality reduction in machine learning pipelines. This project focuses on identifying and removing redundant features using correlation-based analysis, with a primary emphasis on Spearman rank correlation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Algorithms Implemented](#algorithms-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Methodologies](#methodologies)
- [Outputs](#outputs)
- [Requirements](#requirements)

## Overview

This project implements six distinct feature selection algorithms that progressively evolve from simple threshold-based correlation filtering to sophisticated sequential forward selection with merit-based optimization. The main objective is to reduce dataset dimensionality while preserving the most relevant and non-redundant features for classification tasks.

### Key Objectives

- Remove redundant features based on correlation analysis
- Preserve features with high statistical significance
- Balance feature-class relevance with feature independence
- Provide multiple algorithmic approaches for comparison

## Features

- **Multiple Algorithms**: Six different feature selection strategies ranging from basic to advanced
- **Dual Correlation Methods**: Support for both Spearman and Pearson correlation
- **Statistical Significance**: P-value based feature selection
- **Merit Function Optimization**: Advanced evaluation combining relevance and redundancy
- **Comprehensive Outputs**: Correlation matrices, feature groups, and reduced datasets

## Project Structure

```
feature_selection_spearman/
├── spearman_feature_slection_1st_algorithm.ipynb    # Basic correlation-based selection
├── spearman_feature_slection_2nd_algorithm.ipynb    # Enhanced selection with tie handling
├── spearman_feature_slection_4th_algorithm.ipynb    # Hybrid correlation-variance approach
├── duplicates.txt                                    # Output: P-values for feature groups
├── download.png, download(1-3).png                   # Visualization outputs
├── گزارش.pdf, گزارش.docx                             # Project report (Persian)
└── new/
    ├── 3/
    │   └── spearman_feature_slection_3rd_algorithm.ipynb
    ├── 4/
    │   └── spearman_feature_slection_4th_algorithm.ipynb
    ├── 5/
    │   └── spearman_feature_slection_5th_algorithm.ipynb
    ├── 6/
    │   └── 6th_algorithm.ipynb                       # Sequential forward selection
    └── اصلاحیه/                                      # Revised versions
        ├── 3/spearman_feature_slection_3rd_algorithm.ipynb
        ├── 4/spearman_feature_slection_4th_algorithm.ipynb
        └── 5/spearman_feature_slection_5th_algorithm.ipynb
```

## Algorithms Implemented

### 1st Algorithm: Basic Correlation-based Feature Selection

**Location**: `spearman_feature_slection_1st_algorithm.ipynb`

**Methodology**:
- Computes Spearman correlation matrix for all features
- Identifies correlation groups using a threshold (default: 0.95)
- Within each correlated group:
  - Selects feature with minimum p-value (highest statistical significance)
  - Among tied p-values, selects feature with highest absolute correlation to target class
- Separates features into selected and independent (kept) columns

**Key Functions**:
- `read_excel_to_dict()` - Load p-value data from Excel
- `calculate_column_correlations()` - Compute feature-class correlations
- `detect_correlation_groups()` - Identify highly correlated feature groups
- `reduce_correlated_columns()` - Main reduction algorithm

### 2nd Algorithm: Enhanced Correlation-based Selection

**Location**: `spearman_feature_slection_2nd_algorithm.ipynb`

**Methodology**:
- Similar to 1st algorithm with improved tie handling
- When multiple features have the same minimum p-value, **all candidates are selected** (more conservative approach)
- Creates broader feature sets for evaluation

**Advantage**: Better handles cases where statistical significance is tied

### 3rd Algorithm: Correlation Group Detection

**Location**: `new/3/spearman_feature_slection_3rd_algorithm.ipynb`

**Methodology**:
- Enhanced correlation group detection
- Improved handling of complex correlation structures
- Separate outputs for selected features, independent features, and final dataset

**Revision**: Corrected version available in `new/اصلاحیه/3/`

### 4th Algorithm: Hybrid Correlation-Variance Approach

**Location**: `spearman_feature_slection_4th_algorithm.ipynb`, `new/4/`

**Methodology**:
- Combines correlation-based selection with additional heuristics
- Considers maximum correlation thresholds and variance
- Similar p-value and correlation strategies as earlier algorithms

**Revision**: Improved version in `new/اصلاحیه/4/`

### 5th Algorithm: Pairwise Comparison with Dual Correlation Methods

**Location**: `new/5/spearman_feature_slection_5th_algorithm.ipynb`

**Methodology**:
- Implements both **Spearman** and **Pearson** correlation analysis
- Pairwise comparison approach examining each feature pair individually
- Runtime switching between correlation methods

**Key Feature**: Dual-method capability for robustness testing

**Revision**: Corrected version in `new/اصلاحیه/5/`

### 6th Algorithm: Sequential Forward Selection with Merit Function ⭐

**Location**: `new/6/6th_algorithm.ipynb`

**Methodology**: **Most Sophisticated Algorithm**
- Implements Sequential Forward Selection (SFS) with backtracking
- Uses priority queue for candidate management
- Evaluates feature subsets using Merit Function:
  - `Merit = rcf - (k × rff)`
    - **rcf**: Average feature-class correlation (relevance)
    - **rff**: Average feature-feature correlation (redundancy)
    - **k**: Feature count (penalizes large subsets)
- Backtracking mechanism prevents local optima

**Search Algorithm**:
1. Find initial best feature by class correlation
2. Create priority queue with initial feature subset
3. Iteratively add features that maximize merit
4. Use backtracking for search space exploration
5. Terminate on convergence or backtrack limit

**Advantage**: Balances feature-class relevance with feature independence using advanced search strategy

## Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab

### Required Libraries

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn natsort openpyxl
```

Or install from requirements file (if available):

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage Example

```python
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Load your dataset
data = pd.read_excel('your_dataset.xlsx')

# Expected format:
# - First column: Object ID ('obj')
# - Second column: Class label ('class')
# - Remaining columns: Features (f1, f2, f3, ...)

# Run one of the algorithms by opening the corresponding notebook
# Example: spearman_feature_slection_1st_algorithm.ipynb
```

### Running the Notebooks

1. **Open Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Navigate** to desired algorithm notebook

3. **Update Dataset Path**: Modify the data loading cell to point to your Excel file

4. **Execute Cells**: Run all cells sequentially

5. **Review Outputs**: Check generated CSV files and correlation matrices

### Expected Dataset Format

Your Excel file should have the following structure:

| obj | class | f1  | f2  | f3  | ... | fn  |
|-----|-------|-----|-----|-----|-----|-----|
| 1   | 0     | 0.5 | 1.2 | 0.8 | ... | 2.1 |
| 2   | 1     | 0.3 | 1.5 | 0.9 | ... | 1.8 |
| ... | ...   | ... | ... | ... | ... | ... |

- **obj**: Object/Sample identifier
- **class**: Binary or multi-class label
- **f1, f2, ..., fn**: Feature values

## Datasets

The project references several Excel datasets (not included in repository):

- `data_u.xlsx` - Primary dataset (algorithms 1-3)
- `data_u2.xlsx` - Alternative dataset (algorithm 4)
- `data_u6.xlsx` - Dataset for algorithm 5
- `not_sort.xlsx` / `not_sort2.xlsx` - P-value reference files

**Note**: You will need to provide your own datasets in the expected format.

## Methodologies

### 1. Spearman Rank Correlation

- Non-parametric correlation measure
- Handles non-linear relationships
- Robust to outliers
- Primary method for feature-feature and feature-class relationships

### 2. P-value Based Selection

- Evaluates statistical significance of correlations
- Lower p-value indicates stronger evidence
- Primary criterion in algorithms 1-4

### 3. Threshold-based Correlation Grouping

- Groups features with correlation ≥ threshold (typically 0.95)
- Identifies redundant features for removal
- Outputs correlation matrices at various filtering levels

### 4. Merit Function Evaluation

Used in algorithm 6 for advanced feature subset evaluation:

```
Merit = rcf - (k × rff)

Where:
- rcf: Average feature-class correlation (relevance)
- rff: Average feature-feature correlation (redundancy)
- k: Number of features (penalty for large subsets)
```

**Goal**: Maximize class correlation while minimizing inter-feature redundancy

### 5. Search Strategies

- **Algorithms 1-5**: Deterministic, threshold-based elimination
- **Algorithm 6**: Heuristic search with backtracking
  - Sequential Forward Selection (SFS)
  - Priority queue for candidate exploration
  - Backtracking prevents getting stuck in local optima

## Outputs

Each algorithm generates various outputs including:

### Correlation Matrices
- `upper_tri_correlations_*.csv` - Upper triangular correlation matrix
- `lower_tri_correlations_*.csv` - Lower triangular correlation matrix
- `abs_correlations_*.csv` - Absolute correlation values
- `filtered_correlations_*.csv` - Correlations above threshold

### Feature Lists
- `selected_columns_*.csv` - Features selected from correlated groups
- `no_corr_columns_*.csv` - Independent features (no high correlations)
- `final_data_*.csv` - Complete reduced dataset

### Statistical Reports
- `duplicates.txt` - P-value groups showing feature correlation clusters
- Visualization plots (PNG format)

## Requirements

### Core Libraries

```
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
natsort>=7.1.0
openpyxl>=3.0.0
```

### Development Environment

- **Jupyter Notebook** or **JupyterLab**
- Python 3.7 or higher

## Algorithm Comparison

| Algorithm | Method | Complexity | Best For |
|-----------|--------|------------|----------|
| 1st | Basic threshold + p-value | Low | Quick dimensionality reduction |
| 2nd | Enhanced tie handling | Low | Conservative feature retention |
| 3rd | Improved group detection | Medium | Complex correlation structures |
| 4th | Hybrid with variance | Medium | Variance-aware selection |
| 5th | Dual correlation methods | Medium | Robustness verification |
| 6th | Sequential forward search | High | Optimal feature subset discovery |

## Key Insights

1. **Progressive Evolution**: The project shows progression from simple threshold-based methods to sophisticated optimization
2. **Multiple Approaches**: Provides flexibility to choose algorithm based on dataset characteristics
3. **Statistical Rigor**: Emphasizes p-value significance and correlation analysis
4. **Practical Implementation**: Ready-to-use Jupyter notebooks with clear documentation

## Notes

- Some documentation is in Persian (Farsi), suggesting this may be an academic project
- The 6th algorithm is the most computationally intensive but provides the best balance between relevance and redundancy
- Correlation threshold of 0.95 is default but can be adjusted based on requirements
- P-value data should be pre-computed and provided in Excel format

## Contributing

This appears to be a research/academic project. For questions or improvements, please refer to the original documentation in `گزارش.pdf`.

## License

Please refer to the project documentation for licensing information.

---

**Last Updated**: 2025-11-22
