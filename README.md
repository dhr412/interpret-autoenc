# Autoencoder Interpretability via Symbolic Regression

This repository explores the internal mechanisms of autoencoders using symbolic regression. By reverse-engineering the mapping from input space to latent space, we derive human-interpretable mathematical functions that explain how the model compresses information.

## What It Does

*   Trains autoencoders on tabular (UCI Forest Cover) and text (IMDB embeddings) data.
*   Extracts latent space representations ($z$). 
*   Uses Symbolic Regression (PySR) to discover closed-form equations $z_i = f(x)$ connecting inputs to latent dimensions.
*   Analyzes feature importance based on their frequency in the discovered symbolic formulas.
*   Derives symbolic laws governing reconstruction error as a function of latent dimension and feature properties.

## How It Works

The pipeline consists of the following steps:

1.  **Data Preparation**:
    *   **Tabular**: Preprocesses the UCI Forest Cover Type dataset (54 features).
    *   **Text**: Generates 384-dimensional embeddings for IMDB reviews using `all-MiniLM-L6-v2`.
2.  **Model Training**: Trains standard MLP autoencoders to compress inputs into lower-dimensional latent spaces (e.g., 32 dims).
3.  **Symbolic Discovery**: Applies `PySRRegressor` to treat the Encoder as a data-generating process, finding the simplest mathematical expressions that approximate the latent codes.
4.  **Meta-Analysis**: Aggregates results to identify which features are preserved (high importance) and which are discarded (redundant), and models the trade-off between compression ratio and reconstruction quality.

## Why Symbolic Regression?

Neural networks are often labeled as "black boxes". While we can measure their performance, we rarely understand their internal logic. Symbolic regression bridges this gap by translating complex neural weights into explicit mathematical formulas. This reveals:
*   **Interaction Effects**: How features are combined (e.g., linear sums vs. non-linear interactions).
*   **Dimensional Collapse**: Explicitly seeing which inputs map to zero/constant in the latent space.

## Insights

### 1. Latent Space Disentanglement
The analysis revealed that individual latent dimensions often specialize in capturing specific subsets of input features. For the tabular dataset, complex geological features were often compressed into compact trigonometric interactions.

### 2. Feature Importance & Redundancy
Unlike gradient-based saliency maps, symbolic regression provides a global view of feature usage.
*   **Important Features**: Distance to hydrology and elevation were frequently used in latent formulas.
*   **Redundancy**: Approximately 30% of input features were found to be effectively ignored by the autoencoder, never appearing in the symbolic equations for the latent code.

### 3. Reconstruction Law
The reconstruction error follows a predictable symbolic law based on latent dimension size and feature importance. The derived relationship suggests a diminishing return on adding latent dimensions, mathematically formalized by the symbolic model.

## Results

| Metric | Description |
|--------|-------------|
| **Compression Ratio** | 12x (384 $\to$ 32 dims) for text, ~1.7x (54 $\to$ 32 dims) for tabular |
| **Interpretability** | 100% transparent equations for each latent dimension |
| **Feature Selection** | Identified ~17 redundant tabular features |
