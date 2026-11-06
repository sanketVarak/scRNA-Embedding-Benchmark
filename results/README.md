# Results and figures

## Figures already included

`figures/` contains PNG visual outputs extracted from the saved output cells of the supplied original Kaggle notebook. They are preserved as standalone assets and shown in the root README. The images include PBMC3k Harmony null-control plots, pancreas UMAP comparisons, the main metric bar chart, and Geneformer token-cap sensitivity UMAPs.

The two PBMC3k null-control plots are both retained because the saved Leiden panels show different cluster counts (7 and 6). They represent distinct saved outputs, not a comparison generated during repository packaging.

## Regenerating results

Run `notebooks/scrna_embedding_benchmark.ipynb` on Kaggle with Internet enabled and a T4 GPU to regenerate the benchmark outputs. Copy regenerated CSV files into this directory when publishing a new run. Re-running may produce different UMAP layouts or cluster labels depending on settings and runtime details.

Do not commit raw datasets, model weights, credentials, private cache files, or large intermediate embedding arrays. See the root `.gitignore`.
