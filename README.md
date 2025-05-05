# Description
The repository provides the `python` implementation for the algorithms in the paper _"Causal Abstraction Learning based on the Semantic Embedding Principle"_ by D'Acunto et al.

# Organization
The material is organized into two main folders:
- `src`, containing the source code, that is, the algorithms implementation and the `utils.py` file with useful functions (such as the metrics);
- `data`, containing the results saved from the `example.ipynb` notebook, stored as `parquet` files to avoid versioning problems.

Additionally, 
- `example.ipynb` shows how to apply the algorithms, in both the full-prior and partial-prior settings;
- `environment.yml` allows to install the `conda` environment. Please refer to [conda User guide](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html);
- `.gitignore` is the usual file that specifies intentionally untracked files that Git should ignore.

