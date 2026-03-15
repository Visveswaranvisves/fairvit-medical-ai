## Data Splitting & Pipeline Design

We performed a stratified train-validation split using a combined attribute of disease label and demographic group (sex) to ensure fair representation of all groups in both training and evaluation sets.

A custom PyTorch Dataset class was implemented to load chest X-ray images along with labels and demographic attributes. This design enables downstream fairness-aware training and group-wise evaluation.

This pipeline ensures reproducibility, prevents data leakage, and preserves demographic equity across experimental stages.
