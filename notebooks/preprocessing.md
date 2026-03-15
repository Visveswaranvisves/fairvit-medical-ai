## Image Preprocessing & Data Loading

All chest X-ray images were resized to 224×224 to match Vision Transformer input requirements. Pixel values were normalized to ensure stable training.

Data augmentation (random horizontal flipping) was applied to the training set to improve generalization. Validation data was kept deterministic.

PyTorch DataLoaders were used to batch and shuffle data efficiently while preserving associated demographic attributes, enabling fairness-aware model evaluation.
