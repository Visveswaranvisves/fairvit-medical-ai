## Exploratory Data Analysis Findings

1. Class Imbalance:
The dataset contains significantly more non-disease samples than disease samples, indicating a class imbalance that may bias the model toward predicting the majority class.

2. Demographic Imbalance:
Male samples are more frequent than female samples in the dataset, suggesting potential under-representation of female patients.

3. Group-wise Label Distribution:
The number of disease-positive cases differs across demographic groups, which may lead to unequal model performance between males and females.

These observations motivate the need for fairness-aware training and evaluation to ensure equitable diagnostic performance across demographic groups.
