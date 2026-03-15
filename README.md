Fairness-Aware Vision Transformer for Medical Diagnosis

A deep learning project that develops a fairness-aware medical imaging model using a Vision Transformer architecture.
The system predicts chest X-ray disease presence while evaluating demographic fairness across patient groups.

This project includes:

* End-to-end training pipeline
* fairness evaluation metrics
* ROC curve analysis
* hyperparameter tuning
* interactive demo UI using **Streamlit**

# Project Overview

Medical AI systems can unintentionally perform differently across demographic groups.
This project investigates how deep learning models behave across male and female patient groups and evaluates fairness in predictions.

The workflow includes:

1. Dataset preprocessing
2. Vision Transformer model training
3. Performance evaluation
4. Fairness analysis
5. Visualization of fairness metrics
6. Interactive UI for model demonstration

# Features

* Vision Transformer medical image classifier
* Fairness evaluation across demographic groups
* ROC curve visualization per group
* AUC fairness gap analysis
* Hyperparameter tuning experiments
* Interactive prediction demo using Streamlit

# Project Architecture

```
fairvit-med
│
├── app/
│   └── streamlit_app.py          # Interactive UI demo
│
├── data/
│   ├── raw/                      # Original NIH dataset
│   └── processed/                # Processed metadata
│
├── models/
│   └── baseline_model.pth        # Trained model
│
├── notebooks/
│   ├── eda.ipynb
│   ├── train_baseline.ipynb
│   ├── fairness_eval.ipynb
│   ├── model_test.ipynb
│   └── day10_fair_model_eval.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   │   └── vit_model.py          # Vision Transformer model
│   ├── fairness/
│   └── training/
│
├── figures/
│   ├── fairness_auc_bar_chart.png
│   └── fairness_roc_curves.png
│
└── README.md
```


# Dataset

This project uses the **NIH Chest X-ray Dataset**, a large-scale medical imaging dataset.

Dataset link: https://www.kaggle.com/datasets/nih-chest-xrays/data

Dataset characteristics:

 Property    Value                      
 ----------  -------------------------- 
 Images      112,120                    
 Size        ~43 GB                     
 Modalities  Chest X-rays               
 Labels      Multiple thoracic diseases 
 Metadata    Patient gender, age        

For this project:

* Images resized to 224 × 224
* Binary disease classification
* Fairness evaluation across male vs female groups



# Model Architecture

The core model is a **Vision Transformer (ViT)** designed for medical image classification.

Model components:

* Patch embedding layer
* Transformer encoder blocks
* Self-attention mechanism
* Classification head

Advantages:

* Captures global image context
* Effective for large-scale image datasets
* Strong performance on medical imaging tasks

# Training Configuration

 Parameter      Value                      
 -------------  -------------------------- 
 Image size     224 × 224                  
 Batch size     16                         
 Optimizer      AdamW                      
 Learning rate  3e-4                       
 Epochs         3–10                       
 GPU            NVIDIA RTX 4060 Laptop GPU 

Training performed using PyTorch.

---

# Fairness Evaluation

Fairness was evaluated using AUC across demographic groups.

Metrics computed:

* Overall AUC
* Male AUC
* Female AUC
* Fairness gap

Fairness gap is defined as:

```
Fairness Gap = |AUC_male − AUC_female|
```

Lower gap indicates more equitable model performance.


# Results

Example results from the evaluation:

 Group    AUC  
 -------  ---- 
 Overall  0.62 
 Male     0.63 
 Female   0.61 

Fairness Gap:

```
0.02
```

---

# Visualization

The project includes research-quality visualizations:

### AUC Comparison

Shows model performance across demographic groups.

### ROC Curves

Separate ROC curves plotted for:

* Male patients
* Female patients

These plots help identify performance imbalance.


# Streamlit Demo

The project includes an interactive demo built with **Streamlit**.

Features:

* Upload chest X-ray image
* Model prediction
* Probability score
* Fairness metrics display

Run the demo locally:

```bash
cd app
streamlit run streamlit_app.py
```

Then open:

```
http://localhost:8501
```


# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/fairvit-med.git
cd fairvit-med
```

Create environment:

```bash
pip install -r requirements.txt
```

Main dependencies:

* Python
* PyTorch
* torchvision
* pandas
* scikit-learn
* matplotlib
* seaborn
* streamlit

---

# How to Run the Project

### Train Model

Run training notebook:

```
notebooks/train_baseline.ipynb
```

### Evaluate Fairness

```
notebooks/fairness_eval.ipynb
```

### Run UI

```
streamlit run app/streamlit_app.py
```

---

# Example Workflow

```
Dataset
   ↓
Preprocessing
   ↓
Vision Transformer Training
   ↓
Model Evaluation
   ↓
Fairness Metrics
   ↓
Visualization
   ↓
Streamlit Demo
```

---

# Technologies Used

* Python
* **PyTorch**
* **Vision Transformer**
* Pandas
* NumPy
* Matplotlib
* Seaborn
* **Streamlit**

---

# Future Improvements

Potential extensions:

* Add explainability using **Gradient-weighted Class Activation Mapping**
* Expand fairness evaluation to age groups
* Multi-label disease classification
* Deploy as cloud medical AI service
* Model calibration and bias mitigation techniques

---

# Author

**Visveswaran**

Role: Machine Learning Researcher & Developer

Project focus:

* Medical AI
* Fairness in Machine Learning
* Vision Transformers
* Deep Learning for Healthcare

---

# License

This project is intended for **research and educational purposes**.
