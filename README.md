# CIFAR-10 Class Imbalance Analysis (COMS4740 Project)

## Overview
This project analyzes how **controlled class imbalance** affects CNN performance on CIFAR-10.

We compare:
- Balanced vs imbalanced datasets  
- Standard vs class-weighted cross-entropy  

Focus:
- Per-class accuracy  
- Bias toward majority classes  
- Prediction confidence (optional)

---

## Objective
- Simulate class imbalance by reducing samples of selected classes  
- Measure impact on:
  - Accuracy (overall + per-class)
  - Confusion patterns  
- Evaluate if **class-weighted loss** improves minority performance  

---

## Setup

### Environment
- Python 3.x
- PyTorch
- NumPy
- Matplotlib (for visualization)
`pip install -r requirements.txt`

### Dataset
- CIFAR-10 (50k train, 10k test)  
- Keep test set **balanced**

### Imbalance Levels
- Balanced (all classes equal)  
- Mild (some classes ~50%)  
- Severe (some classes ~10–20%)  

---

## Model
- Simple CNN (fixed across all experiments)  

---

## Required Components

### 1. Imbalance Function
```python
def create_imbalance(dataset, class_ratios):
    targets = np.array(dataset.targets)
    indices = []

    for cls in range(10):
        cls_idx = np.where(targets == cls)[0]
        n_samples = int(len(cls_idx) * class_ratios.get(cls, 1.0))
        indices.extend(cls_idx[:n_samples])

    return torch.utils.data.Subset(dataset, indices)
```

### 2. Weighted Loss

```python
def get_class_weights(dataset):
    counts = np.bincount(dataset.targets)
    weights = 1.0 / counts
    return torch.FloatTensor(weights)
```

### 3. Experiments

Run:
- Balanced + standard loss
- Imbalanced + standard loss
- Imbalanced + weighted loss

### 4. Outputs

Overall accuracy
Per-class accuracy
Confusion matrix