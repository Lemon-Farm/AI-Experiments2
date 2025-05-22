# 📚 Deep Learning Experiment Project

Through various deep learning experiments, the theoretical foundations were reproduced in practice and their effects on model performance were analyzed.  
※ All experiments were run with a fixed random seed, repeated 3 times, and averaged.

## 📑 Table of Contents

1. [AlexNet_Augmentation](#🟢-alexnet_augmentation)  
2. [Installation and Execution](#⚙️-installation-and-execution)  

---

## 🟢 AlexNet_Augmentation

### 🎯 Objective

* Evaluate how different data augmentation techniques affect AlexNet’s performance on CIFAR-10.

### 🧠 Background

Data augmentation increases the diversity of training data to improve model generalization.  
In this experiment, we directly verify the effect of techniques such as RandomCrop, RandomHorizontalFlip, and Cutout (RandomErasing) alone and in combination on AlexNet.

### 🛠 Implementation Details

* **Code**: `AlexNet_Augmentation.ipynb`  
* **Framework**: PyTorch  
* **Dataset**: CIFAR-10 (20,000 training images per class / 10,000 test images)  
* **Model**: Standard AlexNet implementation  
* **Augmentation techniques**:  
  1. **None**: ToTensor → Normalize(mean, std)  
  2. **RandomCrop**: RandomCrop(padding=4) → ToTensor → Normalize  
  3. **RandomHorizontalFlip**: RandomHorizontalFlip(p=0.5) → ToTensor → Normalize  
  4. **Cutout**: ToTensor → Normalize → RandomErasing(p=0.3, scale=(0.02, 0.25), ratio=(1.0, 1.0), value=mean)  
  5. **All**: RandomCrop + RandomHorizontalFlip + ToTensor + Normalize + RandomErasing  
* **Training setup**:  
  - Optimizer: SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)  
  - Scheduler: MultiStepLR(milestones=[20, 40], gamma=0.2)  
  - Batch size: 512  
  - Epochs: 50  
  - Worker init function: `worker_init_fn` for seed control  
  - Repetitions: 3 runs per setting, averaged  

### 📊 Results Summary

| Augmentation         | Mean Accuracy (%) | Difference vs. None (%) |
| -------------------- | -----------------: | ----------------------: |
| None                 |             59.19  |                  0.00   |
| RandomCrop           |             64.25  |                  5.06   |
| RandomHorizontalFlip |             63.39  |                  4.20   |
| Cutout               |             61.30  |                  2.11   |
| All (combined)       |             66.96  |                  7.77   |

> **Observation**:  
> - Among single techniques, RandomCrop yielded the largest gain (~5.06%), while Cutout had a modest effect (2.11%).  
> - The combined ‘All’ setting achieved the highest improvement (~7.77%).  
> - Applying augmentation consistently boosted performance.

---

## ⚙️ Installation and Execution

```bash
git clone https://github.com/Lemon-Farm/AI-Experiments2.git
cd AI-Experiments2
```

Open and run the corresponding notebook (.ipynb) to reproduce the results.

---
