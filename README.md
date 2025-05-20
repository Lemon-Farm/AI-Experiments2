# 📚 딥러닝 실험 프로젝트

여러 딥러닝 실험을 통해 이론적 배경을 실제 환경에서 재현하고, 이들이 모델 성능에 미치는 영향을 분석하였습니다.  
※ 모든 실험은 고정된 random seed 및 3회 반복 수행 후 평균값을 사용했습니다.

## 📑 목차

1. [AlexNet_Augmentation](#🟢-alexnet_augmentation)  
2. [설치 및 실행 방법](#⚙️-설치-및-실행-방법)  

---

## 🟢 AlexNet_Augmentation

### 🎯 목표

* 다양한 Data Augmentation 기법을 적용했을 때 AlexNet 모델의 CIFAR-10 성능 변화를 확인  

### 🧠 배경

Data Augmentation은 학습 데이터의 다양성을 증가시켜 모델의 일반화 성능을 높이는 대표적 기법입니다.  
본 실험에서는 RandomCrop, RandomHorizontalFlip, Cutout(RandomErasing) 등의 기법을 단독 및 조합하여 적용했을 때의 성능 향상 효과를 AlexNet 구조에서 직접 검증합니다.

### 🛠 구현 내용

* **코드**: `AlexNet_Augmentation.ipynb`  
* **프레임워크**: PyTorch  
* **데이터셋**: CIFAR-10 (학습 20000장 per Class / 테스트 10,000장)  
* **모델**: 표준 AlexNet 구현  
* **Augmentation 기법**:  
  1. None: ToTensor → Normalize(mean, std)  
  2. RandomCrop(padding=4) → ToTensor → Normalize  
  3. RandomHorizontalFlip(p=0.5) → ToTensor → Normalize  
  4. Cutout: ToTensor → Normalize → RandomErasing(p=0.3, scale=(0.02,0.25), ratio=(1.0,1.0), value=mean)  
  5. All: RandomCrop + RandomHorizontalFlip + ToTensor + Normalize + RandomErasing  
* **학습 설정**:  
  - Optimizer: SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)  
  - Scheduler: MultiStepLR(milestones=[20, 40], gamma=0.2)  
  - Batch size: 512  
  - Epochs: 50  
  - Worker init 함수: `worker_init_fn` 사용 (seed 고정)  
  - 반복 실험: 각 설정당 3회  

### 📊 결과 요약

| Augmentation             | 평균 정확도 (%) | None 대비 차이 (%) |
| ------------------------ | --------------: | -----------------: |
| None                     |          59.19  |              0.00  |
| RandomCrop               |          64.25  |              5.06  |
| RandomHorizontalFlip     |          63.39  |              4.20  |
| Cutout                   |          61.30  |              2.11  |
| All (조합)               |          66.96  |              7.77  |

> **관찰**:  
> - 단일 기법 중 RandomCrop이 약 5.06%로 가장 큰 성능 향상을 보였고, Cutout은 비교적 작은 효과(2.11%)를 보임.  
> - 모든 기법을 조합한 ‘All’ 적용 시 약 7.77%의 최대 성능 향상을 확인.
> - Augmentation 기법을 적용한 학습이 일관적으로 성능 향상을 보임을 확인.

---

## ⚙️ 설치 및 실행 방법

```bash
git clone https://github.com/Lemon-Farm/AI-Experiments2.git
cd AI-Experiments2
```

각 실험 노트북(`.ipynb`)을 열어 실행하면 결과를 재현할 수 있습니다.

---
