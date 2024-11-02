# UNI-D 4th Baseline
## Image Denoising Task Using Restormer

### 코드 구조

```
${PROJECT}
├── README.md
├── preprocess.py
├── model.py
├── train.py
└── test.py
```
- preprocess.py -> preprocess data : unzip후 데이터 경로를 알맞게 수정할 것
- test.py -> Inference : Image Denoising Inference 코드
- train.py -> Training Restormer 
- model.py -> Implementation of Restormer 
