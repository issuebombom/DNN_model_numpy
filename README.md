# Numpy를 활용한 DNN 구현

## Introduction
Deep Neural Network의 기본 구조와 작동 방식에 대해 살펴보면서 이해한 과정을 코드로 구현한 레포지토리 입니다. 데이터 로드와 시각화를 제외한 모델 구현에 있어 numpy 라이브러리만 사용하였습니다.


## Implement
Task: Classification  
Preprocess: MNIST Datasets  
Architecture: 4 Fully-Connected Layers ( 784, 128, 64, 10 )  
Activation Function: ReLU, Leaky ReLU, Sigmoid, Softmax  
Loss Function: MSE, Categorical Cross Entropy  
Evaluation Metric: Accuracy, Precision, Recall  
Optimizer: Adam, SGD(Mini-batch)

## Requirements
python 3.8.16  
numpy, OmegaConf, matplotlib, tqdm  
**reqirements.txt 참조


