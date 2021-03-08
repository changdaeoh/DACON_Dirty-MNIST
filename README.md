# Monthly Dacon 12 - CV learning competition (작성중)
> members : 조재훈, 오창대, 원희지 <br/>
period : 2/3 ~ 3/1 <br/>
collaborative tool : Notion, Google Drive, Google Colab, Github <br/>
framework : TF/Keras(창대, 희지), Pytorch(재훈)

<br/><br/>

## 1. Data
- raw image는 grayscale 이었으나, rgb channel에 pre-train 된 모델을 사용하기위해 rgb로 변환하여 데이터를 준비.
- scaling과 augmentation은 keras의 ImageDataGenerator 이용.
```python 
ImageDataGenerator(rescale=1./255.,
                             rotation_range = 10,
                             width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             horizontal_flip = True,
                             vertical_flip = True,
                             validation_split = 0.1)
```

<br/><br/>

## 2. Model Candidates 
인기있는 CNN 구조들을 그대로 가져와 head 부분만 바꿔서 사용. 
* VGG
* Resnets (50, 101, 152 ...)    
* InceptionResnet v2
* DenseNets
* Xception
* EfficientNets (B0, B1, ... B7)
* Vision Transformer

& Classification head (MLP) <BR/>


<br/><br/>

## 3. Training
각 후보 모델별로 약 50 ~ 100 epoch씩 탐색적인 학습 진행. <br/>
성능이 우수하였던 `InceptionResnet v2, DenseNets, EfficientNetsB4,B5` 위주로 튜닝 진행. <br/><br/>

### 3.1 Optimizer
- Adam, Nadam
- RAdam
- AdamW

optimizer로는 오직 adam의 변형들만 고려하였다.<br/>
대부분의 후보 모델에서 adam보다 nadam이 더 빨리 loss를 줄여나갔으며,<br/>
마감일이 임박했을 때, RAdam와 AdamW 등의 비교적 낯선(?) optimizer를 추가적으로 시도해 보았으나<br/>
Adam과 Nadam등과 성능비교를 수치적으로 정밀하게 해보지는 못했다.
<br/><br/>

### 3.2 LR schedule
- Cosine Annealing with Warm Restarts
- Piecewise Constant Decaying
- Performance based Decaying

코사인감쇄+재시작, 구간별 고정 감쇄, 성능기반 감쇄 등의 방법으로 epoch별 학습률을 조절.<br/>
대부분의 후보모델 학습 진행과정에서 어느 정도 epoch 이후부터는 validation loss의 감소가 saturated 되던 현상이<br/>
`Cosine Annealing with Warm Restarts` 를 적용함으로써 해소되는 듯 보였다.<br/>
> 그러나 이러한 scheduling 전략들간의 차이와 각 전략이 모델성능에 미치는 영향력을 타당하게 판단/평가하기 위해서는<br/>
**충분한 epoch의 학습**을 진행시켜봐야했을 것이다. (특히 Cosine Decay Restarts)<br/>
아쉽게도 colab의 runtime limitation 때문에 이는 이뤄지지 못했다.

<br/>

### 3.3 Loss
- Binary Cross Entropy
### 3.4 Metrics
- Binary Accuracy
### 3.5 others
- batchsize는 `efficient b5` : 16, `ViT` : 128, 나머지 : 32로 설정.
- clf head에 dropout / BN을 다양한 방식으로 시도.
- ![gpu](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/654fe40a-0ec9-49e3-a93d-e12988af1364/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210308%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210308T133233Z&X-Amz-Expires=86400&X-Amz-Signature=09fef8af14aabe861a6fc2063ebf1b4383a6b930ac994ef8dbd7f913743e9cad&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

<br/><br/>

## 4. Prediction
### 4.1 Test Time Augmentation
### 4.2 Monte Carlo Dropout
### 4.3 Ensemble
