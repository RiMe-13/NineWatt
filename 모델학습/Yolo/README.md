# YOLO 설명
-----------
분할된 옥상 이미지에서 여유공간을 탐지하기 위해 사용하는 모델입니다. 

현 폴더에서는 데모 버전과는 별도로 초기 환경 설정에서부터 훈련 및 테스트와 관련하여 설명합니다.

훈련 데이터셋은 미리 구축되어 있다고 가정합니다. 

https://universe.roboflow.com/rooftop1/-together

위 링크는 직접 구축한 데이터셋 예입니다. 품질이 더 좋은 분할 이미지를 갖고 roboflow에서 레이블링 하면 쉽게 yolov8 데이터셋을 구축하실 수 있습니다.

# YOLO 환경설정 및 간단한 훈련, 테스트 예
-----------
## 환경설정

+ 압축 해제

원본 데이터셋을 압축해제하는 코드입니다. 

```
import zipfile
import os

# 압축 파일 경로와 압축 해제 경로 설정
zip_file_path = "together_no_roof.zip"
extract_path = "./datasets/together_no_roof"  # 압축 해제 폴더 이름, datasets 폴더 생성 후 해당 데이터셋 압축 해제 경로 설정

# 압축 해제
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("압축 해제 완료:", extract_path)

```
+ 데이터셋 준비

데이터셋 (together_no_roof.zip 예시)을 압축 해제하고 난뒤 datasets 폴더로 이동시켜 줍니다.

데이터셋의 폴더 구조는 아래와 같이 images/ 폴더와 labels/ 폴더 그리고 dataset.yaml파일로 이루어져 있어야 합니다. 

```
datasets/
└── together_no_roof/
    ├── images/
    │   ├── train/       # 학습 이미지
    │   ├── val/         # 검증 이미지
    │   └── test/        # 테스트 이미지
    ├── labels/
    │   ├── train/       # 학습 라벨
    │   ├── val/         # 검증 라벨
    │   └── test/        # 테스트 라벨
    └── dataset.yaml      # 데이터셋 설정 파일

```
+ .yaml 파일 예시

    .yaml 파일에는 클래스 종류와 개수, 훈련/검증/테스트 셋 경로가 포함되어있어야 합니다. 예시는 아래 이미지와 같습니다. 


![image](https://github.com/user-attachments/assets/fa3d2426-7334-4244-a93a-2d901b8f5b07)


+ YOLO 설치

  pip install 이후 yolov8n-seg.pt 가중치 파일 사용 가능합니다.

  
```
pip install ultralytics
```

YOLO("yolov8n-seg.pt") 메서드를 통해 모델을 로드합니다.

```
from ultralytics import YOLO

# YOLOv8 모델 로드 및 학습 코드 예시
model = YOLO("yolov8n-seg.pt")
```

+ GPU 사용 점검

GPU 사용이 가능한지 점검합니다. GPU 환경이 설정되어야 원할한 훈련이 됩니다.

```
import torch
print(torch.cuda.is_available())  # True가 나와야 GPU 사용 가능
```
GPU가 있는데도 False가 나온다면 아래 pip install을 통해 torch 버전을 업그레이드 합니다.

```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install --upgrade --force-reinstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

```
----------
## 모델 훈련
+ 훈련 코드

모델 훈련 코드입니다. 앞서 로드한 model에 .train을 통해서 훈련을 진행합니다. 

```
# 모델 학습
# 'data' 파라미터에 커스텀 데이터셋의 경로가 포함된 YAML 파일을 지정
model.train(
    data="/home/work/ninewatt/datasets/together_no_roof/data.yaml",  # dataset.yaml 파일 경로
    epochs=1000,                     # 훈련 에포크 수
    patience=300,                    # patience 란 오버피팅을 방지하기 위해 validation loss가 더이상 감소하지 않을 경우 학습을 조기 종료 시키는 옵션입니다. YOLO에는 기본적으로 patience가 설정되어 있습니다.
                                      # 현재는 300으로 설정해둬, 300 에폭 동안 validation loss가 더이상 감소하지 않을 경우 학습을 조기 종료시킵니다.
    imgsz=640,                     # 이미지 크기 (예: 640x640)
    batch=16,                      # 배치 크기입니다. 
    name="model_no_roof_yolo_segmentation", # 모델 저장 이름입니다.
    device=0,  # GPU를 사용하도록 설정합니다. 꼭 cuda.available()을 확인해야합니다.
    lr0=0.002, # 초기 러닝레이트 입니다.
    lrf=0.01,  # 최종 러닝레이트 입니다. 내부에서 러닝 레이트를 점점 증가시키며 최적화를 진행합니다.
    save_period=100  # 100 에폭마다 체크포인트를 저장합니다. 체크 포인트는 학습이 예상치 못하게 종료되었을때 다시 학습을 진행시킬 수 있는 저장구간입니다.
)
```

만약 훈련 도중 일정 에폭구간 동안 validation loss가 향상된 점이 없다면 early stopping을 통해 모델이 훈련을 조기 종료합니다. 

patience 를 설정하지 않아도 yolo에는 기본적으로 10 에폭으로 설정되어있습니다. 

![image](https://github.com/user-attachments/assets/37519db3-f345-4ec2-9765-2f46af39f579)

훈련이 완료되면 runs/ 폴더가 생성되며, runs/segment/내가지정한 모델명 폴더에 아래와 같은 가중치 파일과 훈련 결과 파일들이 저장됩니다.

weights 폴더에는 훈련완료된 가중치 파일들이 있습니다. 이중 best.pt파일이 가장 잘 훈련된 가중치 파일입니다.

confusion_matrix, Box_curve, results, Mask_curve 파일들을 통해 훈련 결과 성능을 볼 수 있습니다. 이 파일들은 validation 과 train 관련 수치들입니다.

![image](https://github.com/user-attachments/assets/cff6c53b-4108-4a20-ab4d-8852139af38c)



---------------
## 테스트 코드

    훈련이 완료되었다면 test 데이터셋을 통해 모델의 성능을 더 객관적으로 평가할 수 있습니다. 

    아래는 테스트 할 수 있는 YOLO 명령어입니다. mode=val 고정이며, **split=test** 만 지정해주면 됩니다. model= 에는 훈련결과 나온 .pt 가중치 파일 경로,  data=에는 .yaml 파일 경로가 필요합니다.


```
!yolo task=segment mode=val model=/home/work/ninewatt/runs/segment/학습결과저장된폴더/weights/best.pt data=/home/work/ninewatt/datasets/together_no_roof/data.yaml split=test
```

![image](https://github.com/user-attachments/assets/cc4532d0-e40f-492c-b918-783949841988)

  훈련완료시 위와 같은 수치들과 함께 runs/segment/val 폴더에 테스트 결과가 저장됩니다. mode=val 고정이어서 val로 폴더명이 잡히는 것이니, 테스트 데이터 결과로 보시면 됩니다.


