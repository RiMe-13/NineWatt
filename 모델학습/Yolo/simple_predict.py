import cv2
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 모델 경로 및 설정
# model_path 에는 훈련된 모델의 best.pt 경로를
# data_path 에는 data.yaml 파일을
# source_path 에는 test 데이터셋의 이미지 경로를
# label_path에는 test데이터셋의 레이블(ground truth) 경로를 지정해줍니다. 

model_path = "/home/work/choiminsung/runs/segment/model_no_roof_yolo_segmentation3/weights/best.pt"
data_path = "/home/work/choiminsung/datasets/together_no_roof/data.yaml"
source_path = "/home/work/choiminsung/datasets/together_no_roof/test/images"
label_path = "/home/work/choiminsung/datasets/together_no_roof/test/labels"  # 정답 레이블 경로


# YOLO 모델 로드
model = YOLO(model_path)

# 예측 수행
results = model.predict(
    source=source_path,  # 테스트 이미지 경로 (폴더 또는 파일)
    conf=0.6,                    # 신뢰도 임계값 (0.0 ~ 1.0)
    save=True,                    # 결과 저장
    imgsz=640,                    # 입력 이미지 크기
    device=0                      # GPU 사용 (CPU는 "cpu"로 지정)
)
