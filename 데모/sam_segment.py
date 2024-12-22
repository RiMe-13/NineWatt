import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import matplotlib.pyplot as plt

# SAM 모델 초기화
sam_checkpoint = "sam_vit_l_0b3195.pth"  # SAM 모델 체크포인트 경로 (업로드 필요)
model_type = "vit_l"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# SAM 분할 함수
def segment_image(original_image, bbox_image):
    # PIL 이미지를 NumPy 배열로 변환
    original_np = np.array(original_image)
    bbox_np = np.array(bbox_image)

    # bbox 이미지에서 빨간 박스 감지
    lower_red = np.array([200, 0, 0])
    upper_red = np.array([255, 50, 50])
    red_mask = cv2.inRange(bbox_np, lower_red, upper_red)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None  # 빨간 박스가 없는 경우

    # 가장 큰 윤곽선 선택 및 바운딩 박스 계산
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    input_box = np.array([x, y, x + w, y + h])

    # SAM을 사용하여 마스크 예측
    predictor.set_image(original_np)
    masks, _, _ = predictor.predict(box=input_box)

    # 가장 큰 마스크 선택
    largest_mask_idx = np.argmax([np.sum(mask) for mask in masks])
    largest_mask = masks[largest_mask_idx]

    # 마스크 적용된 이미지 생성
    segmented_image = np.zeros_like(original_np)
    segmented_image[largest_mask] = original_np[largest_mask]

    # NumPy 배열을 PIL 이미지로 변환
    return Image.fromarray(segmented_image)
