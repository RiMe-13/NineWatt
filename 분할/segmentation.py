#bbox 이미지와 원본이미지를 활용한 segmentation 예입니다. 경로와 폴더 구조에 따라 코드를 변형하시면 됩니다.
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from segment_anything import SamPredictor, sam_model_registry
from pathlib import Path

# SAM 모델 초기화
sam_checkpoint = "sam_vit_l_0b3195.pth"
model_type = "vit_l"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 이미지 폴더 경로
base_dir = Path("imagetestfinal_unzipped/")

# 각 구 폴더 탐색
for gu_folder in base_dir.iterdir():
    print(gu_folder.name)
    if gu_folder.name != "강서구" or not gu_folder.is_dir():
        continue

    bbox_dir = gu_folder / "bbox"
    original_dir = gu_folder / "original"

    # bbox 폴더 내 파일들 탐색
    for image_file in bbox_dir.iterdir():
        if image_file.name.endswith("_bbox.jpg"):
            base_name = image_file.name.replace("_bbox.jpg", ".jpg")
            bbox_image_path = str(image_file)
            original_image_path = str(original_dir / base_name)

            # bbox 이미지 읽기 및 빨간색 박스 감지
            bbox_image = cv2.imread(bbox_image_path)
            bbox_image_rgb = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB)

            lower_red = np.array([0, 0, 200])
            upper_red = np.array([20, 20, 255])
            red_mask = cv2.inRange(bbox_image, lower_red, upper_red)

            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # 가장 큰 윤곽선 찾기 및 바운딩 박스 생성
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(bbox_image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

                plt.figure(figsize=(8, 8))
                plt.imshow(bbox_image_rgb)
                plt.title(f'Red Box Detected in: {image_file.name}')
                plt.show()

                # 원본 이미지에서 세그멘테이션 수행
                if os.path.exists(original_image_path):
                    original_image = cv2.imread(original_image_path)
                    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                    input_box = np.array([x, y, x + w, y + h])
                    predictor.set_image(original_image_rgb)

                    masks, _, _ = predictor.predict(box=input_box)

                    # 가장 큰 세그멘테이션 마스크 선택
                    largest_mask_idx = np.argmax([np.sum(mask) for mask in masks])
                    largest_mask = masks[largest_mask_idx]

                    # 마스크된 객체 추출
                    cut_out_image = np.zeros_like(original_image_rgb)
                    cut_out_image[largest_mask] = original_image_rgb[largest_mask]

                    plt.figure(figsize=(10, 10))

                    plt.subplot(1, 2, 1)
                    plt.imshow(original_image_rgb)
                    plt.title(f'Original Image: {base_name}')

                    plt.subplot(1, 2, 2)
                    plt.imshow(cut_out_image)
                    plt.title(f'Largest Segmented Object')

                    plt.show()

                else:
                    print(f"Original image not found for: {base_name}")

            else:
                print(f"No red box found in image: {image_file.name}")
