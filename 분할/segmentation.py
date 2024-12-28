import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from segment_anything import SamPredictor, sam_model_registry

# SAM 모델 초기화
sam_checkpoint = "sam_vit_l_0b3195.pth"
model_type = "vit_l"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 이미지가 저장된 경로 (하위 폴더 없이, 바로 여기에 *_bbox.jpg와 .jpg가 존재)
base_dir = "sample_folder"

# base_dir 내 파일들을 순회
for image_file in os.listdir(base_dir):
    # _bbox.jpg로 끝나는 파일만 확인
    if image_file.endswith("_bbox.jpg"):
        # 원본 이미지 파일명 생성
        base_name = image_file.replace("_bbox.jpg", ".jpg")

        bbox_image_path = os.path.join(base_dir, image_file)
        original_image_path = os.path.join(base_dir, base_name)

        # bbox 이미지 읽기 및 빨간색 박스 감지
        bbox_image = cv2.imread(bbox_image_path)
        # 만약 bbox_image가 None이면 이미지 경로가 잘못되었거나 파일이 없다는 뜻이므로 체크
        if bbox_image is None:
            print(f"Cannot read bbox image: {bbox_image_path}")
            continue

        bbox_image_rgb = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB)

        # 빨간색 영역 추출 (단순 Threshold 예시)
        lower_red = np.array([0, 0, 200])
        upper_red = np.array([20, 20, 255])
        red_mask = cv2.inRange(bbox_image, lower_red, upper_red)

        # 컨투어 찾기
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # 가장 큰 윤곽선 찾기 및 바운딩 박스 생성
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # 디버깅/확인용 박스 시각화
            bbox_copy = bbox_image_rgb.copy()
            cv2.rectangle(bbox_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

            plt.figure(figsize=(8, 8))
            plt.imshow(bbox_copy)
            plt.title(f'Red Box Detected in: {image_file}')
            plt.show()

            # 원본 이미지에서 세그멘테이션 수행
            if os.path.exists(original_image_path):
                original_image = cv2.imread(original_image_path)
                if original_image is None:
                    print(f"Cannot read original image: {original_image_path}")
                    continue

                original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                # SAM에 사용할 바운딩 박스
                input_box = np.array([x, y, x + w, y + h])

                # 이미지 설정 후 예측
                predictor.set_image(original_image_rgb)
                masks, scores, logits = predictor.predict(box=input_box)

                # 여러 마스크 중 가장 큰 마스크 선택
                largest_mask_idx = np.argmax([np.sum(mask) for mask in masks])
                largest_mask = masks[largest_mask_idx]

                # 마스크 영역만 오려내기
                cut_out_image = np.zeros_like(original_image_rgb)
                cut_out_image[largest_mask] = original_image_rgb[largest_mask]

                # 결과 시각화
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
            print(f"No red box found in image: {image_file}")
