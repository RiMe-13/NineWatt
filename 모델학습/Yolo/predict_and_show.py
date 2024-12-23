#이번에는 predict와 더불어 plt을 활용한 시각화 예입니다.

import cv2
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 모델 경로 및 설정
model_path = "/home/work/choiminsung/runs/segment/model_no_roof_yolo_segmentation3/weights/best.pt"
data_path = "/home/work/choiminsung/datasets/together_no_roof/data.yaml"
source_path = "/home/work/choiminsung/datasets/together_no_roof/test/images"
label_path = "/home/work/choiminsung/datasets/together_no_roof/test/labels"  # 정답 레이블 경로

# YOLO 모델 로드
model = YOLO(model_path)

# 정답 레이블 로드
with open(data_path, "r") as f:
    data = yaml.safe_load(f)

# 클래스별 색상 매핑
label_colors = {
    "roof": (0, 255, 255),            # 청록색
    "roofavailablearea": (0, 255, 0), # 초록색
    "heli": (0, 0, 255)              # 파란색
}

# 예측 수행
results = model.predict(
    source=source_path,
    data=data_path,
    conf=0.7,  # Confidence Threshold
    save=False  # 저장 없이 결과만 반환
)

# 예측 및 시각화
for result in results:
    try:
        # 원본 이미지 파일명 추출
        original_filename = os.path.basename(result.path)
        file_stem, _ = os.path.splitext(original_filename)

        # 원본 이미지 읽기
        original_image_path = os.path.join(source_path, original_filename)
        original_image = cv2.imread(original_image_path)

        if original_image is None:
            print(f"[WARNING] Unable to read original image: {original_filename}")
            continue

        image_height, image_width = original_image.shape[:2]

        # YOLO 예측 이미지 생성 (result.plot() 사용)
        predicted_image = result.plot()

        # 흑백 바이너리 이미지 생성
        binary_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        if result.masks is not None:
            for mask in result.masks.data:
                mask_np = mask.cpu().numpy()
                binary_mask_part = (mask_np > 0.5).astype(np.uint8) * 255
                binary_mask_resized = cv2.resize(binary_mask_part, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                binary_mask = np.maximum(binary_mask, binary_mask_resized)

        # 정답 오버레이 이미지 생성
        overlay_image = original_image.copy()
        if os.path.exists(os.path.join(label_path, f"{file_stem}.txt")):
            label_file = os.path.join(label_path, f"{file_stem}.txt")
            with open(label_file, "r") as lf:
                for line in lf:
                    # 라벨 정보 읽기
                    parts = line.strip().split()
                    class_id = int(parts[0])  # 클래스 ID
                    normalized_coords = list(map(float, parts[1:]))

                    # 클래스 이름 매핑
                    class_name = data["names"][class_id] if "names" in data else f"class_{class_id}"
                    color = label_colors.get(class_name, (255, 255, 255))  # 기본 색상은 흰색

                    # 좌표 변환
                    polygon_coords = [
                        (int(x * original_image.shape[1]), int(y * original_image.shape[0]))
                        for x, y in zip(normalized_coords[::2], normalized_coords[1::2])
                    ]
                    points = np.array(polygon_coords, dtype=np.int32)

                    # 다각형 오버레이
                    cv2.polylines(overlay_image, [points], isClosed=True, color=color, thickness=2)
                    cv2.fillPoly(overlay_image, [points], color)

        # 시각화
        plt.figure(figsize=(20, 5))

        # Plot 1: 원본 이미지
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        # Plot 2: Ground Truth Overlay
        plt.subplot(1, 4, 2)
        plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        plt.title("Ground Truth Overlay")
        plt.axis("off")

        # Plot 3: YOLO Predicted Image
        plt.subplot(1, 4, 3)
        plt.imshow(cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB))
        plt.title("YOLO Predicted Image")
        plt.axis("off")

        # Plot 4: 흑백 바이너리 이미지
        plt.subplot(1, 4, 4)
        plt.imshow(binary_mask, cmap="gray")
        plt.title("Binary Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"[ERROR] Error processing image {file_stem}: {e}")
        continue
