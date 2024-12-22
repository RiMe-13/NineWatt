import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

def process_image_yolo(input_image):
    """PIL 이미지에 대해 마스크 생성"""
    # YOLO 모델 설정
    model_path = "best.pt"
    model = YOLO(model_path)

    # 이미지를 OpenCV 형식으로 변환
    img = np.array(input_image)

    # 예측 수행
    results = model.predict(
        source=img,
        conf=0.7,  # Confidence Threshold
        save=False  # 저장 없이 결과만 반환
    )

    # 클래스 이름 (YOLO 데이터셋에서 클래스 ID와 매핑)
    class_names = model.names  # {0: 'roof', 1: 'roofavailablearea', 2: 'heli'}

    # 예측 및 헬리패드 확인
    for result in results:
        try:
            image_height, image_width = img.shape[:2]

            heli_detected = False

            # 흑백 바이너리 이미지 생성
            binary_masks_per_class = {class_id: np.zeros((image_width, image_height), dtype=np.uint8) for class_id in class_names.keys()}

            if result.masks is not None:
                for mask, cls in zip(result.masks.data, result.boxes.cls):
                    mask_np = mask.cpu().numpy()
                    binary_mask_part = (mask_np > 0.5).astype(np.uint8) * 255
                    binary_mask_resized = cv2.resize(binary_mask_part, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    
                    class_id = int(cls)
                    if class_names[class_id] == "heli":
                        heli_detected = True
                    binary_masks_per_class[class_id] = np.maximum(binary_masks_per_class[class_id], binary_mask_resized)

            # 헬리패드 검출 결과 반환
            if (heli_detected):
                print(f"[INFO] Helipad detected in image")
                return Image.fromarray(binary_masks_per_class[0]), heli_detected
            
            print(f"[INFO] No helipad detected in image")
            return Image.fromarray(binary_masks_per_class[1]), heli_detected
        
        except Exception as e:
            print(f"[ERROR] Error processing image {result.path}: {e}")
            continue
