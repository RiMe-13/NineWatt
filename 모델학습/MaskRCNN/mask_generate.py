import os
import cv2
import logging
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog

def setup_cfg(weights_path):
    """모델 설정"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 4000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
    cfg.MODEL.RPN.NMS_THRESH = 0.8
    cfg.MODEL.RPN.FG_IOU_THRESHOLD = 0.6
    return cfg

def process_all_images(image_dir, weights_path):
    """지정된 디렉토리의 모든 이미지에 대해 마스크 생성"""
    # json 파일의 categories 기반으로 클래스 정보 직접 설정
    thing_classes = ['roofavailablearea', 'heli', 'roof', 'roofavailablearea']
    
    # 설정 및 예측기 생성
    cfg = setup_cfg(weights_path)
    predictor = DefaultPredictor(cfg)

    # 출력 디렉토리 생성
    output_dir = os.path.join(os.getcwd(), "mask")
    os.makedirs(output_dir, exist_ok=True)
    
    # 클래스별 디렉토리 생성 (중복 제거)
    unique_classes = list(set(thing_classes))
    for class_name in unique_classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    # 지원하는 이미지 확장자
    valid_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']

    # 최상위 디렉토리의 이미지만 처리
    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            img_path = os.path.join(image_dir, file)
            img_name = os.path.splitext(file)[0]
            
            logger.info(f"Processing image: {img_path}")
            
            # 이미지 로드 및 예측
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not load image: {img_path}")
                continue
            
            h, w = img.shape[:2]
            outputs = predictor(img)
            instances = outputs["instances"].to("cpu")
            
            if len(instances) > 0:
                pred_masks = instances.pred_masks.numpy()
                pred_classes = instances.pred_classes.numpy()
                
                # 클래스별로 마스크 및 영역 정보 저장
                class_masks = {}
                class_areas = {}
                
                # 각 예측에 대해
                for mask, class_id in zip(pred_masks, pred_classes):
                    class_name = thing_classes[class_id]
                    
                    # roofavailablearea 클래스인 경우
                    if 'roofavailablearea' in class_name:
                        area = np.sum(mask)  # 마스크의 면적 계산
                        if class_name not in class_areas or area > class_areas[class_name]['area']:
                            class_areas[class_name] = {
                                'mask': mask,
                                'area': area
                            }
                    # 다른 클래스의 경우 기존처럼 처리
                    else:
                        if class_name not in class_masks:
                            class_masks[class_name] = np.zeros((h, w), dtype=np.uint8)
                        class_masks[class_name] = np.logical_or(class_masks[class_name], mask)
                
                # roofavailablearea 클래스의 최대 영역 마스크를 class_masks에 추가
                if 'roofavailablearea' in class_areas:
                    class_masks['roofavailablearea'] = class_areas['roofavailablearea']['mask']
                
                # 각 클래스별 마스크 저장
                for class_name, merged_mask in class_masks.items():
                    # 마스크를 이미지로 변환 (0 또는 255)
                    mask_img = (merged_mask * 255).astype(np.uint8)
                    
                    # 마스크 파일 저장
                    mask_path = os.path.join(output_dir, class_name, f"{img_name}.png")
                    cv2.imwrite(mask_path, mask_img)
                    
                    # 컬러 오버레이 마스크 저장
                    color_mask = np.zeros_like(img)
                    color_mask[merged_mask] = [0, 255, 0]
                    masked_img = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
                    color_mask_path = os.path.join(output_dir, class_name, f"{img_name}_overlay.jpg")
                    cv2.imwrite(color_mask_path, masked_img)
                    
                    if 'roofavailablearea' in class_name:
                        logger.info(f"Saved largest roofavailablearea mask for {img_name} with area: {class_areas['roofavailablearea']['area']}")
                
            logger.info(f"Processed {img_name}")

if __name__ == "__main__":
    # 경로 설정
    image_dir = "/home/work/kimjongjip/solar"  # 이미지가 있는 디렉토리
    weights_path = "/home/work/kimjongjip/output/solar_detection_20241124_064315/model_stage_1_best.pth"  # 모델 가중치 경로
    
    # 마스크 생성 실행
    process_all_images(image_dir, weights_path)


