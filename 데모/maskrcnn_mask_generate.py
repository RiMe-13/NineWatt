import os
import cv2
import logging
import torch
import numpy as np
from PIL import Image
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog

def setup_cfg(weights_path):
    """모델 설정"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def process_single_image(input_image):
    """PIL 이미지에 대해 마스크 생성"""
    # json 파일의 categories 기반으로 클래스 정보 직접 설정
    thing_classes = ['roofavailablearea', 'heli', 'roofavailablearea']

    weights_path = "model_final.pth"  # 모델 가중치 경로
    
    # 설정 및 예측기 생성
    cfg = setup_cfg(weights_path)
    predictor = DefaultPredictor(cfg)

    # PIL 이미지를 OpenCV 형식으로 변환
    img = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    # 이미지 예측
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    
    # 클래스별 결과 저장
    class_masks = {}
    class_areas = {}

    heli_detected = False
    
    if len(instances) > 0:
        pred_masks = instances.pred_masks.numpy()
        pred_classes = instances.pred_classes.numpy()
        
        # 각 예측에 대해
        for mask, class_id in zip(pred_masks, pred_classes):
            class_name = thing_classes[class_id]

            if 'heli' in class_name:
                heli_detected = True
            # # roofavailablearea 클래스인 경우
            # if 'roofavailablearea' in class_name:
            #     area = np.sum(mask)  # 마스크의 면적 계산
            #     if class_name not in class_areas or area > class_areas[class_name]['area']:
            #         class_areas[class_name] = {
            #             'mask': mask,
            #             'area': area
            #         }
            # # 다른 클래스의 경우 기존처럼 처리
            # else:
            if class_name not in class_masks:
                class_masks[class_name] = np.zeros((h, w), dtype=np.uint8)
            class_masks[class_name] = np.logical_or(class_masks[class_name], mask)
        
        # # roofavailablearea 클래스의 최대 영역 마스크를 class_masks에 추가
        # if 'roofavailablearea' in class_areas:
        #     class_masks['roofavailablearea'] = class_areas['roofavailablearea']['mask']
    
    result = {}
    for class_name, merged_mask in class_masks.items():
        mask_img = (merged_mask * 255).astype(np.uint8)
        result[class_name] = mask_img

    # 헬리패드 검출 시 헬리패드 마스크 반환, 아닐 경우 available area 마스크 반환
    if (heli_detected):
        print(f"[INFO] Helipad detected in image")
        return Image.fromarray(result['heli'])
    
    print(f"[INFO] No helipad detected in image")
    return Image.fromarray(result['roofavailablearea'])
