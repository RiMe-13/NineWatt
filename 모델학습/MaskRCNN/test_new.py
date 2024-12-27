import os
import cv2
import random
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import torch
import logging

# 로깅 설정
logger = logging.getLogger("register_dataset")
logging.basicConfig(level=logging.INFO)

def register_dataset(name, json_file, image_root):
    """COCO 형식의 데이터셋을 등록하고 category_id를 재매핑하며, 세그멘테이션 유효성을 검사합니다."""
    from detectron2.structures import BoxMode

    with open(json_file, 'r') as f:
        dataset = json.load(f)

    # 데이터셋의 카테고리 확인
    categories = {cat['id']: cat['name'] for cat in dataset['categories']}
    sorted_category_ids = sorted(categories.keys())
    category_id_to_contiguous_id = {k: i for i, k in enumerate(sorted_category_ids)}
    thing_classes = [categories[k] for k in sorted_category_ids]

    def get_dicts():
        dataset_dicts = []
        for img in dataset['images']:
            record = {}
            record["file_name"] = os.path.join(image_root, img['file_name'])
            record["image_id"] = img['id']
            record["height"] = img['height']
            record["width"] = img['width']
            
            annos = [anno for anno in dataset['annotations'] if anno['image_id'] == img['id']]
            objs = []
            for anno in annos:
                # category_id 재매핑
                if anno['category_id'] not in category_id_to_contiguous_id:
                    logger.warning(f"Skipping annotation ID {anno['id']} due to invalid category_id.")
                    continue
                mapped_category_id = category_id_to_contiguous_id[anno['category_id']]
                
                # segmentation 유효성 검사
                segmentation = anno.get('segmentation', [])
                if not segmentation:
                    logger.warning(f"Skipping annotation ID {anno['id']} due to empty segmentation.")
                    continue
                
                # 폴리곤 형식의 유효성 검사
                if isinstance(segmentation, list):
                    valid_segmentation = []
                    for seg in segmentation:
                        if len(seg) >= 6:  # 최소 3개의 점 (x, y) * 3
                            valid_segmentation.append(seg)
                        else:
                            logger.warning(f"Skipping segmentation in annotation ID {anno['id']} due to insufficient points.")
                    if not valid_segmentation:
                        logger.warning(f"Skipping annotation ID {anno['id']} due to no valid segmentation.")
                        continue
                elif isinstance(segmentation, dict):
                    if not all(k in segmentation for k in ('counts', 'size')):
                        logger.warning(f"Skipping annotation ID {anno['id']} due to invalid RLE segmentation.")
                        continue
                    valid_segmentation = segmentation
                else:
                    logger.warning(f"Skipping annotation ID {anno['id']} due to unknown segmentation format.")
                    continue
                
                obj = {
                    "bbox": anno['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": mapped_category_id,
                    "segmentation": valid_segmentation
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts

    DatasetCatalog.register(name, get_dicts)
    MetadataCatalog.get(name).set(thing_classes=thing_classes)

def setup_cfg(weights_path, num_classes):
    """모델 설정"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 점수 임계값 더욱 낮춤
    cfg.MODEL.RPN.NMS_THRESH = 0.2
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def compute_confusion_matrix(predictor, dataset_dicts, metadata, iou_threshold=0.5):
    """컨퓨전 매트릭스를 계산합니다."""
    num_classes = len(metadata.thing_classes)
    y_true, y_pred = [], []

    for data in dataset_dicts:
        img = cv2.imread(data["file_name"])
        if img is None:
            logger.warning(f"Image {data['file_name']} could not be loaded.")
            continue
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes.numpy()
        pred_masks = instances.pred_masks.numpy()

        if len(pred_classes) == 0:
            logger.info(f"No predictions for image {data['file_name']}.")

        gt_classes = [anno['category_id'] for anno in data['annotations']]
        gt_masks = []

        # Ground Truth Mask 생성
        for anno in data['annotations']:
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            segmentation = anno["segmentation"]
            if isinstance(segmentation, list):
                for seg in segmentation:
                    pts = np.array(seg).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
            elif isinstance(segmentation, dict):
                # RLE segmentation인 경우 RLE를 사용하여 마스크 생성 (추가 구현 필요)
                # 현재는 단순히 건너뜀
                logger.warning(f"RLE segmentation detected for annotation ID {anno['category_id']}. Skipping mask generation.")
                continue
            else:
                logger.warning(f"Unknown segmentation format for annotation ID {anno['category_id']}. Skipping mask generation.")
                continue
            gt_masks.append(mask)
        gt_masks = np.array(gt_masks)

        # 매칭 상태 추적
        gt_matched = [False] * len(gt_classes)

        # 예측 결과와 Ground Truth 매칭
        for pred_class, pred_mask in zip(pred_classes, pred_masks):
            best_iou, best_gt_idx = 0, -1
            for idx, (gt_class, gt_mask) in enumerate(zip(gt_classes, gt_masks)):
                if gt_matched[idx]:
                    continue
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                if union == 0:
                    continue
                iou = intersection / union
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, idx

            if best_iou >= iou_threshold and best_gt_idx != -1:
                y_true.append(gt_classes[best_gt_idx])
                y_pred.append(pred_class)
                gt_matched[best_gt_idx] = True
            else:
                # 매칭 실패 -> False Positive
                y_true.append(num_classes)  # Background로 처리
                y_pred.append(pred_class)

        # 매칭되지 않은 Ground Truth는 False Negative로 처리
        for gt_idx, matched in enumerate(gt_matched):
            if not matched:
                y_true.append(gt_classes[gt_idx])  # 실제 클래스
                y_pred.append(num_classes)  # Background로 처리

    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)) + [num_classes])

    # 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d',
                xticklabels=metadata.thing_classes + ['Background'],
                yticklabels=metadata.thing_classes + ['Background'],
                cmap='Blues')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.show()
    return conf_matrix

def visualize_test_results(dataset_name, predictor, num_images=5):
    """테스트 데이터셋의 Ground Truth와 예측 결과를 시각화합니다."""
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    for d in random.sample(dataset_dicts, min(num_images, len(dataset_dicts))):
        img = cv2.imread(d["file_name"])
        if img is None:
            logger.warning(f"Image {d['file_name']} could not be loaded.")
            continue

        # Ground Truth 시각화
        visualizer_gt = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        vis_gt = visualizer_gt.draw_dataset_dict(d)

        # 예측 결과
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        # 예측 결과 확인
        if len(instances) == 0:
            logger.info(f"No instances predicted for image {d['file_name']}.")
        else:
            logger.info(f"Predicted {len(instances)} instances for image {d['file_name']}.")

        # 시각화
        visualizer_pred = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        vis_pred = visualizer_pred.draw_instance_predictions(instances)

        # 두 이미지를 나란히 표시
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(vis_gt.get_image())
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(vis_pred.get_image())
        plt.title("Predictions")
        plt.axis('off')

        plt.show()

def run_test(base_dir, weights_path, num_images=5):
    """모델을 테스트 데이터셋으로 평가하고 결과를 시각화합니다."""
    # train 데이터셋 등록하여 원본 클래스 정보 가져오기
    train_json = os.path.join(base_dir, 'train', '_annotations.coco.json')
    train_root = os.path.join(base_dir, 'train')
    if 'solar_train' in DatasetCatalog:
        DatasetCatalog.remove('solar_train')
        MetadataCatalog.remove('solar_train')
    register_dataset('solar_train', train_json, train_root)

    # 테스트 데이터셋 등록
    test_json_file = os.path.join(base_dir, 'test', '_annotations.coco.json')
    test_image_root = os.path.join(base_dir, 'test')
    dataset_name = 'test'
    if dataset_name in DatasetCatalog:
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)
    register_dataset(dataset_name, test_json_file, test_image_root)

    # 학습 시의 클래스 수 사용
    num_classes = len(MetadataCatalog.get('solar_train').thing_classes)
    cfg = setup_cfg(weights_path, num_classes)
    predictor = DefaultPredictor(cfg)

    # 데이터셋 확인
    dataset_dicts = DatasetCatalog.get(dataset_name)
    print("Dataset image IDs:", [d["image_id"] for d in dataset_dicts])

    # 테스트 데이터셋 시각화 (Ground Truth와 예측 결과 함께)
    print("Testing visualization on sample images...")
    visualize_test_results(dataset_name, predictor, num_images=num_images)

    # 평가 수행 (tasks 인자 제거)
    print("Evaluating the model...")
    evaluator = COCOEvaluator(
        dataset_name, 
        cfg, 
        False, 
        output_dir="./output"
        # tasks=["segm"]  # 이 부분 제거
    )
    test_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(predictor.model, test_loader, evaluator)

    # COCOEvaluator의 _coco_eval 속성이 없을 수 있으므로 안전하게 처리
    coco_eval = getattr(evaluator, "_coco_eval", None)
    if coco_eval and 'bbox' in coco_eval:
        print("Precisions shape:", coco_eval['bbox'].eval['precision'].shape)
        print("Class names length:", len(MetadataCatalog.get(dataset_name).thing_classes))

    print("Evaluation Results:", results)

    # 혼동 행렬 계산
    print("Computing confusion matrix...")
    conf_matrix = compute_confusion_matrix(predictor, dataset_dicts, MetadataCatalog.get(dataset_name))

    return results, conf_matrix

if __name__ == "__main__":
    # 로그 설정
    from detectron2.utils.logger import setup_logger
    setup_logger()

    # 경로 설정
    base_dir = "/image_dataset"  # 데이터셋 경로
    weights_path = "/your_path/your_model.pth"  # 모델 가중치 경로

    # 테스트 실행
    results, conf_matrix = run_test(base_dir, weights_path, num_images=77)


