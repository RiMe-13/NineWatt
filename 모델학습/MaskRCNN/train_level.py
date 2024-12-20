# 필요한 라이브러리 임포트
import os
import torch
import json
from datetime import datetime
import random
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, HookBase, DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
import matplotlib.pyplot as plt
from detectron2.structures import BoxMode
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset
import numpy as np

def clear_gpu_cache():
   """GPU 캐시를 정리합니다."""
   torch.cuda.empty_cache()
   if torch.cuda.is_available():
       torch.cuda.synchronize()

# EarlyStoppingHook 클래스 정의
class EarlyStoppingHook(HookBase):
   def __init__(self, patience, metric_name="bbox/AP", minimize=False):
       self.patience = patience
       self.metric_name = metric_name
       self.minimize = minimize
       self.best_metric = None
       self.num_bad_epochs = 0

   def after_eval(self):
       metrics = self.trainer.storage.latest().get('eval')
       if metrics is None:
           return

       current_metric = metrics.get(self.metric_name)
       if current_metric is None:
           return

       if self.best_metric is None:
           self.best_metric = current_metric
           self.num_bad_epochs = 0
       else:
           if self.minimize:
               is_better = current_metric < self.best_metric
           else:
               is_better = current_metric > self.best_metric

           if is_better:
               self.best_metric = current_metric
               self.num_bad_epochs = 0
           else:
               self.num_bad_epochs += 1

       if self.num_bad_epochs >= self.patience:
           print(f"얼리 스토핑: {self.patience}번의 평가 동안 '{self.metric_name}'가 개선되지 않았습니다.")
           self.trainer.storage.put_scalar("early_stop", True)
           self.trainer.iter = self.trainer.max_iter

# StageTrainer 클래스 정의
class StageTrainer(DefaultTrainer):
   def __init__(self, cfg, stage=0):
       clear_gpu_cache()  # GPU 캐시 정리
       super().__init__(cfg)
       self.stage = stage
       self.setup_stage(stage)
   
   def setup_stage(self, stage):
       # Stage 0: res5만 학습
       if stage == 0:
           for name, param in self.model.backbone.named_parameters():
               if 'res5' in name:
                   param.requires_grad = True
               else:
                   param.requires_grad = False
       
       # Stage 1: res4, res5 학습
       elif stage == 1:
           for name, param in self.model.backbone.named_parameters():
               if 'res4' in name or 'res5' in name:
                   param.requires_grad = True
               else:
                   param.requires_grad = False
       
       # Stage 2: 전체 백본 학습
       else:
           for param in self.model.backbone.parameters():
               param.requires_grad = True

   @classmethod
   def build_evaluator(cls, cfg, dataset_name):
       output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
       return COCOEvaluator(dataset_name, output_dir=output_folder)

   def build_hooks(self):
       hooks = super().build_hooks()
       hooks.insert(-1, EarlyStoppingHook(patience=5, metric_name="bbox/AP"))
       return hooks

def register_dataset(name, json_file, image_root):
   if name in DatasetCatalog:
       DatasetCatalog.remove(name)
       MetadataCatalog.remove(name)
       
   with open(json_file, 'r') as f:
       dataset = json.load(f)
   
   categories = {cat['id']: cat['name'] for cat in dataset['categories']}
   category_id_to_contiguous_id = {k: i for i, k in enumerate(sorted(categories.keys()))}
   thing_classes = [categories[k] for k in sorted(categories.keys())]

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
               mapped_category_id = category_id_to_contiguous_id[anno['category_id']]
               obj = {
                   "bbox": anno['bbox'],
                   "bbox_mode": BoxMode.XYWH_ABS,
                   "category_id": mapped_category_id,
                   "segmentation": anno['segmentation']
               }
               objs.append(obj)
           record["annotations"] = objs
           dataset_dicts.append(record)
       return dataset_dicts

   DatasetCatalog.register(name, get_dicts)
   MetadataCatalog.get(name).set(thing_classes=thing_classes)

def visualize_dataset(dataset_name, predictor, num_samples=3):
   dataset_dicts = DatasetCatalog.get(dataset_name)
   metadata = MetadataCatalog.get(dataset_name)

   for d in random.sample(dataset_dicts, min(num_samples, len(dataset_dicts))):
       img = cv2.imread(d["file_name"])
       visualizer_gt = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
       vis_gt = visualizer_gt.draw_dataset_dict(d)

       outputs = predictor(img)
       instances = outputs["instances"].to("cpu")

       visualizer_pred = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
       vis_pred = visualizer_pred.draw_instance_predictions(instances)

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

def get_stage_config(stage):
   """각 스테이지별 설정을 반환합니다."""
   configs = {
       0: {  # Stage 0: res5만 학습
           'iterations': 3000,
           'lr': 0.001,
           'rpn_config': {
               'PRE_NMS_TOPK_TRAIN': 1000,
               'POST_NMS_TOPK_TRAIN': 500,
               'NMS_THRESH': 0.7,
               'LOSS_WEIGHT': 2.0
           }
       },
       1: {  # Stage 1: res4, res5 학습
           'iterations': 3000,
           'lr': 0.0005,
           'rpn_config': {
               'PRE_NMS_TOPK_TRAIN': 1000,
               'POST_NMS_TOPK_TRAIN': 500,
               'NMS_THRESH': 0.7,
               'LOSS_WEIGHT': 2.0
           }
       },
       2: {  # Stage 2: 전체 백본 학습
           'iterations': 2000,
           'lr': 0.0001,
           'rpn_config': {
               'PRE_NMS_TOPK_TRAIN': 1000,
               'POST_NMS_TOPK_TRAIN': 500,
               'NMS_THRESH': 0.7,
               'LOSS_WEIGHT': 2.0
           }
       }
   }
   return configs[stage]

def train_model(base_dir="/home/work/kimjongjip/solar"):
    # 데이터셋 등록
    for split in ['train', 'valid', 'test']:
        json_file = os.path.join(base_dir, split, '_annotations.coco.json')
        image_root = os.path.join(base_dir, split)
        dataset_name = f"solar_{split}"
        register_dataset(dataset_name, json_file, image_root)

    # 기본 설정
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("solar_train",)
    cfg.DATASETS.TEST = ("solar_valid",)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # 공통 설정 - 메모리 사용량 최적화
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get("solar_train").thing_classes)
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.DEVICE = "cuda"
    
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    cfg.INPUT.RANDOM_ROTATION = [-15, 15]

    # 출력 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.OUTPUT_DIR = f"output/solar_detection_{timestamp}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    best_model_path = None
    best_performance = 0

    # 단계별 학습
    for stage in range(3):
        clear_gpu_cache()  # 각 단계 시작 전 GPU 캐시 정리
        print(f"\n=== Stage {stage} 학습 시작 ===")
        stage_cfg = get_stage_config(stage)
        
        # 스테이지별 설정 업데이트
        cfg.SOLVER.MAX_ITER = stage_cfg['iterations']
        cfg.SOLVER.BASE_LR = stage_cfg['lr']
        cfg.SOLVER.STEPS = (int(stage_cfg['iterations'] * 0.7), 
                          int(stage_cfg['iterations'] * 0.9))
        cfg.SOLVER.CHECKPOINT_PERIOD = int(stage_cfg['iterations'] / 5)

        # RPN 설정 업데이트
        rpn_config = stage_cfg['rpn_config']
        cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = rpn_config['PRE_NMS_TOPK_TRAIN']
        cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = rpn_config['POST_NMS_TOPK_TRAIN']
        cfg.MODEL.RPN.NMS_THRESH = rpn_config['NMS_THRESH']
        cfg.MODEL.RPN.LOSS_WEIGHT = rpn_config['LOSS_WEIGHT']

        # 이전 스테이지의 best 모델이 있다면 로드
        if best_model_path is not None:
            cfg.MODEL.WEIGHTS = best_model_path

        try:
            trainer = StageTrainer(cfg, stage=stage)
            trainer.resume_or_load(resume=False)
            trainer.train()
        except RuntimeError as e:
            print(f"Stage {stage} 학습 중 오류 발생: {e}")
            clear_gpu_cache()
            continue

        try:
            # 현재 스테이지 평가
            evaluator = COCOEvaluator("solar_valid", cfg, False, output_dir=cfg.OUTPUT_DIR)
            val_loader = build_detection_test_loader(cfg, "solar_valid")
            results = inference_on_dataset(trainer.model, val_loader, evaluator)
            
            # 최고 성능 모델 저장
            current_performance = results["segm"]["AP"]
            if current_performance > best_performance:
                best_performance = current_performance
                best_model_path = os.path.join(cfg.OUTPUT_DIR, f"model_stage_{stage}_best.pth")
                torch.save(trainer.model.state_dict(), best_model_path)
            else:
                not_best_performance = current_performance
                not_best_model_path = os.path.join(cfg.OUTPUT_DIR, f"model_stage_{stage}_not_best.pth")
                torch.save(trainer.model.state_dict(), not_best_model_path)
                
        except Exception as e:
            print(f"Stage {stage} 평가 중 오류 발생: {e}")
            clear_gpu_cache()
            continue

    # 최종 테스트 평가
    print("\n=== 최종 평가 시작 ===")
    cfg.DATASETS.TEST = ("solar_test",)
    cfg.MODEL.WEIGHTS = best_model_path
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("solar_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg, "solar_test")
    results = inference_on_dataset(predictor.model, test_loader, evaluator)
    print("최종 결과:", results)

    # 테스트 데이터셋 시각화
    print("\n테스트 데이터셋 시각화...")
    visualize_dataset("solar_test", predictor, num_samples=5)


# 메인 코드 시작 부분에 추가
if __name__ == "__main__":
    setup_logger()
    train_model()




