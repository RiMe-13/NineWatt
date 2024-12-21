# Mask RCNN 환경

## 환경 설정

- detectron2 설치
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## 버전 
- detectron2 0.6
- openCV 4.10.0.84
- torch 1.12.0a0+8a1a93a
- numpy 1.22.3

## 사용 모델
'''
mask_rcnn_R_50_FPN_3x.yaml
'''

## 데이터셋 
- CoCo형식 데이터셋
- 데이터 셋 폴더 구조
'''
/home/work/kimjongjip/solar  
├── train  
│ ├── annotations.coco.json  
│ ├── images  
│ │ ├── 000000000000.jpg  
│ │ ├── 000000000001.jpg  
│ │ └── ...  
├── valid  
└── test
'''

## 학습 및 검증
'''
python train_level.py
python test_new.py
'''

## 마스크 생성
'''
python mask_generate.py
'''

