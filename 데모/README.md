# Demo

## 환경 설정
### 모델 가중치 다운로드
- [Google Drive](https://drive.google.com/file/d/1fNrkIfhOnGlnAF9g11uj0PHbQ1rkxtxc/view?usp=drive_link)

### detectron2, SAM 설치
```
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/facebookresearch/detectron2.git
```

### 필요 라이브러리 설치
```
pip install torch torchvision torchaudio
pip install opencv-python-headless matplotlib numpy
pip install gradio pillow ultralytics
```

### DB 비밀번호 변경경
```
your_password -> DB 비밀번호
```

## 사용 버전
- Python 3.10.12
- detectron2 0.6
- SAM 1
- YOLOv8seg
- ultralytics 8.3.32
- PIL 10.3.0
- openCV 4.10.0.84
- torch 2.5.1+cu124
- torchvision 0.20.1+cu124
- numpy 1.26.4
- gradio 5.6.0
- psycopg2 2.9.9
- matplotlib 3.8.4

## 사용 모델
```
best.pt (YOLOv8seg)
sam_vit_l_0b3195.pth (SAM)
model_final.pth (Mask-RCNN)
```

## 최종 디렉토리 구조 
```
Ninewatt/데모/  
├── images  
│ ├── evtol.png  
├── best.pt    
├── calculate_circle.py  
├── demo.py  
├── get_image.py  
├── mask_generate.py  (maskRCNN)
├── model_final.pth  
├── sam_segment.py  (SAM)
├── sam_vit_l_0b3195.pth  
├── get_image.py  
└── yolo_mask_generate.py (YoloV8)
```

## 데모 실행
```
python3 demo.py
```

## Reference
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Gradio](https://www.gradio.app/)
