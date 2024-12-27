# image dataset

## 데이터셋 소개
서울특별시 내의 버티포트 후보지로 판단한 건물 중 SAM의 분할 성능이 좋은 건물 위성 이미지 총 766장을 모아놓은 데이터셋

## 데이터셋 구조
- `모델학습/`: 딥러닝 모델의 Fine-tuning을 위한 데이터셋
  - `train` : `valid` : `test` = 7 : 2 : 1로 구성
- `분할/`: 건물 원본 및 bounding box 위성사진 데이터셋
  - SAM을 이용한 건물 분할에 사용
  - `original/`: 건물 원본 이미지, SAM input으로 사용
  - `bbox/`: 건물 bounding box 이미지, SAM의 prompt로 활용
