# SAM 분할
SAM 분할을 위한 코드입니다. SAM은 기본적으로 입력 prompt가 주어지면 해당 prompt에 따라 이미지 분할을 진행하는 general-purpose 모델입니다.

SAM의 prompt 에는 점, 바운딩 박스, 텍스트 등이 있습니다만 논문을 통해 나온 결과로는 바운딩 박스를 통한 분할 성능이 가장 좋은 것으로 알려져 있습니다.

번외로는 아무런 prompt를 주지 않고 auto-segmentation을 하는 기능도 있습니다만 가장 정확한 분할을 위하여 바운딩 박스를 통해 진행하였습니다.

참고하셔야 할 사항은 데이터 전처리를 통해 구축된 바운딩 박스 이미지와 원본 이미지가 1대1로 매칭되어있다는 가정하에 코드를 작성하였습니다.

바운딩 박스 이미지가 주어지면 해당 이미지에서 빨간색 박스를 cv2 라이브러리를 통해 감지하고, 박스의 좌푯값을 원본 이미지와 함께 SAM에 전달한 뒤, plot을 통해 분할 이미지를 시각화하는 코드입니다.

SAM은 분할 결과로 한 이미지당 세가지 분할 이미지를 줍니다. 이는 사용자에게 여러가지 결과 케이스를 제공하여 애매 모호한 결과에 대해 선택할 수 있게 함입니다.

저희는 건물 옥상이라는 상대적으로 큰 이미지를 목표로 하기에 SAM의 분할 결과중 가장 큰 분할 이미지를 최종 결과로 뽑게 하였습니다.


## 환경설정

SAM install을 진행합니다.

```
!는 사용 환경에따라 빼시면 됩니다. 
!pip install torch torchvision torchaudio
!pip install git+https://github.com/facebookresearch/segment-anything.git
!pip install opencv-python-headless matplotlib numpy
```

SAM의 기본 가중치 파일이 필요합니다. vit_l (large) 모델 가중치 파일을 다운받는 코드입니다. vit_h / vit_l / vit_b 세 종류가 있고, 

vit_h 모델 가중치가 가장 무겁고 정확도가 높은 것으로 알려져 있으나 실험 결과 저희의 옥상 분할 테스크에서는 vit_l 모델로도 충분한 성능을 발휘했습니다.


```
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```
sample_folder 라는 폴더를 생성합니다. 전처리된 이미지와 경계 박스 이미지들을 넣어둡니다. 

## segmentation.py 설명

sample_foler 라는 폴더 안에 .jpg로 끝나는 원본 이미지와 _bbox.jpg로 끝나는 경계 박스 이미지가 같이 공존하고 있다고 가정합니다.

이미지이름_bbox.jpg 와 같이 _bbox.jpg로 끝나는 이미지들을 순회하며 원본 이미지인 이미지이름.jpg를 찾습니다. 

bbox.jpg 이미지에서 cv2라이블러리를 통해 빨간색 박스를 감지합니다. 그 후, SAM 모델에 해당 박스값과 원본 jpg파일을 넘겨주어 분할을 진행합니다. 

마지막으로 plt를 통해 시각화를 진행합니다. 

## 샘플 결과 화면

![image](https://github.com/user-attachments/assets/308f916d-93a8-4455-bb5c-f9e4d1a0bfb8)

위쪽 이미지에는 _bbox.jpg 이미지가, 좌측 하단에는 원본 이미지가, 우측 하단에는 경계 박스를 기준으로 SAM 분할을 진행한 가장 큰 마스크 추출 이미지가 존재합니다.

