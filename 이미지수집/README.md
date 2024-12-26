# 이미지 수집 및 전처리
건물 이미지 수집을 진행하는 디렉토리입니다.

DB로부터 건물의 특성을 가져오고, Google Maps API를 활용하여 원본 이미지 및 bounding box 이미지를 가져와 DB에 저장하는 역할을 합니다.
## 개발 환경 
- Ubuntu 22.04 LTS (WSL 2 활용)
- Python 3.10.12

## 필요 라이브러리
- os
- psycopg2
- requests
- base64

## 사용 API
- [Google Maps Static API (1000장 당 $4)](https://developers.google.com/maps/documentation/maps-static/overview)

## 사용 목적
```
Google API 활용하여 위성 사진 수집 및 해당 사진 DB 내에 저장
현재 DB 내 verti_demo 테이블에 후보지 별 건물 위성 사진 저장 완료
```

## DB 정보 및 API Key 변경
```
your_host -> DB 호스트 입력
YOUR_API_KEY -> Google Maps API Key 입력
your_password -> DB 비밀번호 입력
```

## 실행 방법
```
python3 image_db.py
```
