# 이미지 수집 및 전처리

## 버전 
- Python 3.10.12

## 필요 라이브러리
- os
- psycopg2
- requests
- base64

## 사용 API
- Google Maps Static API (1000장 당 $4)

## 사용 목적
```
위성 사진 수집 및 해당 사진 DB 내에 저장
현재 DB 내 verti_demo 테이블에 후보지 별 건물 위성 사진 저장 완료
```

## 실행 방법
```
python3 image_db.py
```