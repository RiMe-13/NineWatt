import os
import psycopg2
import requests
import base64

def update_zoom_level(id, zoom_lev):
    conn = psycopg2.connect(
        host="greenplannerdb.co9wquvnfbh3.ap-northeast-2.rds.amazonaws.com",
        database="vertiport",
        user="ninewattdev",
        password="vtp1qazXSW@"
    )

    cur = conn.cursor()

    query = f"""
    UPDATE verti_demo SET zoom_level = {zoom_lev} WHERE id = {id};
    """

    try:
        # 쿼리 실행
        cur.execute(query)
        # 변경 사항 저장
        conn.commit()
        print(f"{id} 업데이트 성공!")
    except Exception as e:
        # 에러 발생 시 롤백
        conn.rollback()
        print(f"{id} 업데이트 실패:", e)
    finally:
        # 연결 종료
        cur.close()
        conn.close()

def update_image_url(id, bbox_url, original_url):
    conn = psycopg2.connect(
        host="greenplannerdb.co9wquvnfbh3.ap-northeast-2.rds.amazonaws.com",
        database="vertiport",
        user="ninewattdev",
        password="vtp1qazXSW@"
    )

    cur = conn.cursor()

    query = f"""
    UPDATE verti_demo SET original_image = '{original_url}', bbox_image = '{bbox_url}' WHERE id = {id};
    """

    try:
        # 쿼리 실행
        cur.execute(query)
        # 변경 사항 저장
        conn.commit()
        print(f"{id} 이미지 URL 업데이트 성공!")
    except Exception as e:
        # 에러 발생 시 롤백
        conn.rollback()
        print(f"{id} 업데이트 실패:", e)
    finally:
        # 연결 종료
        cur.close()
        conn.close()

def update_image_base64(id, bbox_base64, original_base64):
    conn = psycopg2.connect(
        host="greenplannerdb.co9wquvnfbh3.ap-northeast-2.rds.amazonaws.com",
        database="vertiport",
        user="ninewattdev",
        password="vtp1qazXSW@"
    )

    cur = conn.cursor()

    query = f"""
    UPDATE verti_demo SET original_image = '{original_base64}', bbox_image = '{bbox_base64}' WHERE id = {id};
    """

    try:
        # 쿼리 실행
        cur.execute(query)
        # 변경 사항 저장
        conn.commit()
        print(f"{id} base64 이미지 업데이트 성공!")
    except Exception as e:
        # 에러 발생 시 롤백
        conn.rollback()
        print(f"{id} 업데이트 실패:", e)
    finally:
        # 연결 종료
        cur.close()
        conn.close()

def download_image(id, lev, width, height, x1, y1, x2, y2, x3, y3, x4, y4, center_x, center_y):
    max_length = max(width, height)

    if max_length < 35:
        zoom_lev = 21
    elif max_length < 55:
        zoom_lev = 20
    elif max_length < 85:
        zoom_lev = 19
    elif max_length < 150:
        zoom_lev = 18
    elif max_length < 300:
        zoom_lev = 17
    else:
        zoom_lev = 16

    # update_zoom_level(id, zoom_lev)

    # 구별 폴더 경로 생성
    base_dir = 'image_test_final_2/'
    bbox_dir = os.path.join(base_dir, 'bbox')
    original_dir = os.path.join(base_dir, 'original')
    os.makedirs(bbox_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)

    # API URL 생성
    api_url = f"https://maps.googleapis.com/maps/api/staticmap?zoom={zoom_lev}&size=640x640&maptype=satellite&path=color:0xff0000ff|weight:5|{y1},{x1}|{y2},{x2}|{y3},{x3}|{y4},{x4}|{y1},{x1}&key=AIzaSyAKirLGwgjFBjzNMjvLMt8r4-m2jxH7T2Q"
    api_url2 = f"https://maps.googleapis.com/maps/api/staticmap?center={center_y},{center_x}&zoom={zoom_lev}&size=640x640&maptype=satellite&key=AIzaSyAKirLGwgjFBjzNMjvLMt8r4-m2jxH7T2Q"
    
    # update_image_url(id, api_url, api_url2)
    
    # Bounding box 이미지 다운로드
    response = requests.get(api_url)
    if response.status_code == 200:
        #with open(f'{bbox_dir}/{id}_bbox.jpg', 'wb') as file:
        #    file.write(response.content)
        bbox_base64 = base64.b64encode(response.content).decode('utf-8')
        print(f"Image successfully downloaded and saved as {bbox_dir}/{id}_bbox.jpg")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")

    # Original 이미지 다운로드
    response = requests.get(api_url2)
    if response.status_code == 200:
        #with open(f'{original_dir}/{id}.jpg', 'wb') as file:
        #    file.write(response.content)
        original_base64 = base64.b64encode(response.content).decode('utf-8')
        print(f"Image successfully downloaded and saved as {original_dir}/{id}.jpg")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
    
    update_image_base64(id, bbox_base64, original_base64)
    

# PostgreSQL에 연결
conn = psycopg2.connect(
    host="greenplannerdb.co9wquvnfbh3.ap-northeast-2.rds.amazonaws.com",
    database="vertiport",
    user="ninewattdev",
    password="vtp1qazXSW@"
)

# 커서 생성
cur = conn.cursor()

# SQL 쿼리 작성
query = """
WITH envelope AS (
SELECT id, circle_lev, 주소, geom, ST_Envelope(ST_Transform(geom, 5186)) AS bbox
FROM verti_debug
WHERE circle_lev != 0 AND 주소 LIKE '%서울%' AND id IN (SELECT id FROM verti_demo)
)
SELECT id, circle_lev, 
    ABS(ST_X(ST_PointN(ST_ExteriorRing(bbox), 3)) - ST_X(ST_PointN(ST_ExteriorRing(bbox), 1))) AS width,  
    ABS(ST_Y(ST_PointN(ST_ExteriorRing(bbox), 2)) - ST_Y(ST_PointN(ST_ExteriorRing(bbox), 1))) AS height,
    ST_X(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 1), 4326)) AS x1, ST_Y(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 1), 4326)) AS y1,
    ST_X(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 2), 4326)) AS x2, ST_Y(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 2), 4326)) AS y2,
    ST_X(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 3), 4326)) AS x3, ST_Y(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 3), 4326)) AS y3,
    ST_X(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 4), 4326)) AS x4, ST_Y(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 4), 4326)) AS y4,
    (ST_X(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 1), 4326)) + 
     ST_X(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 2), 4326)) + 
     ST_X(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 3), 4326)) + 
     ST_X(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 4), 4326))) / 4 AS center_x,
    (ST_Y(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 1), 4326)) + 
     ST_Y(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 2), 4326)) + 
     ST_Y(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 3), 4326)) + 
     ST_Y(ST_Transform(ST_PointN(ST_ExteriorRing(bbox), 4), 4326))) / 4 AS center_y
FROM envelope
ORDER BY id;
"""

# 쿼리 실행
cur.execute(query)

# 결과 가져오기 및 이미지 다운로드
results = cur.fetchall()
for row in results:
    id, lev, width, height, x1, y1, x2, y2, x3, y3, x4, y4, center_x, center_y = row
    download_image(id, lev, width, height, x1, y1, x2, y2, x3, y3, x4, y4, center_x, center_y)

# 커서 및 연결 닫기
cur.close()
conn.close()
