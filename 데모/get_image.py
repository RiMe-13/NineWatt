import io
import psycopg2
import base64
from PIL import Image

def download_image(id, zoom_lev, original_base64, bbox_base64):
    # Original 이미지 디코딩
    original_image = base64.b64decode(original_base64)
    original_img = Image.open(io.BytesIO(original_image))

    # Bounding box 이미지 디코딩
    bbox_image = base64.b64decode(bbox_base64)
    bbox_img = Image.open(io.BytesIO(bbox_image))

    return original_img, bbox_img, zoom_lev


def get_images(id):
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
    query = f"""
    SELECT id, zoom_level, original_image, bbox_image
    FROM verti_demo
    WHERE id = {id};
    """

    # 쿼리 실행
    cur.execute(query)
    print(f"Query executed for id: {id}")

    # 결과 가져오기 및 이미지 다운로드
    results = cur.fetchall()
    for row in results:
        id, zoom_lev, original_base64, bbox_base64 = row
        original_img, bbox_img, zoom_lev = download_image(id, zoom_lev, original_base64, bbox_base64)

    # 커서 및 연결 닫기
    cur.close()
    conn.close()

    return original_img, bbox_img, zoom_lev