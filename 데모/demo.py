import gradio as gr
import psycopg2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
from get_image import get_images  # 이미지 가져오기 함수 가져오기
from sam_segment import segment_image  # SAM 분할 함수 가져오기
from mask_generate import process_single_image  # 마스크 생성 함수 가져오기
from yolo_mask_generate import process_image_yolo  # YOLO 마스크 생성 함수 가져오기
from calculate_circle import find_max_circle, draw_circle_on_original_image, bring_evtol  # 최대 원 계산 함수 가져오기

# PostgreSQL 연결 설정
def get_db_connection():
    return psycopg2.connect(
        user="ninewattdev",
        password="your_password",
        host="your_host",
        port="5432",
        database="vertiport"
    )

# 주소 가져오기 함수
def get_address_from_db(selected_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        query = """
            SELECT 도로명주소, 구주소
            FROM verti_demo
            WHERE id = %s
        """
        cursor.execute(query, (selected_id,))
        result = cursor.fetchone()
        if result:
            도로명주소, 구주소 = result
            return 도로명주소 if 도로명주소 else 구주소
        else:
            return "해당 ID의 주소를 찾을 수 없습니다."
    except Exception as e:
        return f"데이터베이스 오류: {e}"
    finally:
        if connection:
            cursor.close()
            connection.close()

# DB 업데이트 함수
def update_db(selected_id, size, diameter, heli):
    if size == "small":
        verti_lev = 1
    elif size == "medium":
        verti_lev = 2
    elif size == "large":
        verti_lev = 3
    else:
        verti_lev = 0
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        query = f"""
            UPDATE verti_demo
            SET verti_lev = {verti_lev}, diameter = {diameter}, heli = {heli}
            WHERE id = {selected_id};
        """
        cursor.execute(query)
        connection.commit()
        print(f"{selected_id} DB 업데이트 완료")
    except Exception as e:
        print(f"데이터베이스 오류: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

# 이미지 크기 설정
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 600

# 초기 빈 이미지를 생성하여 고정된 크기로 설정
def create_placeholder_image(width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    placeholder = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(placeholder)
    text = "The result image will be shown here."

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    draw.text(
        ((width - text_width) / 2, (height - text_height) / 2),
        text,
        fill=(200, 200, 200),
        font=font
    )
    return placeholder

placeholder_image = create_placeholder_image()

# Helper 함수: 버티포트 크기 만들기
def generate_button_html(label, diameter, height, active=False, color="#db7385", text_color = "black"):
    if active:
        return f'''
        <button style="
            flex: 1;
            background-color: {color};
            color: {text_color};
            border: none;
            padding: 10px 0;
            border-radius: 5px;
            font-weight: bold;
            cursor: default;
            box-sizing: border-box;
            height: {height}px;
        " disabled>{label} : {diameter:.2f}m</button>
        '''
    else:
        return f'''
        <button style="
            flex: 1;
            background-color: lightgray;
            color: black;
            border: none;
            padding: 10px 0;
            border-radius: 5px;
            cursor: default;
            box-sizing: border-box;
            height: {height}px;
        " disabled>{label} : {diameter:.2f}m</button>
        '''

# New Helper 함수: 헬리패드 버튼 HTML 생성
def generate_heli_button_html(active=False, color="#db7385", text_color="black"):
    if active:
        return f'''
        <button style="
            width: 100%;
            background-color: {color};
            color: {text_color};
            border: none;
            padding: 10px 0;
            border-radius: 5px;
            font-weight: bold;
            cursor: default;
            box-sizing: border-box;
            height: 50px;
            width: 80px;
        " disabled>헬리패드</button>
        '''
    else:
        return f'''
        <button style="
            width: 100%;
            background-color: lightgray;
            color: black;
            border: none;
            padding: 10px 0;
            border-radius: 5px;
            cursor: default;
            box-sizing: border-box;
            height: 50px;
            width: 80px;
        " disabled>헬리패드</button>
        '''

# Modified Helper 함수: 네모 박스 HTML 생성 (Includes Heli Button)
def generate_box_html(small_btn, medium_btn, large_btn, heli_btn):
    return f"""
    <div style="
        border: 1px solid #ccc;
        padding: 20px;
        border-radius: 5px;
        margin-top: 10px;
    ">
        <strong>버티포트 크기 기준</strong>
        <div style="
            display: flex;
            justify-content: space-between;
            gap: 10px; /* 버튼 간의 간격 */
            margin-top: 10px;
        ">
            {small_btn}
            {medium_btn}
            {large_btn}
        </div>
        <div style="
            margin-top: 20px;
        ">
            {heli_btn}
        </div>
    </div>
    """

# Gradio 인터페이스
with gr.Blocks() as demo:
    # Custom CSS for step buttons
    gr.HTML("""
    <style>
    /* 단계 버튼 색상 변경 */
    #segment_building_button, 
    #detect_obstacles_button, 
    #calculate_area_button, 
    #result_button {
        background-color: orange !important;
        color: black !important;
    }
    </style>
    """)

    gr.Markdown("""
        SKKU 나인와트  
        # 도심 건물 옥상을 대상으로 UAM 이착륙장 입지 선정 모델 개발
    """)

    with gr.Row():
        # 왼쪽 사이드바
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("탐지를 원하는 건물 ID를 선택해주세요.") 
            
            building_id = gr.Dropdown(
                label="건물 ID 선택",
                choices=["Please select an ID", 
                "273710", "273869", "275708", "275735", "275904", "277349", "277382", "277484", "280414", "281016", "281861", "281914",
                "282060", "282377", "282417", "282438", "282989", "285693", "285901", "285909", "286067", "286358", "286946", "287106",
                "288950", "290487", "291025", "291038", "291040", "292823", "292876", "294595", "295000", "295018", "295021", "295081",
                "273710", "273869", "275708", "275735", "275904", "277349", "277382", "277484", "280414", "281016", "281861", "281914",
                "282060", "282377", "282417", "282438", "282989", "285693", "285901", "285909", "286067", "286358", "286946", "287106",
                "288950", "290487", "291025", "291038", "291040", "292823", "292876", "294595", "295000", "295018", "295021", "295081",
                "295714", "296094", "296418", "297399", "297402", "297635", "298914", "298931", "298937", "303363", "303447", "303519",
                "304662", "304940", "304943", "305182", "305298", "305304", "305346", "305348", "305350", "305428", "305446", "305449",
                "305791", "305792", "305812", "305814", "305820", "305849", "305857", "305888", "305904", "306173", "306729", "306985",
                "311935", "311938", "311955", "311959", "311960", "313813", "313823", "319357", "319382", "324650", "324799", "324831",
                "325777", "326325", "327211", "327221", "327222", "327479", "328156", "330313", "331103", "331202", "331350", "332015",
                "332026", "333632", "333859", "334289", "335126", "335181", "336043", "336746", "337312", "337547", "337689", "338556",
                "338760", "340322", "340438", "343697", "344234", "344242", "344795", "344796", "344806", "348893", "355437", "355447",
                "355696", "359083", "359525", "359634", "359685", "359727", "359729", "359744", "360430", "360449", "360464", "360467",
                "360543", "360576", "361551", "362008", "362131", "362132", "362297", "362322", "362407", "362721", "362722", "363047",
                "363315", "363586", "363908", "364064", "365113", "369552", "369553", "369555", "369559", "369791", "369864", "372307",
                "372502", "372999", "374033", "374462", "374465", "374467", "375661", "375827", "375832", "385485", "385767", "385773",
                "385802", "385834", "385846", "385850", "386310", "386665", "387288", "388082", "388116", "388203", "388248", "388350",
                "389560", "389567", "389733", "389750", "389856", "389929", "389969", "390061", "390151", "390174", "390490", "390524",
                "390580", "391079", "391093", "391127", "391301", "391322", "391451", "391753", "391975", "392264", "392265", "392266",
                "392343", "392351", "392717", "392798", "392804", "393019", "393321", "393675", "393759", "394132", "395305", "395534",
                "396928", "397561", "397663", "398635", "399977", "400041", "400310", "400607", "400627", "400673", "400842", "400843",
                "401343", "401973", "401974", "402060", "402119", "402131", "402132", "402310", "402354", "402362", "402376", "402395",
                "402750", "403464", "403769", "403772", "403865", "403866", "403920", "403931", "403982", "404245", "404330", "404448",
                "404565", "404570", "404572", "404579", "404581", "404582", "404586", "404594", "404600", "404636", "404638", "404640",
                "404733", "404734", "404741", "404900", "404901", "404904", "405461", "405501", "405925", "405953", "406047", "406280",
                "406312", "406343", "406346", "406347", "406348", "406355", "406359", "406361", "406364", "406373", "407289", "408253",
                "408360", "408563", "408569", "409106", "409168", "409261", "409341", "409506", "409533", "409916", "409971", "410050",
                "411052", "412712", "412777", "412786", "413279", "413608", "414071", "414955", "415036", "416262", "416272", "416281",
                "416283", "416284", "416290", "416528", "416610", "416614", "416698", "416748", "416777", "416779", "416782", "416792",
                "416801", "416819", "416824", "416829", "417190", "417191", "417663", "421399", "422110", "422965", "423745", "425200",
                "425419", "425429", "425433", "425581", "425952", "425977", "426183", "426200", "426210", "426540", "427383", "427653",
                "428129", "428187", "428190", "428299", "428309", "428358", "428364", "428376", "428381", "428388", "428390", "428450",
                "428576", "428940", "431060", "432083", "432094", "432831", "433397", "435881", "436953", "437854", "438198", "440214",
                "440420", "440985", "441165", "441231", "441235", "441273", "441415", "441432", "442036", "442049", "442192", "442265",
                "442655", "442676", "442749", "442869", "443395", "443531", "443826", "444154", "444261", "444293", "444598", "444631",
                "444694", "444764", "444831", "445438", "445457", "445680", "445811", "446166", "446605", "448237", "449200", "449203",
                "449228", "449230", "449231", "449232", "449233", "449235", "449254", "449324", "449326", "449339", "449341", "449345",
                "449385", "449387", "449389", "453732", "453788", "453822", "453841", "454251", "454254", "454255", "454256", "454262",
                "454850", "454876", "454935", "454957", "454972", "454976", "454979", "454988", "454990", "454992", "454993", "455007",
                "455149", "455150", "455152", "455153", "455154", "455169", "455180", "456269", "456313", "456605", "456634", "456945",
                "456949", "456965", "458048", "458083", "458087", "458088", "458095", "458097", "458100", "458103", "458107", "458109",
                "458112", "458113", "458116", "458123", "458138", "458143", "458145", "458147", "458148", "458155", "458168", "458190",
                "459271", "460869", "461035", "461506", "461981", "462144", "462159", "462186", "462225", "462251", "462295", "462366",
                "462438", "462474", "462542", "463227", "463557", "464770", "464782", "464837", "465050", "465068", "466630", "466870",
                "467021", "467517", "467647", "467689", "467706", "468690", "468776", "469367", "470295", "471798", "472229", "472812",
                "474503", "474617", "479910", "480120", "480586", "480587", "480863", "480866", "480872", "480883", "480893", "480903",
                "480908", "480912", "480915", "480924", "480942", "480974", "481587", "481639", "481712", "481937", "481948", "481949",
                "482010", "482015", "482026", "482077", "482108", "482150", "482219", "482231", "482325", "482411", "482418", "482432",
                "482528", "482604", "482609", "482610", "482611", "482615", "482626", "482852", "482855", "483108", "483143", "484155",
                "484232", "484257", "484303", "484502", "484699", "484768", "484798", "484822", "485664", "485948", "485949", "486444",
                "486455", "486887", "487574", "487582", "487589", "487730", "487941", "488554", "488589", "490712", "490730", "490758",
                "490853", "494602", "494699", "496756", "496784", "496806", "496894", "496916", "496930", "496936", "496956", "496968",
                "496983", "497000", "497112", "497509", "498079", "498491", "498523", "498608", "498802", "499136", "499167", "499246",
                "499633", "499772", "499824", "500173", "500177", "500283", "500391", "500644", "500930", "501120", "501182", "501234",
                "501300", "501319", "501392", "501402", "501444", "501449", "501498", "501509", "501510", "501512", "501516", "501606",
                "501641", "501700", "501834", "501865", "502752", "502864", "503137", "504090", "504124", "504256", "504671", "504712",
                "504719", "504735", "504943", "506886", "506900", "506989", "508104", "508425", "508464", "508547", "508886", "508887",
                "509058", "509583", "509592", "509608", "509609", "509855", "509952", "510148", "510321", "510614", "510719", "510756",
                "510831", "510833", "510834", "510835", "510873", "510938", "511684", "512444", "512460", "514647", "515148", "515209",
                "516890", "517067", "517083", "517463", "517824", "518651", "519223", "519261", "519546", "520181", "520422", "520620",
                "520705", "520738", "520816", "520831", "520833", "520926", "521170", "523092", "523166", "523641", "524222", "525119",
                "525137", "525588", "526700", "529500", "535335", "542888", "545312", "551827", "552163", "553294", "553358", "553419",
                "553915", "554478", "554917", "554927", "554966", "555019", "555025", "555036", "555043", "555059", "556139", "556523",
                "562604", "563143", "563155", "563327", "563450", "563677", "563797", "565551", "565583", "565828", "566994", "567614",
                "567657", "567658", "568238", "570024", "572019", "578694", "580578", "581510", "586521", "586554", "586692", "586857",
                "586938", "587623", "588496", "589740", "589741", "589880", "589947", "591604", "591643", "591976"
                ],
                value="Please select an ID",
                interactive=True
            )

            address_placeholder = gr.Textbox(
                label="건물 주소",
                placeholder="주소가 여기에 표시됩니다.",
                interactive=False
            )

            # 단계 버튼에 고유한 elem_id 부여
            segment_building_button = gr.Button("건물 분할", interactive=False, elem_id="segment_building_button")
            detect_obstacles_button = gr.Button("가용 공간 추출", interactive=False, elem_id="detect_obstacles_button")
            calculate_area_button = gr.Button("최대 원 계산", interactive=False, elem_id="calculate_area_button")
            result_button = gr.Button("버티포트 수용성 판단", interactive=False, elem_id="result_button")

            # (1) 버튼 그룹화 및 레이블 추가 using gr.HTML
            # 초기 버튼 상태 설정 (높이: 50, 60, 70)
            # Generate the heli button HTML
            initial_heli_html = generate_heli_button_html()

            initial_box_html = generate_box_html(
                generate_button_html('소형', 2 * math.sqrt(2) * 6, 50),
                generate_button_html('중형', 2 * math.sqrt(2) * 12, 60),
                generate_button_html('대형', 2 * math.sqrt(2) * 16, 70),
                initial_heli_html  # Include the heli button inside the box
            )

            # 그리드 박스 컴포넌트 생성
            port_size_box = gr.HTML(initial_box_html, elem_id="port_size_box")
            
            # Remove the separate heli_size_box
            # heli_size_box = gr.HTML(generate_heli_button_html(), elem_id="heli_size_box")  # Removed

        # 결과 이미지
        with gr.Column(scale=2):
            result_image_output = gr.Image(
                label="결과 이미지",
                value=placeholder_image,
                interactive=False,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT  # 크기 유지
            )
            # 결과 텍스트를 표시할 Textbox
            result_text = gr.Textbox(
                label="Diameter",
                placeholder=" ",
                interactive=False,
            )

        # 파이프라인
        with gr.Column(scale=1, min_width=200):
            pipeline_0 = gr.Image(label="건물 이미지", value=None, interactive=False, width=125, height=125)
            arrow_0_1 = gr.HTML("<div style='font-size:24px; color: orange; text-align: center; margin:0;'>↓</div>")
            pipeline_1 = gr.Image(label="건물 분할", value=None, interactive=False, width=125, height=125)
            arrow_1_2 = gr.HTML("<div style='font-size:24px; color: orange; text-align: center; margin:0;'>↓</div>")
            pipeline_2 = gr.Image(label="가용 면적", value=None, interactive=False, width=125, height=125)
            arrow_2_3 = gr.HTML("<div style='font-size:24px; color: orange; text-align: center; margin:0;'>↓</div>")
            pipeline_3 = gr.Image(label="최대 원", value=None, interactive=False, width=125, height=125)

    # 상태 저장
    bbox_image_state = gr.State()
    intermediate_output_1 = gr.State()
    intermediate_output_2 = gr.State()
    pipeline_updated = gr.State([False, False, False, False])  # For positions 0 to 3
    zoom_level_state = gr.State()
    max_diameter_state = gr.State()
    max_location_state = gr.State()
    heli_detection_state = gr.State(value=False) # 헬리패드 판단 여부
    building_id_state = gr.State() # 건물 ID

    # 주소 업데이트 함수
    def update_address(selected_id):
        # 모든 파이프라인 이미지과 업데이트 여부 초기화
        reset_pipeline = [
            None,  # pipeline_0
            None,  # pipeline_1
            None,  # pipeline_2
            None   # pipeline_3
        ]
        pipeline_updated.value = [False, False, False, False]

        # 화살표 초기화 (margin:0; 추가)
        reset_arrows = [
            "<div style='font-size:24px; color: lightgray; text-align: center; margin:0;'>↓</div>",
            "<div style='font-size:24px; color: lightgray; text-align: center; margin:0;'>↓</div>",
            "<div style='font-size:24px; color: lightgray; text-align: center; margin:0;'>↓</div>"
        ]

        # 모든 버튼 비활성화 (segment_building_button만 이미지가 유효할 때 활성화)
        buttons_state = [
            gr.update(interactive=False),  # detect_obstacles_button
            gr.update(interactive=False),  # calculate_area_button
            gr.update(interactive=False),  # result_button
        ]

        # Reset the heli button within the same box
        reset_buttons_html = generate_box_html(
            generate_button_html('소형', 2 * math.sqrt(2) * 6, 50),
            generate_button_html('중형', 2 * math.sqrt(2) * 12, 60),
            generate_button_html('대형', 2 * math.sqrt(2) * 16, 70),
            generate_heli_button_html()  # Reset Heli button
        )

        if selected_id == "Please select an ID":
            address = "ID를 선택해주세요."
            result_image = placeholder_image
            bbox_image = None
            zoom_lev = None
            # segment_building_button remains disabled
            segment_button_state = gr.update(interactive=False)
        else:
            address = get_address_from_db(selected_id)
            building_id_state.value = int(selected_id)
            # Fetch images
            try:
                original_image, bbox_image, zoom_lev = get_images(selected_id)
                # 다음 단계 버튼 활성화
                result_image = original_image
                # Enable segment_building_button
                segment_button_state = gr.update(interactive=True)
            except Exception as e:
                print(f"Error in fetch_images in update_address: {e}")
                result_image = placeholder_image
                bbox_image = None
                zoom_lev = None
                segment_button_state = gr.update(interactive=False)

        result_text_reset = gr.update(value="") # 결과 텍스트 초기화

        # (2) 버튼들 초기화 (높이: 50, 60, 70)
        # reset_buttons_html already includes the Heli button

        # 결과 이미지 초기화
        return [
            address,  # 주소
            *buttons_state,  # detect_obstacles_button, calculate_area_button, result_button
            segment_button_state,  # segment_building_button
            result_image,  # 결과 이미지
            bbox_image,  # bbox_image_state
            zoom_lev,    # zoom_level_state
            *reset_pipeline,  # 파이프라인 이미지 초기화
            *reset_arrows,  # 화살표 초기화
            pipeline_updated.value,  # 파이프라인 업데이트 여부 초기화
            reset_buttons_html,  # 버튼들 초기화 (HTML) including Heli button
            result_text_reset  # 결과 텍스트 초기화
        ]

    building_id.change(
        fn=update_address,
        inputs=building_id,
        outputs=[
            address_placeholder,
            detect_obstacles_button, calculate_area_button, result_button,
            segment_building_button,
            result_image_output,  # 결과 이미지 출력
            bbox_image_state,
            zoom_level_state,
            pipeline_0, pipeline_1, pipeline_2, pipeline_3,
            arrow_0_1, arrow_1_2, arrow_2_3,
            pipeline_updated,
            port_size_box,  # 버튼들 초기화 (gr.HTML) including Heli button
            # heli_size_box,  # Removed
            result_text  # 결과 텍스트 초기화
        ]
    )

    # 1. 건물 분할 버튼 클릭 이벤트
    def process_building(input_image, bbox_image):
        if input_image is None or bbox_image is None:
            return placeholder_image, gr.update(interactive=False)
        try:
            # 이미지 형식 변환: PIL Image -> NumPy 배열
            if isinstance(input_image, Image.Image):
                input_image = np.array(input_image.convert("RGB"))
            if isinstance(bbox_image, Image.Image):
                bbox_image = np.array(bbox_image.convert("RGB"))

            # 건물 탐지 함수 호출
            result = segment_image(input_image, bbox_image)

            # 결과 이미지 형식 변환: NumPy 배열 -> PIL Image
            if isinstance(result, np.ndarray):
                result = Image.fromarray(result)

            intermediate_output_1.value = result

            # 다음 단계 버튼 활성화
            return result, gr.update(interactive=True)
        except Exception as e:
            print(f"Error in process_building: {e}")
            return placeholder_image, gr.update(interactive=False)

    segment_building_button.click(
        fn=process_building,
        inputs=[result_image_output, bbox_image_state],
        outputs=[result_image_output, detect_obstacles_button]
    )

    # (1) 건물 분할 후 파이프라인 업데이트
    def update_pipeline_0(result_image, pipeline_updated):
        if not pipeline_updated[0]:
            pipeline_updated[0] = True
            # 화살표 색상을 검은색으로 변경
            new_arrow = "<div style='font-size:24px; color: black; text-align: center;'>↓</div>"
            return [result_image, pipeline_updated, new_arrow]
        else:
            return [gr.NO_CHANGE, gr.NO_CHANGE, gr.NO_CHANGE]

    segment_building_button.click(
        fn=update_pipeline_0,
        inputs=[result_image_output, pipeline_updated],
        outputs=[pipeline_0, pipeline_updated, arrow_0_1]
    )

    # 2. 가용 면적 추출 버튼 클릭 이벤트
    def process_obstacles(input_image):
        if input_image is None:
            return placeholder_image, gr.update(interactive=False), generate_box_html(
                generate_button_html('소형', 2 * math.sqrt(2) * 6, 50),
                generate_button_html('중형', 2 * math.sqrt(2) * 12, 60),
                generate_button_html('대형', 2 * math.sqrt(2) * 16, 70),
                generate_heli_button_html()  # Include Heli button
            )
        try:
            # 이미지 형식 변환: PIL Image -> NumPy 배열
            if isinstance(input_image, Image.Image):
                input_image = np.array(input_image.convert("RGB"))

            # 가용 면적 추출 함수 호출
            # result = process_single_image(input_image)
            result, heli = process_image_yolo(input_image)

            # 결과 이미지 형식 변환: NumPy 배열 -> PIL Image
            if isinstance(result, np.ndarray):
                result = Image.fromarray(result)

            intermediate_output_2.value = result
            heli_detection_state.value = heli

            # 결과 이미지가 모두 검은색인 경우
            mask_array = np.array(result)
            if np.all(mask_array == 0):
                updated_box_html = generate_box_html(
                    generate_button_html('소형', 2 * math.sqrt(2) * 6, 50),
                    generate_button_html('중형', 2 * math.sqrt(2) * 12, 60),
                    generate_button_html('대형', 2 * math.sqrt(2) * 16, 70),
                    generate_heli_button_html()  # Include Heli button reset
                )
                update_db(building_id_state.value, "impossible", 0, False)
                print("No available space detected.")
                return result, gr.update(interactive=False), updated_box_html, gr.update(value="빈 공간이 검출되지 않아 버티포트 설치 불가")  # Return updated_box_html including Heli

            if heli_detection_state.value:
                updated_box_html = generate_box_html(
                    generate_button_html('소형', 2 * math.sqrt(2) * 6, 50),
                    generate_button_html('중형', 2 * math.sqrt(2) * 12, 60),
                    generate_button_html('대형', 2 * math.sqrt(2) * 16, 70),
                    generate_heli_button_html(active=heli, color="#0073cf" if heli else "#db7385", text_color="white" if heli else "black")  # Update Heli button
                )
                result, max_d, max_l = find_max_circle(result)
                result = draw_circle_on_original_image(intermediate_output_1.value, max_l, max_d // 2, 0)
                zoom_lev = zoom_level_state.value
                if zoom_lev == 16:
                    diameter = max_d * 1.815
                elif zoom_lev == 17:
                    diameter = max_d * 0.913
                elif zoom_lev == 18:
                    diameter = max_d * 0.4603
                elif zoom_lev == 19:
                    diameter = max_d * 0.2315
                elif zoom_lev == 20:
                    diameter = max_d * 0.116
                elif zoom_lev == 21:
                    diameter = max_d * 0.0585
                else:
                    diameter = max_d

                if diameter < 2 * math.sqrt(2) * 6:
                    size = "impossible"
                elif diameter < 2 * math.sqrt(2) * 12:
                    size = "small"
                elif diameter < 2 * math.sqrt(2) * 16:
                    size = "medium"
                else:
                    size = "large"

                result_text_update = gr.update(value="헬리패드 사용 가능")
                update_db(building_id_state.value, size, diameter, True)
                return result, gr.update(interactive=False), updated_box_html, result_text_update
            
            # 다음 단계 버튼 활성화
            reset_buttons_html = generate_box_html(
                generate_button_html('소형', 2 * math.sqrt(2) * 6, 50),
                generate_button_html('중형', 2 * math.sqrt(2) * 12, 60),
                generate_button_html('대형', 2 * math.sqrt(2) * 16, 70),
                generate_heli_button_html(active=heli, color="#0073cf" if heli else "#db7385", text_color="white" if heli else "black")  # Update Heli button
            )
            
            return result, gr.update(interactive=True), reset_buttons_html, gr.update(value="")
        except Exception as e:
            print(f"Error in process_obstacles: {e}")
            reset_buttons_html = generate_box_html(
                generate_button_html('소형', 2 * math.sqrt(2) * 6, 50),
                generate_button_html('중형', 2 * math.sqrt(2) * 12, 60),
                generate_button_html('대형', 2 * math.sqrt(2) * 16, 70),
                generate_heli_button_html()  # Include Heli button reset
            )
            return placeholder_image, gr.update(interactive=False), reset_buttons_html, gr.update(value="결과 정보를 생성할 수 없습니다.")

    detect_obstacles_button.click(
        fn=process_obstacles,
        inputs=result_image_output,
        outputs=[result_image_output, calculate_area_button, port_size_box, result_text]  # port_size_box now includes Heli button
    )

    # (2) 장애물 탐지 후 파이프라인 업데이트
    def update_pipeline_1(result_image, pipeline_updated):
        if not pipeline_updated[1]:
            pipeline_updated[1] = True
            new_arrow = "<div style='font-size:24px; color: black; text-align: center;'>↓</div>"
            return [result_image, pipeline_updated, new_arrow]
        else:
            return [gr.NO_CHANGE, gr.NO_CHANGE, gr.NO_CHANGE]

    detect_obstacles_button.click(
        fn=update_pipeline_1,
        inputs=[result_image_output, pipeline_updated],
        outputs=[pipeline_1, pipeline_updated, arrow_1_2]
    )

    # 3. 최대 원 계산 버튼 클릭 이벤트
    # 수정된 process_area 함수: D = ...을 계산하여 result_text에 표시
    def process_area(input_image, zoom_level):
        if input_image is None:
            return placeholder_image, gr.update(interactive=False), None, None, gr.update(value="")
        try:
            # 입력 이미지가 NumPy 배열이면 PIL 이미지로 변환
            if isinstance(input_image, np.ndarray):
                input_image = Image.fromarray(input_image)
            
            # 최대 원 계산 함수 호출
            result, max_diameter, max_location = find_max_circle(input_image)
            
            # Zoom level에 따른 D 계산
            if zoom_level == 16:
                diameter = max_diameter * 1.815
            elif zoom_level == 17:
                diameter = max_diameter * 0.913
            elif zoom_level == 18:
                diameter = max_diameter * 0.4603
            elif zoom_level == 19:
                diameter = max_diameter * 0.2315
            elif zoom_level == 20:
                diameter = max_diameter * 0.116
            elif zoom_level == 21:
                diameter = max_diameter * 0.0585
            else:
                return placeholder_image, gr.update(interactive=False), None, None, gr.update(value="Zoom Level을 알 수 없습니다.")

            result_text_value = f"D = {diameter:.2f}m"
            
            # 결과 텍스트 설정
            result_text_update = gr.update(value=result_text_value)
            
            # 다음 단계 버튼 활성화
            return result, gr.update(interactive=True), max_diameter, max_location, result_text_update
        except Exception as e:
            print(f"Error in process_area: {e}")
            return placeholder_image, gr.update(interactive=False), None, None, gr.update(value="결과 정보를 생성할 수 없습니다.")

    # 수정된 calculate_area_button 클릭 이벤트: zoom_level_state 추가, result_text 추가
    calculate_area_button.click(
        fn=process_area,
        inputs=[result_image_output, zoom_level_state],  # zoom_level_state 추가
        outputs=[result_image_output, result_button, max_diameter_state, max_location_state, result_text]  # result_text 추가
    )

    # (3) 최대 원 계산 후 파이프라인 업데이트
    def update_pipeline_2(result_image, pipeline_updated):
        if not pipeline_updated[2]:
            pipeline_updated[2] = True
            new_arrow = "<div style='font-size:24px; color: black; text-align: center;'>↓</div>"
            return [result_image, pipeline_updated, new_arrow]
        else:
            return [gr.NO_CHANGE, gr.NO_CHANGE, gr.NO_CHANGE]

    calculate_area_button.click(
        fn=update_pipeline_2,
        inputs=[result_image_output, pipeline_updated],
        outputs=[pipeline_2, pipeline_updated, arrow_2_3]
    )

    # 4. 결과 버튼 클릭 이벤트
    def process_text(zoom_lev, max_diameter):
        if zoom_lev == 16:
            diameter = max_diameter * 1.815
        elif zoom_lev == 17:
            diameter = max_diameter * 0.913
        elif zoom_lev == 18:
            diameter = max_diameter * 0.4603
        elif zoom_lev == 19:
            diameter = max_diameter * 0.2315
        elif zoom_lev == 20:
            diameter = max_diameter * 0.116
        elif zoom_lev == 21:
            diameter = max_diameter * 0.0585
        else:
            return "Zoom Level을 알 수 없습니다.", None, None

        result_text = f"D = {diameter:.2f}m"
        if diameter < 2 * math.sqrt(2) * 6:
            result_text += "(버티포트 설치 불가)"
            size = "impossible"
        elif diameter < 2 * math.sqrt(2) * 12:
            # result_text += "소형 eVTOL 공간 확보가 가능합니다."
            size = "small"
        elif diameter < 2 * math.sqrt(2) * 16:
            # result_text += "중형 eVTOL 공간 확보가 가능합니다."
            size = "medium"
        else:
            # result_text += "대형 eVTOL 공간 확보가 가능합니다."
            size = "large"
        if heli_detection_state.value:
            result_text = "헬리패드 사용 가능"
        return result_text, diameter, size

    def show_result(input_image, zoom_lev, max_diameter, max_location):
        if input_image is None:
            return placeholder_image, "결과 이미지를 표시할 수 없습니다.", gr.NO_CHANGE, gr.NO_CHANGE
        try:
            # zoom_lev과 max_diameter 값을 활용
            result_text_value, diameter, size = process_text(zoom_lev, max_diameter)
            result_image = draw_circle_on_original_image(intermediate_output_1.value, max_location, max_diameter // 2, diameter)
            if size != "impossible" :
                result_image = bring_evtol(result_image, max_location, max_diameter // 2)
            
            # 버튼 업데이트
            if size == "small":
                updated_box_html = generate_box_html(
                    generate_button_html('소형', 2 * math.sqrt(2) * 6, 50, active=True, color="#0073cf", text_color="white"),
                    generate_button_html('중형', 2 * math.sqrt(2) * 12, 60),
                    generate_button_html('대형', 2 * math.sqrt(2) * 16, 70),
                    generate_heli_button_html(active=heli_detection_state.value, color="#0073cf" if heli_detection_state.value else "#db7385", text_color="white" if heli_detection_state.value else "black")
                )
            elif size == "medium":
                updated_box_html = generate_box_html(
                    generate_button_html('소형', 2 * math.sqrt(2) * 6, 50),
                    generate_button_html('중형', 2 * math.sqrt(2) * 12, 60, active=True, color="#0073cf", text_color="white"),
                    generate_button_html('대형', 2 * math.sqrt(2) * 16, 70),
                    generate_heli_button_html(active=heli_detection_state.value, color="#0073cf" if heli_detection_state.value else "#db7385", text_color="white" if heli_detection_state.value else "black")
                )
            elif size == "large":
                updated_box_html = generate_box_html(
                    generate_button_html('소형', 2 * math.sqrt(2) * 6, 50),
                    generate_button_html('중형', 2 * math.sqrt(2) * 12, 60),
                    generate_button_html('대형', 2 * math.sqrt(2) * 16, 70, active=True, color="#0073cf", text_color="white"),
                    generate_heli_button_html(active=heli_detection_state.value, color="#0073cf" if heli_detection_state.value else "#db7385", text_color="white" if heli_detection_state.value else "black")
                )
            else:
                updated_box_html = generate_box_html(
                    generate_button_html('소형', 2 * math.sqrt(2) * 6, 50),
                    generate_button_html('중형', 2 * math.sqrt(2) * 12, 60),
                    generate_button_html('대형', 2 * math.sqrt(2) * 16, 70),
                    generate_heli_button_html(active=heli_detection_state.value, color="#0073cf" if heli_detection_state.value else "#db7385", text_color="white" if heli_detection_state.value else "black")
                )
            
            # 헬리패드 버튼 업데이트 is now included in updated_box_html

            # DB 업데이트
            update_db(building_id_state.value, size, diameter, heli_detection_state.value)

            return result_image, result_text_value, updated_box_html
        except Exception as e:
            print(f"Error in show_result: {e}")
            return placeholder_image, "결과 정보를 생성할 수 없습니다.", gr.NO_CHANGE, gr.NO_CHANGE

    result_button.click(
        fn=show_result,
        inputs=[result_image_output, zoom_level_state, max_diameter_state, max_location_state],
        outputs=[result_image_output, result_text, port_size_box]  # port_size_box now includes Heli button
    )

    # (4) 결과 후 파이프라인 업데이트
    def update_pipeline_3(result_image, pipeline_updated):
        if not pipeline_updated[3]:
            pipeline_updated[3] = True
            return [result_image, pipeline_updated]
        else:
            return [gr.NO_CHANGE, gr.NO_CHANGE]

    result_button.click(
        fn=update_pipeline_3,
        inputs=[result_image_output, pipeline_updated],
        outputs=[pipeline_3, pipeline_updated]
    )

    # 초기화 함수
    def initialize():
        reset_arrows = [
            "<div style='font-size:20px; color: lightgray; text-align: center;'>↓</div>",
            "<div style='font-size:20px; color: lightgray; text-align: center;'>↓</div>",
            "<div style='font-size:20px; color: lightgray; text-align: center;'>↓</div>"
        ]
        reset_buttons_html = generate_box_html(
            generate_button_html('소형', 2 * math.sqrt(2) * 6, 50),
            generate_button_html('중형', 2 * math.sqrt(2) * 12, 60),
            generate_button_html('대형', 2 * math.sqrt(2) * 16, 70),
            generate_heli_button_html()  # Include Heli button reset
        )
        return [
            "",  # 주소 초기화
            gr.update(interactive=False), gr.update(interactive=False),
            gr.update(interactive=False),  # 버튼들 초기화
            gr.update(interactive=False),  # segment_building_button
            placeholder_image,  # 결과 이미지 초기화
            None,  # bbox_image_state 초기화
            None,  # zoom_level_state 초기화
            None, None, None, None,  # 파이프라인 이미지 초기화
            *reset_arrows,  # 화살표 초기화
            [False, False, False, False],  # 파이프라인 업데이트 여부 초기화
            reset_buttons_html,  # 버튼들 초기화 (gr.HTML) including Heli button
            "",  # 결과 텍스트 초기화
        ]

    demo.load(
        fn=initialize,
        inputs=None,
        outputs=[
            address_placeholder,
            detect_obstacles_button, calculate_area_button, result_button,
            segment_building_button,
            result_image_output,
            bbox_image_state,
            zoom_level_state,
            pipeline_0, pipeline_1, pipeline_2, pipeline_3,
            arrow_0_1, arrow_1_2, arrow_2_3,
            pipeline_updated,
            port_size_box,  # 버튼들 초기화 (gr.HTML) including Heli button
            # heli_size_box,  # Removed
            result_text
        ]
    )

# 앱 실행
if __name__ == "__main__":
    demo.launch()
