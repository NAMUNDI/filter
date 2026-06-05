"""
app.py
------
Streamlit 기반 이미지 필터 웹 애플리케이션.
filters.py 모듈의 필터 함수를 연동하여 업로드한 이미지에
실시간으로 다양한 효과를 적용하고 원본과 결과를 나란히 표시한다.

실행 방법:
    streamlit run app.py
"""

import io
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# filters.py 모듈에서 필터 함수 임포트
from filters import (
    apply_cartoon_filter,
    apply_cctv_filter,
    apply_color_highlight_filter,
    apply_sketch_filter,
)

# ---------------------------------------------------------------------------
# 페이지 기본 설정
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🎨 Image Filter Studio",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 세션 상태 초기화
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

# ---------------------------------------------------------------------------
# 커스텀 CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* 전체 배경 */
        .stApp { background-color: #0f1117; }

        /* 사이드바 배경 */
        section[data-testid="stSidebar"] {
            background: linear-gradient(160deg, #1a1d2e 0%, #16192b 100%);
            border-right: 1px solid #2d2f45;
        }

        /* 메인 타이틀 */
        .main-title {
            text-align: center;
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .main-subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 0.95rem;
            margin-bottom: 1.8rem;
        }

        /* 이미지 컨테이너 카드 */
        .img-card {
            background: #1e2130;
            border: 1px solid #2d3250;
            border-radius: 14px;
            padding: 1rem;
            text-align: center;
        }
        .img-label {
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }
        .label-original { color: #60a5fa; }
        .label-filtered  { color: #a78bfa; }

        /* 필터 정보 배지 */
        .filter-badge {
            display: inline-block;
            background: #2d2f45;
            color: #a78bfa;
            border-radius: 20px;
            padding: 0.2rem 0.8rem;
            font-size: 0.8rem;
            font-weight: 600;
            margin-top: 0.4rem;
        }

        /* 슬라이더 라벨 */
        .param-label {
            color: #9ca3af;
            font-size: 0.82rem;
            margin-bottom: 0.1rem;
        }

        /* 구분선 */
        hr { border-color: #2d3250 !important; }

        /* 업로드 박스 */
        [data-testid="stFileUploader"] {
            border: 1.5px dashed #4b5563;
            border-radius: 10px;
            padding: 0.5rem;
        }

        /* 다운로드 버튼 */
        .stDownloadButton > button {
            width: 100%;
            background: linear-gradient(90deg, #6d28d9, #4f46e5);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem;
            transition: opacity 0.2s;
        }
        .stDownloadButton > button:hover { opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 유틸리티 함수
# ---------------------------------------------------------------------------

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """
    PIL 이미지를 OpenCV BGR ndarray로 변환한다.
    RGBA(투명도 채널 포함) PNG 예외 처리를 포함한다.

    Args:
        pil_img: PIL.Image 객체

    Returns:
        np.ndarray: BGR 형식의 uint8 배열
    """
    try:
        # RGBA → 흰 배경에 알파 합성 후 RGB로 변환
        if pil_img.mode == "RGBA":
            background = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
            # 알파 채널을 이용해 흰 배경 위에 합성
            merged = Image.alpha_composite(background, pil_img)
            pil_img = merged.convert("RGB")
        elif pil_img.mode != "RGB":
            # L, P, CMYK 등 그 외 모드 → RGB
            pil_img = pil_img.convert("RGB")

        rgb_array = np.array(pil_img, dtype=np.uint8)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array

    except Exception as e:
        st.error(f"이미지 변환 오류: {e}")
        return np.zeros((100, 100, 3), dtype=np.uint8)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """OpenCV BGR ndarray → Streamlit 표시용 RGB ndarray 변환."""
    if bgr.ndim == 2:
        # 그레이스케일 결과는 3채널로 확장
        return cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def ndarray_to_png_bytes(arr: np.ndarray) -> bytes:
    """np.ndarray → PNG 바이트열 변환 (다운로드용)."""
    rgb = bgr_to_rgb(arr)
    pil_out = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 필터 메타데이터 정의
# ---------------------------------------------------------------------------
FILTER_OPTIONS = {
    "🎨 카툰 (Cartoon)": "cartoon",
    "✏️ 스케치 (Sketch)": "sketch",
    "🔴 색상 강조 — 빨강 (Color Highlight: Red)": "highlight_red",
    "🟢 색상 강조 — 초록 (Color Highlight: Green)": "highlight_green",
    "🔵 색상 강조 — 파랑 (Color Highlight: Blue)": "highlight_blue",
    "📹 CCTV": "cctv",
}

FILTER_DESC = {
    "cartoon":        "bilateralFilter로 면 단순화 + adaptiveThreshold 외곽선 합성",
    "sketch":         "그레이 변환 → 반전 → 가우시안 블러 → cv2.divide 스케치",
    "highlight_red":  "HSV 색 공간에서 빨강만 컬러로 남기고 나머지 흑백 처리",
    "highlight_green":"HSV 색 공간에서 초록만 컬러로 남기고 나머지 흑백 처리",
    "highlight_blue": "HSV 색 공간에서 파랑만 컬러로 남기고 나머지 흑백 처리",
    "cctv":           "흑백 + 노이즈 + 스캔라인 + 가우시안 비네팅 효과",
}


def apply_selected_filter(bgr: np.ndarray, filter_key: str, params: dict) -> np.ndarray:
    """
    선택된 필터 키와 파라미터로 필터를 적용하고 결과를 반환한다.

    Args:
        bgr        : BGR 형식 입력 이미지
        filter_key : FILTER_OPTIONS value (예: "sketch")
        params     : 필터별 파라미터 딕셔너리

    Returns:
        np.ndarray: 필터가 적용된 이미지
    """
    try:
        if filter_key == "cartoon":
            return apply_cartoon_filter(bgr)

        elif filter_key == "sketch":
            ksize = params.get("ksize", 21)
            return apply_sketch_filter(bgr, ksize=ksize)

        elif filter_key == "highlight_red":
            return apply_color_highlight_filter(bgr, target_color="red")

        elif filter_key == "highlight_green":
            return apply_color_highlight_filter(bgr, target_color="green")

        elif filter_key == "highlight_blue":
            return apply_color_highlight_filter(bgr, target_color="blue")

        elif filter_key == "cctv":
            noise_level = params.get("noise_level", 30)
            return apply_cctv_filter(bgr, noise_level=noise_level)

        else:
            st.warning(f"알 수 없는 필터 키: {filter_key}")
            return bgr

    except Exception as e:
        st.error(f"필터 적용 중 오류가 발생했습니다: {e}")
        return bgr


# ===========================================================================
# 사이드바 UI
# ===========================================================================
with st.sidebar:
    st.markdown("## 🖼️ Image Filter Studio")
    st.markdown("---")

    # ── 작업 모드 선택 ───────────────────────────────────────────────────────
    st.markdown("### 🛠️ 작업 모드 선택")
    app_mode = st.radio(
        label="작업 모드",
        options=["📷 이미지 파일 업로드", "🎥 실시간 웹캠 스트리밍"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    # ── 이미지 업로드 (업로드 모드인 경우만 표시) ──────────────────────────────
    uploaded_file = None
    if app_mode == "📷 이미지 파일 업로드":
        st.markdown("### 📂 이미지 업로드")
        st.caption("JPG, JPEG, PNG, BMP, WEBP 지원 · PNG의 투명(RGBA) 채널은 자동으로 흰 배경에 합성됩니다.")

        uploaded_file = st.file_uploader(
            label="이미지 파일을 여기에 드래그하거나 클릭하여 선택하세요",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )
        st.markdown("---")

    # ── 필터 선택 ────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ 필터 선택")
    selected_label = st.selectbox(
        label="적용할 필터",
        options=list(FILTER_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    selected_key = FILTER_OPTIONS[selected_label]

    # 필터 설명 표시
    st.markdown(
        f'<div class="filter-badge">💡 {FILTER_DESC[selected_key]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── 필터별 파라미터 슬라이더 (동적 표시) ─────────────────────────────────
    params: dict = {}

    if selected_key == "sketch":
        st.markdown("### 🎛️ 파라미터")
        st.markdown(
            '<p class="param-label">블러 커널 크기 (ksize) — 클수록 부드럽고 밝은 스케치</p>',
            unsafe_allow_html=True,
        )
        ksize_val = st.slider(
            label="블러 커널 크기",
            min_value=3,
            max_value=51,
            value=21,
            step=2,           # 홀수만 허용
            label_visibility="collapsed",
        )
        # 홀수 보정 (step=2 + min=3이면 항상 홀수이지만 방어 처리)
        if ksize_val % 2 == 0:
            ksize_val += 1
        params["ksize"] = ksize_val
        st.caption(f"현재 값: **{ksize_val}**  (범위: 3 ~ 51, 홀수)")

    elif selected_key == "cctv":
        st.markdown("### 🎛️ 파라미터")
        st.markdown(
            '<p class="param-label">노이즈 강도 (noise_level) — 클수록 입자 잡음이 강해짐</p>',
            unsafe_allow_html=True,
        )
        noise_val = st.slider(
            label="노이즈 강도",
            min_value=0,
            max_value=120,
            value=30,
            step=5,
            label_visibility="collapsed",
        )
        params["noise_level"] = noise_val
        st.caption(f"현재 값: **{noise_val}**  (범위: 0 ~ 120)")

    elif selected_key in ("highlight_red", "highlight_green", "highlight_blue"):
        # 색상 강조 필터는 추가 파라미터 없음 — 안내 메시지 표시
        st.markdown("### ℹ️ 파라미터")
        color_name = {"highlight_red": "빨강 🔴", "highlight_green": "초록 🟢", "highlight_blue": "파랑 🔵"}
        st.info(f"**{color_name[selected_key]}** 색상만 원본 컬러로 유지하고\n나머지는 흑백 처리합니다.\n\n별도 파라미터가 없습니다.")

    else:
        # 카툰 필터
        st.markdown("### ℹ️ 파라미터")
        st.info("카툰 필터는 별도 파라미터 없이\nbilateralFilter × 4회 반복으로 동작합니다.")

    # ── 웹캠 모드 장치 설정 추가 ─────────────────────────────────────────────
    cam_index = 0
    if app_mode == "🎥 실시간 웹캠 스트리밍":
        st.markdown("---")
        st.markdown("### 🎥 웹캠 장치 설정")
        cam_index = st.number_input(
            label="카메라 장치 번호 (Device Index)",
            min_value=0,
            max_value=5,
            value=0,
            step=1,
            help="노트북 기본 웹캠은 0입니다. 모니터 외장 카메라 등을 사용하실 경우 1, 2 등으로 올려서 시도해 보세요."
        )

    st.markdown("---")
    st.caption("📌 filters.py 모듈 연동 · OpenCV + NumPy")


# ===========================================================================
# 메인 화면
# ===========================================================================
st.markdown('<h1 class="main-title">🎨 Image Filter Studio</h1>', unsafe_allow_html=True)

if app_mode == "📷 이미지 파일 업로드":
    st.markdown(
        '<p class="main-subtitle">이미지를 업로드하고 필터를 선택하면 실시간으로 효과를 미리볼 수 있습니다.</p>',
        unsafe_allow_html=True,
    )

    if uploaded_file is None:
        # ── 업로드 전 안내 화면 ──────────────────────────────────────────────────
        st.markdown("---")
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.markdown(
                """
                <div style='text-align:center; padding: 3rem 1rem;
                            background:#1e2130; border-radius:16px;
                            border: 1.5px dashed #374151;'>
                    <div style='font-size:4rem;'>📂</div>
                    <div style='font-size:1.15rem; color:#9ca3af; margin-top:0.8rem;'>
                        왼쪽 사이드바에서 이미지를 업로드해주세요.
                    </div>
                    <div style='font-size:0.85rem; color:#4b5563; margin-top:0.5rem;'>
                        JPG · JPEG · PNG · BMP · WEBP
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        # ── 이미지 처리 ──────────────────────────────────────────────────────────
        try:
            pil_img = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"이미지를 열 수 없습니다: {e}")
            st.stop()

        # 이미지 정보 표시
        mode_info = f"모드: {pil_img.mode}  |  크기: {pil_img.width} × {pil_img.height} px"
        if pil_img.mode == "RGBA":
            mode_info += "  |  ⚠️ 투명 채널(Alpha) 감지 — 흰 배경으로 자동 변환"
        st.caption(mode_info)

        # PIL → BGR 변환 (RGBA 예외 처리 포함)
        bgr_img = pil_to_bgr(pil_img)

        # 선택된 필터 적용
        with st.spinner("✨ 필터를 적용하는 중..."):
            filtered_bgr = apply_selected_filter(bgr_img, selected_key, params)

        # BGR → RGB (Streamlit 표시용)
        original_rgb  = bgr_to_rgb(bgr_img)
        filtered_rgb  = bgr_to_rgb(filtered_bgr)

        # ── 2-컬럼 레이아웃: 원본 | 결과 ────────────────────────────────────────
        col_orig, col_filt = st.columns(2, gap="large")

        with col_orig:
            st.markdown(
                '<div class="img-label label-original">📷 원본 이미지</div>',
                unsafe_allow_html=True,
            )
            st.image(original_rgb, use_container_width=True)

        with col_filt:
            st.markdown(
                f'<div class="img-label label-filtered">🎨 필터 결과 — {selected_label.split("(")[0].strip()}</div>',
                unsafe_allow_html=True,
            )
            st.image(filtered_rgb, use_container_width=True)

        # ── 다운로드 버튼 ─────────────────────────────────────────────────────────
        st.markdown("---")
        dl_col1, dl_col2, dl_col3 = st.columns([1, 1.2, 1])
        with dl_col2:
            png_bytes = ndarray_to_png_bytes(filtered_bgr)
            safe_name = selected_key.replace(" ", "_")
            st.download_button(
                label="⬇️  결과 이미지 다운로드 (PNG)",
                data=png_bytes,
                file_name=f"filtered_{safe_name}.png",
                mime="image/png",
                use_container_width=True,
            )

else:
    # ── 실시간 웹캠 스트리밍 모드 ──────────────────────────────────────────────
    st.markdown(
        '<p class="main-subtitle">웹캠 카메라를 켜고 필터를 선택하면 실시간 필터 영상을 감상할 수 있습니다.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # 레이아웃 구성: 좌측 설명/상태, 우측 웹캠 스트림 화면
    col_info, col_feed = st.columns([1, 2], gap="large")

    with col_info:
        st.markdown(
            """
            <div class="img-card" style="margin-bottom: 1rem;">
                <div class="img-label label-original" style="font-size: 1.1rem; text-align: left;">🎥 카메라 제어 센터</div>
                <hr style="margin: 0.5rem 0;">
                <p style="font-size: 0.9rem; color: #9ca3af; text-align: left; margin-bottom: 1rem;">
                    [카메라 켜기] 버튼을 눌러 로컬 웹캠 스트림을 활성화합니다. 사이드바의 필터 및 조절 슬라이더로 파라미터를 실시간 변경할 수 있습니다.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ON / OFF 버튼
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("▶️ 카메라 켜기", use_container_width=True, type="primary"):
                st.session_state.webcam_running = True
                st.rerun()
        with btn_col2:
            if st.button("⏹️ 카메라 끄기", use_container_width=True):
                st.session_state.webcam_running = False
                st.rerun()

        # 현재 카메라 상태 표시
        if st.session_state.webcam_running:
            st.success("● 실시간 스트리밍 중 (LIVE)")
        else:
            st.info("○ 카메라 꺼짐 (OFF)")

        st.markdown(
            f"""
            <div style="background: #1a1d2e; border: 1px solid #2d2f45; border-radius: 10px; padding: 0.8rem; font-size: 0.82rem; color: #9ca3af;">
                💡 <b>작동 안내</b><br>
                - 카메라가 켜지면 선택된 카메라 장치 번호(현재: {cam_index}번)를 로드합니다.<br>
                - 다른 애플리케이션(Zoom, Discord 등)이 카메라를 점유 중일 경우 연결되지 않을 수 있습니다.<br>
                - 화면이 나오지 않거나 에러 발생 시, 사이드바에서 장치 번호를 1, 2 등으로 변경해 보세요.<br>
                - 모드를 변경하거나 [카메라 끄기]를 클릭하면 리소스가 안전하게 릴리즈됩니다.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_feed:
        st.markdown(
            f'<div class="img-label label-filtered">🎨 실시간 필터 스트림 — {selected_label}</div>',
            unsafe_allow_html=True,
        )
        frame_placeholder = st.empty()

        if st.session_state.webcam_running:
            # 윈도우 환경 대응: CAP_DSHOW 먼저 시도
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_index)

            if not cap.isOpened():
                st.error(f"⚠️ 웹캠 카메라 장치 {cam_index}번을 열 수 없습니다. 다른 장치 번호(1, 2 등)로 변경하여 시도해 보세요.")
                st.session_state.webcam_running = False
                st.rerun()
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                try:
                    # 루프
                    while st.session_state.webcam_running:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            # 연속 실패 시 잠시 대기
                            time.sleep(0.05)
                            continue

                        # 필터 적용
                        filtered_frame = apply_selected_filter(frame, selected_key, params)

                        # BGR -> RGB 변환
                        rgb_frame = bgr_to_rgb(filtered_frame)

                        # 프레임 출력
                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        # CPU 과점유 방지 및 약 30FPS 타겟팅을 위한 대기
                        time.sleep(0.03)
                except Exception as e:
                    st.error(f"스트리밍 중 에러가 발생했습니다: {e}")
                finally:
                    cap.release()
                    frame_placeholder.empty()
                    st.session_state.webcam_running = False
                    st.rerun()
        else:
            # 꺼진 상태 안내 화면
            frame_placeholder.markdown(
                """
                <div style="display: flex; flex-direction: column; justify-content: center; align-items: center;
                            height: 350px; background: #1a1d2e; border: 1px dashed #374151; border-radius: 14px;">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">📷</div>
                    <div style="color: #6b7280; font-size: 1rem;">카메라가 꺼진 상태입니다.</div>
                    <div style="color: #4b5563; font-size: 0.8rem; margin-top: 0.3rem;">[▶️ 카메라 켜기] 버튼을 누르면 시작됩니다.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
