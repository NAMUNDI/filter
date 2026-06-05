"""
filters.py
----------
OpenCV(cv2)와 NumPy를 사용하여 구현한 이미지 필터 알고리즘 모음.

포함 함수:
    - apply_cartoon_filter      : 만화(카툰) 스타일 필터
    - apply_sketch_filter       : 연필 스케치 스타일 필터
    - apply_color_highlight_filter : 특정 색상 강조 필터
    - apply_cctv_filter         : CCTV/감시카메라 스타일 필터
"""

import cv2
import numpy as np
import datetime
import time


# ---------------------------------------------------------------------------
# 1. 만화(Cartoon) 스타일 필터
# ---------------------------------------------------------------------------

def apply_cartoon_filter(img: np.ndarray) -> np.ndarray:
    """
    입력 이미지에 만화(카툰) 스타일 필터를 적용한다.

    처리 과정:
        1. bilateralFilter로 색상 면을 부드럽게 단순화
        2. 그레이스케일 변환 후 adaptiveThreshold로 외곽선(Edge) 추출
        3. 단순화된 컬러 이미지와 외곽선 마스크를 bitwise_and로 합성

    Args:
        img (np.ndarray): BGR 형식의 입력 이미지 (H x W x 3, uint8).

    Returns:
        np.ndarray: 만화 스타일이 적용된 BGR 이미지.
                    오류 발생 시 원본 이미지를 반환한다.
    """
    try:
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("유효하지 않은 이미지 입력입니다.")

        # ── 입력이 그레이스케일(1채널)이면 BGR로 변환 ────────────────────
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # ── 1단계: 양방향 필터로 색상 면 단순화 ──────────────────────────
        # d         : 필터링에 참여하는 픽셀 이웃 지름
        # sigmaColor: 색상 공간 표준편차 (클수록 더 넓은 색상 범위를 혼합)
        # sigmaSpace: 좌표 공간 표준편차 (클수록 더 넓은 공간 범위를 혼합)
        color = img.copy()
        for _ in range(4):  # 반복 적용으로 페인팅 효과 강화
            color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)

        # ── 2단계: 외곽선(Edge) 추출 ─────────────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 메디안 블러로 노이즈 제거 후 adaptive threshold 적용
        gray_blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray_blur,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=9,   # 이웃 블록 크기 (홀수)
            C=2            # 평균에서 뺄 상수값
        )

        # ── 3단계: 외곽선(단채널)을 3채널로 변환 후 컬러와 합성 ──────────
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color, edges_colored)

        return cartoon

    except Exception as e:
        print(f"[apply_cartoon_filter] 오류 발생: {e}")
        # None 또는 복구 불가 입력: 빈 배열 반환
        if img is None or not isinstance(img, np.ndarray):
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return img


# ---------------------------------------------------------------------------
# 2. 연필 스케치(Sketch) 필터
# ---------------------------------------------------------------------------

def apply_sketch_filter(img: np.ndarray, ksize: int = 21) -> np.ndarray:
    """
    입력 이미지에 연필 스케치 스타일 필터를 적용한다.

    처리 과정:
        1. 그레이스케일 변환
        2. 그레이스케일 이미지 반전(Invert)
        3. 반전 이미지에 가우시안 블러 적용
        4. cv2.divide로 원본 그레이와 블러 이미지를 나누어 스케치 효과 생성

    Args:
        img    (np.ndarray): BGR 형식의 입력 이미지 (H x W x 3, uint8).
        ksize  (int)       : 가우시안 블러 커널 크기 (홀수, 기본값 21).
                             값이 클수록 부드럽고 밝은 스케치.

    Returns:
        np.ndarray: 스케치 효과가 적용된 그레이스케일 이미지 (H x W, uint8).
                    오류 발생 시 원본 이미지를 반환한다.
    """
    try:
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("유효하지 않은 이미지 입력입니다.")

        # ksize는 반드시 홀수여야 함
        if ksize % 2 == 0:
            ksize += 1
        if ksize < 1:
            ksize = 1

        # ── 1단계: 그레이스케일 변환 (이미 그레이면 그대로 사용) ──────────
        if img.ndim == 2:
            gray = img.copy()
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── 2단계: 그레이스케일 이미지 반전 ──────────────────────────────
        gray_inv = cv2.bitwise_not(gray)

        # ── 3단계: 반전 이미지에 가우시안 블러 ───────────────────────────
        blurred_inv = cv2.GaussianBlur(gray_inv, (ksize, ksize), sigmaX=0)

        # ── 4단계: cv2.divide를 이용한 스케치 효과 합성 ──────────────────
        # divide(a, 255 - b, scale=256.0) ≈ a / (255 - b) * 256
        sketch = cv2.divide(gray, 255 - blurred_inv, scale=256.0)

        return sketch

    except Exception as e:
        print(f"[apply_sketch_filter] 오류 발생: {e}")
        # None 또는 복구 불가 입력: 빈 배열 반환
        if img is None or not isinstance(img, np.ndarray):
            return np.zeros((1, 1), dtype=np.uint8)
        return img


# ---------------------------------------------------------------------------
# 3. 특정 색상 강조(Color Highlight) 필터
# ---------------------------------------------------------------------------

# HSV 색상 범위 정의 (OpenCV: H=0~179, S=0~255, V=0~255)
_COLOR_RANGES = {
    "red": [
        (np.array([0,   100, 80]),  np.array([10,  255, 255])),   # 낮은 빨강 (0°~10°)
        (np.array([160, 100, 80]),  np.array([179, 255, 255])),   # 높은 빨강 (160°~179°)
    ],
    "green": [
        (np.array([35,  60,  60]),  np.array([85,  255, 255])),   # 초록 (35°~85°)
    ],
    "blue": [
        (np.array([90,  60,  60]),  np.array([130, 255, 255])),   # 파랑 (90°~130°)
    ],
}


def apply_color_highlight_filter(
    img: np.ndarray,
    target_color: str = "red"
) -> np.ndarray:
    """
    입력 이미지에서 특정 색상(Red / Green / Blue)만 원본 컬러로 남기고,
    나머지 영역은 흑백(그레이스케일)으로 변환한다.

    처리 과정:
        1. BGR → HSV 색 공간 변환
        2. cv2.inRange로 목표 색상 범위에 해당하는 마스크 생성
        3. 마스크 영역은 원본 컬러, 나머지는 그레이스케일로 합성

    Args:
        img          (np.ndarray): BGR 형식의 입력 이미지 (H x W x 3, uint8).
        target_color (str)       : 강조할 색상 이름. "red" | "green" | "blue"
                                   (기본값 "red", 대소문자 무관)

    Returns:
        np.ndarray: 색상 강조 필터가 적용된 BGR 이미지.
                    오류 발생 시 원본 이미지를 반환한다.
    """
    try:
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("유효하지 않은 이미지 입력입니다.")

        color_key = target_color.lower().strip()
        if color_key not in _COLOR_RANGES:
            raise ValueError(
                f"지원하지 않는 색상입니다: '{target_color}'. "
                f"지원 색상: {list(_COLOR_RANGES.keys())}"
            )

        # ── 1단계: HSV 변환 ───────────────────────────────────────────────
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # ── 2단계: 색상 마스크 생성 ───────────────────────────────────────
        # 하나의 색상이 여러 범위를 가질 수 있음 (예: 빨강)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for lower, upper in _COLOR_RANGES[color_key]:
            mask |= cv2.inRange(hsv, lower, upper)

        # 마스크 형태 개선 (열림/닫힘 연산으로 노이즈 제거)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # ── 3단계: 마스크 기반 합성 ──────────────────────────────────────
        # 색상 영역: 원본 컬러
        color_part = cv2.bitwise_and(img, img, mask=mask)

        # 비색상 영역: 그레이스케일 → BGR로 변환 후 역마스크 적용
        gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        inv_mask  = cv2.bitwise_not(mask)
        gray_part = cv2.bitwise_and(gray_bgr, gray_bgr, mask=inv_mask)

        # 두 영역 합산
        result = cv2.add(color_part, gray_part)

        return result

    except Exception as e:
        print(f"[apply_color_highlight_filter] 오류 발생: {e}")
        # None 또는 복구 불가 입력: 빈 배열 반환
        if img is None or not isinstance(img, np.ndarray):
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return img


# ---------------------------------------------------------------------------
# 4. CCTV / 감시카메라 스타일 필터
# ---------------------------------------------------------------------------

def apply_cctv_filter(img: np.ndarray, noise_level: int = 30) -> np.ndarray:
    """
    입력 이미지에 CCTV / 감시카메라 스타일 필터를 적용한다.

    처리 과정:
        1. 흑백(그레이스케일) 변환 후 BGR 3채널로 복원
        2. cv2.randu로 랜덤 노이즈 생성 및 추가
        3. 가로 스캔라인 효과 (해상도 비례 두께 및 주기)
        4. 약한 녹색 틴트 (야간 투시 카메라 느낌)
        5. 가우시안 커널 기반 외곽 비네팅(Vignetting) 효과 적용
        6. CCTV OSD 오버레이 (REC 깜빡임, 타임스탬프, 카메라 이름 등)

    Args:
        img         (np.ndarray): BGR 형식의 입력 이미지 (H x W x 3, uint8).
        noise_level (int)       : 추가할 랜덤 노이즈 세기 (0~255, 기본값 30).

    Returns:
        np.ndarray: CCTV 스타일이 적용된 BGR 이미지.
                    오류 발생 시 원본 이미지를 반환한다.
    """
    try:
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("유효하지 않은 이미지 입력입니다.")

        noise_level = int(np.clip(noise_level, 0, 255))
        h, w = img.shape[:2]

        # ── 1단계: 흑백 변환 ─────────────────────────────────────────────
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 3채널 유지

        # ── 2단계: 단색(Monochromatic) 노이즈 추가 ───────────────────────
        if noise_level > 0:
            noise_1ch = np.zeros((h, w, 1), dtype=np.uint8)
            cv2.randu(noise_1ch, 0, noise_level)
            noise = np.repeat(noise_1ch, 3, axis=2)
            result = cv2.add(result, noise)

        # ── 3단계: 가로 스캔라인 음영 효과 ───────────────────────────────
        # 해상도 축소 시 뭉개지지 않도록 8픽셀 주기로 3픽셀씩 스캔라인 적용
        scanline_mask = np.ones((h, w, 3), dtype=np.float32)
        scanline_mask[0::8, :, :] = 0.60
        scanline_mask[1::8, :, :] = 0.60
        scanline_mask[2::8, :, :] = 0.60
        result = np.clip(
            result.astype(np.float32) * scanline_mask, 0, 255
        ).astype(np.uint8)

        # ── 4단계: 약한 녹색 틴트 (CCTV 특유의 야간 투시 색감) ────────────
        tint = result.astype(np.float32)
        tint[:, :, 1] = np.clip(tint[:, :, 1] * 1.18, 0, 255)  # Green +18%
        tint[:, :, 0] = np.clip(tint[:, :, 0] * 0.82, 0, 255)  # Blue  -18%
        result = tint.astype(np.uint8)

        # ── 5단계: 가우시안 커널 기반 비네팅 효과 ────────────────────────
        sigma_x = w / 2.5
        sigma_y = h / 2.5

        kernel_x = cv2.getGaussianKernel(w, sigma_x)
        kernel_y = cv2.getGaussianKernel(h, sigma_y)
        vignette = kernel_y @ kernel_x.T
        vignette_norm = vignette / vignette.max()

        vignette_3ch = np.dstack([vignette_norm] * 3)
        result = np.clip(
            result.astype(np.float64) * vignette_3ch, 0, 255
        ).astype(np.uint8)

        # ── 6단계: CCTV OSD 텍스트 오버레이 ─────────────────────────────
        base_size = min(w, h)
        font_scale = max(0.42, base_size / 650.0)
        thick = max(1, int(base_size / 450))
        shadow_thick = thick + 2
        
        # 1초 주기로 REC 빨간불 깜빡임 효과 (소수점 초 기준 판단)
        rec_on = int(time.time() * 2) % 2 == 0
        cam_text = "CAM 01"

        def draw_osd(text_str, x, y, color=(255, 255, 255)):
            # 검은 테두리(그림자) 효과
            cv2.putText(result, text_str, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), shadow_thick, cv2.LINE_AA)
            # 본체 흰색(혹은 연회색) 텍스트
            cv2.putText(result, text_str, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick, cv2.LINE_AA)

        # OSD 좌표 비율 계산
        margin_x = max(10, int(w * 0.04))
        margin_y = max(20, int(h * 0.08))
        circle_r = max(4, int(base_size * 0.012))

        # 좌상단 REC
        rec_x = margin_x
        rec_y = margin_y
        
        # 그림자용 검은 원
        cv2.circle(result, (rec_x, rec_y - circle_r // 2), circle_r + 1, (0, 0, 0), -1, cv2.LINE_AA)
        if rec_on:
            cv2.circle(result, (rec_x, rec_y - circle_r // 2), circle_r, (0, 0, 255), -1, cv2.LINE_AA)
            draw_osd("REC", rec_x + circle_r + 8, rec_y, color=(255, 255, 255))
        else:
            cv2.circle(result, (rec_x, rec_y - circle_r // 2), circle_r, (0, 0, 100), -1, cv2.LINE_AA)
            draw_osd("REC", rec_x + circle_r + 8, rec_y, color=(170, 170, 170))

        # 우상단 CAM 01
        cam_w, _ = cv2.getTextSize(cam_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)[0]
        draw_osd(cam_text, w - cam_w - margin_x, rec_y)

        # 우하단 현재 타임스탬프
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_w, _ = cv2.getTextSize(now_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)[0]
        draw_osd(now_str, w - time_w - margin_x, h - int(h * 0.05))

        # 좌하단 해상도 및 프레임 레이트 안내
        info_str = f"{w}x{h} 30FPS"
        draw_osd(info_str, margin_x, h - int(h * 0.05))

        return result

    except Exception as e:
        print(f"[apply_cctv_filter] 오류 발생: {e}")
        # None 또는 복구 불가 입력: 빈 배열 반환
        if img is None or not isinstance(img, np.ndarray):
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return img


# ---------------------------------------------------------------------------
# 간단한 사용 예시 (직접 실행 시)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # 테스트용 이미지 경로를 인자로 받거나, 색상 이미지를 직접 생성
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    if image_path:
        src = cv2.imread(image_path)
        if src is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            sys.exit(1)
    else:
        # 테스트용 색상 그라데이션 이미지 생성 (300 x 300)
        src = np.zeros((300, 300, 3), dtype=np.uint8)
        for i in range(300):
            src[:, i, 0] = i * 255 // 299          # Blue 채널
            src[i, :, 2] = i * 255 // 299          # Red 채널
        src[:, :, 1] = 60                          # Green 채널 고정

    filters_to_apply = [
        ("cartoon",         lambda: apply_cartoon_filter(src)),
        ("sketch",          lambda: apply_sketch_filter(src, ksize=21)),
        ("highlight_red",   lambda: apply_color_highlight_filter(src, "red")),
        ("highlight_green", lambda: apply_color_highlight_filter(src, "green")),
        ("highlight_blue",  lambda: apply_color_highlight_filter(src, "blue")),
        ("cctv",            lambda: apply_cctv_filter(src, noise_level=30)),
    ]

    for name, fn in filters_to_apply:
        out = fn()
        output_path = f"output_{name}.png"
        cv2.imwrite(output_path, out)
        print(f"[OK] {name:20s} -> {output_path}")

    print("\n모든 필터 적용 완료.")
