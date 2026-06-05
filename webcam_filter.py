"""
webcam_filter.py
----------------
cv2.VideoCapture를 사용해 웹캠 영상을 실시간으로 입력받고,
filters.py에 정의된 필터를 즉시 적용하여 OpenCV 창에 출력한다.

조작 방법 (OpenCV 창에 포커스가 있어야 함):
    [SPACE]     카메라 ON / OFF 토글
    [R]         카메라 재연결 시도
    [1]         카툰 필터
    [2]         스케치 필터
    [3]         색상 강조 - 빨강
    [4]         색상 강조 - 초록
    [5]         색상 강조 - 파랑
    [6]         CCTV 필터
    [0]         필터 없음 (원본)
    [UP] / [DN] 파라미터 증가 / 감소 (스케치·CCTV 전용)
    [Q] / [ESC] 종료

실행 방법:
    python webcam_filter.py
    python webcam_filter.py --device 1       # 웹캠 인덱스 지정
    python webcam_filter.py --width 1280 --height 720
"""

import argparse
import sys
import time
from typing import Optional

import cv2
import numpy as np

# filters.py에서 필터 함수 임포트
try:
    from filters import (
        apply_cartoon_filter,
        apply_cctv_filter,
        apply_color_highlight_filter,
        apply_sketch_filter,
    )
except ImportError as e:
    print(f"[ERROR] filters.py를 불러올 수 없습니다: {e}")
    print("        webcam_filter.py와 같은 디렉터리에 filters.py가 있는지 확인하세요.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 상수 및 필터 메타데이터
# ---------------------------------------------------------------------------

# 키 코드 (cv2.waitKey 반환값)
KEY_0     = ord("0")
KEY_1     = ord("1")
KEY_2     = ord("2")
KEY_3     = ord("3")
KEY_4     = ord("4")
KEY_5     = ord("5")
KEY_6     = ord("6")
KEY_Q     = ord("q")
KEY_R     = ord("r")       # 재연결
KEY_SPACE = ord(" ")       # 카메라 ON/OFF 토글
KEY_ESC   = 27
KEY_UP    = 82             # ↑ 방향키 (Windows)
KEY_DN    = 84             # ↓ 방향키 (Windows)

# 필터 키 매핑 (표시용 이름, 내부 키)
FILTER_MAP = {
    KEY_0: ("[0] No Filter",          "none"),
    KEY_1: ("[1] Cartoon",            "cartoon"),
    KEY_2: ("[2] Sketch",             "sketch"),
    KEY_3: ("[3] Color - Red",        "highlight_red"),
    KEY_4: ("[4] Color - Green",      "highlight_green"),
    KEY_5: ("[5] Color - Blue",       "highlight_blue"),
    KEY_6: ("[6] CCTV",               "cctv"),
}

# 파라미터 범위 정의
PARAM_CONFIG = {
    "sketch": {
        "label":   "ksize",
        "default": 21,
        "min":     3,
        "max":     51,
        "step":    2,
    },
    "cctv": {
        "label":   "noise",
        "default": 30,
        "min":     0,
        "max":     120,
        "step":    5,
    },
}

# HUD 색상
COLOR_WHITE   = (255, 255, 255)
COLOR_YELLOW  = (0,   220, 255)
COLOR_GREEN   = (80,  220, 100)
COLOR_GRAY    = (160, 160, 160)
COLOR_RED     = (60,  60,  220)
COLOR_ORANGE  = (30,  140, 255)
COLOR_BG      = (20,  20,   30)


# ---------------------------------------------------------------------------
# HUD (Heads-Up Display) 렌더링
# ---------------------------------------------------------------------------

def draw_hud(frame: np.ndarray, filter_name: str,
             param_val: Optional[int], param_label: Optional[str],
             fps: float, cam_active: bool, paused: bool) -> np.ndarray:
    """
    프레임 위에 반투명 HUD 패널을 그린다.
    cam_active : 카메라 ON 여부
    paused     : 마지막 프레임을 고정 표시 중 여부
    """
    overlay = frame.copy()
    h, w    = frame.shape[:2]
    alpha   = 0.65

    # ── 상단 HUD 패널 ────────────────────────────────────────────────────────
    cv2.rectangle(overlay, (0, 0), (w, 72), COLOR_BG, -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # 카메라 상태 배지 (● LIVE / ■ PAUSED)
    if cam_active and not paused:
        badge_color = (40, 200, 40)
        badge_text  = "● LIVE"
    else:
        badge_color = COLOR_ORANGE
        badge_text  = "■ PAUSED"
    cv2.putText(frame, badge_text,
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, badge_color, 2, cv2.LINE_AA)

    # 필터 이름
    cv2.putText(frame, f"Filter: {filter_name}",
                (110, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, COLOR_YELLOW, 2, cv2.LINE_AA)

    # 파라미터
    if param_val is not None and param_label:
        param_text = f"{param_label}: {param_val}  (UP/DN to adjust)"
        cv2.putText(frame, param_text,
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, COLOR_GREEN, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No extra params",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, COLOR_GRAY, 1, cv2.LINE_AA)

    # CAM ON/OFF 마우스 클릭 버튼 그리기 (우상단 FPS 왼편)
    # 버튼 영역: x: w - 230 ~ w - 120, y: 12 ~ 42
    btn_x1, btn_y1 = w - 230, 12
    btn_x2, btn_y2 = w - 120, 42
    btn_color = (40, 180, 40) if cam_active and not paused else COLOR_ORANGE
    btn_text = "CAM OFF" if cam_active and not paused else "CAM ON"
    
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_color, -1, cv2.LINE_AA)
    # 버튼 테두리
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), COLOR_WHITE, 1, cv2.LINE_AA)
    
    # 텍스트 중앙 맞춤
    (tw3, th3), _ = cv2.getTextSize(btn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    bx = btn_x1 + (btn_x2 - btn_x1 - tw3) // 2
    by = btn_y1 + (btn_y2 - btn_y1 + th3) // 2
    cv2.putText(frame, btn_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)

    # FPS (우상단)
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
    cv2.putText(frame, fps_text,
                (w - tw - 10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, COLOR_WHITE, 1, cv2.LINE_AA)

    # ── 하단 조작 안내 ─────────────────────────────────────────────────────
    hint = "[SPACE]/[Mouse Click] Cam ON/OFF  [R] Reconnect  [1-6] Filter  [UP/DN] Param  [Q] Quit"
    (tw2, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.rectangle(overlay, (0, h - 24), (w, h), COLOR_BG, -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.putText(frame, hint,
                ((w - tw2) // 2, h - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, COLOR_GRAY, 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# 필터 적용 래퍼
# ---------------------------------------------------------------------------

def apply_filter(frame: np.ndarray, filter_key: str, param: int) -> np.ndarray:
    """
    filter_key에 따라 알맞은 필터 함수를 호출한다.
    예외 발생 시 원본을 반환하여 스트림을 유지한다.
    """
    try:
        if filter_key == "none":
            return frame
        elif filter_key == "cartoon":
            return apply_cartoon_filter(frame)
        elif filter_key == "sketch":
            result = apply_sketch_filter(frame, ksize=param)
            if result.ndim == 2:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            return result
        elif filter_key == "highlight_red":
            return apply_color_highlight_filter(frame, target_color="red")
        elif filter_key == "highlight_green":
            return apply_color_highlight_filter(frame, target_color="green")
        elif filter_key == "highlight_blue":
            return apply_color_highlight_filter(frame, target_color="blue")
        elif filter_key == "cctv":
            return apply_cctv_filter(frame, noise_level=param)
        else:
            return frame
    except Exception as e:
        print(f"[WARN] apply_filter({filter_key}) error: {e}")
        return frame


# ---------------------------------------------------------------------------
# 웹캠 열기 및 설정
# ---------------------------------------------------------------------------

def open_camera(device_index: int, width: int, height: int,
                retry: int = 3) -> Optional[cv2.VideoCapture]:
    """웹캠을 열고 해상도를 설정한다. 실패 시 retry 횟수만큼 재시도."""
    for attempt in range(1, retry + 1):
        print(f"[INFO] Camera connect attempt {attempt}/{retry}  (device={device_index})")

        # Windows: CAP_DSHOW 먼저 시도
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(device_index)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[INFO] Camera connected  {aw}x{ah}")
            return cap

        print(f"[WARN] Cannot open camera. Retrying in 1s...")
        time.sleep(1.0)

    print("[ERROR] All connection attempts failed.")
    return None


# ---------------------------------------------------------------------------
# 에러/일시정지 화면 생성
# ---------------------------------------------------------------------------

def make_status_frame(width: int = 640, height: int = 480,
                      title: str = "No Camera",
                      subtitle: str = "Press [SPACE] to toggle  /  [R] to reconnect  /  [Q] to quit",
                      icon: str = "!",
                      icon_color: tuple = (60, 60, 220)) -> np.ndarray:
    """카메라 없음 / 일시정지 상태를 나타내는 화면을 생성한다."""
    frame = np.full((height, width, 3), fill_value=COLOR_BG, dtype=np.uint8)
    cx, cy = width // 2, height // 2 - 30

    cv2.circle(frame, (cx, cy), 55, icon_color, 4)
    (iw, _), _ = cv2.getTextSize(icon, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)
    cv2.putText(frame, icon, (cx - iw // 2, cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, icon_color, 4, cv2.LINE_AA)

    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.putText(frame, title, ((width - tw) // 2, cy + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_WHITE, 2, cv2.LINE_AA)

    (sw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    cv2.putText(frame, subtitle, ((width - sw) // 2, cy + 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_GRAY, 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------------------------

def run(device_index: int = 0, width: int = 640, height: int = 480) -> None:  # noqa: C901
    """
    웹캠 필터 실시간 루프.

    상태 기계:
        cam_active=True  : 카메라 ON  → 매 프레임 읽어 필터 적용
        cam_active=False : 카메라 OFF → last_frame 고정 표시
        cap=None         : 연결 실패  → 에러 화면 표시

    키 조작:
        [SPACE] : 카메라 ON↔OFF 토글 (OFF 시 cap.release() 호출)
        [R]     : 카메라 강제 재연결
        [1-6]   : 필터 전환
        [UP/DN] : 파라미터 조절
        [Q/ESC] : 종료
    """
    cap         : Optional[cv2.VideoCapture] = None
    window_name  = "Webcam Filter Studio  [SPACE: cam toggle  Q: quit]"

    # ── 상태 변수 ─────────────────────────────────────────────────────────────
    cam_active          = True          # 카메라 ON 여부 (토글 가능)
    last_frame          = None          # 마지막으로 성공적으로 읽은 프레임
    current_filter_key  = "none"
    current_filter_name = "[0] No Filter"
    current_param       = 21
    read_fail_count     = 0
    MAX_READ_FAIL       = 30

    # FPS 측정
    fps         = 0.0
    frame_count = 0
    fps_timer   = time.perf_counter()

    try:
        # ── 창 생성 ────────────────────────────────────────────────────────────
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

        # ── 마우스 이벤트 콜백 정의 및 등록 ─────────────────────────────────
        def on_mouse(event, x, y, flags, param):
            nonlocal cam_active, cap, read_fail_count, device_index
            if event == cv2.EVENT_LBUTTONDOWN:
                # 버튼 영역: x: width - 230 ~ width - 120, y: 12 ~ 42
                btn_x1, btn_y1 = width - 230, 12
                btn_x2, btn_y2 = width - 120, 42
                if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
                    cam_active = not cam_active
                    if not cam_active:
                        if cap is not None:
                            cap.release()
                            cap = None
                        print("[INFO] Camera OFF (via Mouse click)")
                    else:
                        print(f"[INFO] Camera ON - reconnecting to device {device_index} (via Mouse click)...")
                        cap = open_camera(device_index, width, height, retry=3)
                        if cap is None:
                            print(f"[WARN] Reconnect to device {device_index} failed. Staying in OFF state.")
                            cam_active = False
                        else:
                            read_fail_count = 0

        cv2.setMouseCallback(window_name, on_mouse)

        # ── 초기 카메라 연결 ──────────────────────────────────────────────────
        cap = open_camera(device_index, width, height, retry=3)

        if cap is None:
            cam_active = False
            print("[ERROR] Camera not found. Showing error screen.")
            print("        Press [R] to retry connection, [Q] to quit.")

        print("[INFO] Ready.  Window keys:")
        print("       [SPACE] Cam ON/OFF  [R] Reconnect  [1-6] Filter  [UP/DN] Param  [Q] Quit")

        # ── 메인 루프 ──────────────────────────────────────────────────────────
        while True:

            # ── 키 입력 (1ms 대기) ────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            # 창 닫힘(X 버튼) 감지
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Window closed.")
                    break
            except cv2.error:
                break

            # ── 종료 키 ───────────────────────────────────────────────────────
            if key in (KEY_Q, KEY_ESC):
                print("[INFO] Quit key pressed.")
                break

            # ── [SPACE] 카메라 ON / OFF 토글 ─────────────────────────────────
            if key == KEY_SPACE:
                cam_active = not cam_active
                if not cam_active:
                    # OFF: 카메라 해제
                    if cap is not None:
                        cap.release()
                        cap = None
                    print("[INFO] Camera OFF (paused on last frame)")
                else:
                    # ON: 카메라 재연결 시도
                    print("[INFO] Camera ON - reconnecting...")
                    cap = open_camera(device_index, width, height, retry=3)
                    if cap is None:
                        print("[WARN] Reconnect failed. Staying in OFF state.")
                        cam_active = False
                    else:
                        read_fail_count = 0
                continue  # 이번 이터레이션은 화면 갱신만

            # ── [R] 강제 재연결 ────────────────────────────────────────────────
            if key == KEY_R:
                print("[INFO] [R] Force reconnect...")
                if cap is not None:
                    cap.release()
                    cap = None
                cap = open_camera(device_index, width, height, retry=3)
                if cap is not None:
                    cam_active = True
                    read_fail_count = 0
                    print("[INFO] Reconnected successfully.")
                else:
                    cam_active = False
                    print("[WARN] Reconnect failed.")
                continue

            # ── 필터 전환 키 (0 ~ 6) ─────────────────────────────────────────
            if key in FILTER_MAP:
                current_filter_name, current_filter_key = FILTER_MAP[key]
                if current_filter_key in PARAM_CONFIG:
                    current_param = PARAM_CONFIG[current_filter_key]["default"]
                print(f"[INFO] Filter: {current_filter_name}")

            # ── 파라미터 또는 카메라 장치 인덱스 조절 키 ───────────────────────
            if cap is None or not cam_active:
                # 카메라가 오프라인일 때는 방향키로 연결할 장치 인덱스(Device Index) 변경
                if key == KEY_UP:
                    device_index = min(device_index + 1, 5)
                    print(f"[INFO] Changed target camera device index to: {device_index} (Press [R] to reconnect to this device)")
                elif key == KEY_DN:
                    device_index = max(device_index - 1, 0)
                    print(f"[INFO] Changed target camera device index to: {device_index} (Press [R] to reconnect to this device)")
            else:
                # 정상 스트리밍 중에는 필터 파라미터 조절
                if current_filter_key in PARAM_CONFIG:
                    cfg = PARAM_CONFIG[current_filter_key]
                    if key == KEY_UP:
                        current_param = min(current_param + cfg["step"], cfg["max"])
                    elif key == KEY_DN:
                        current_param = max(current_param - cfg["step"], cfg["min"])
                    if current_filter_key == "sketch" and current_param % 2 == 0:
                        current_param += 1

            # ── 카메라 OFF 상태: 마지막 프레임 고정 표시 ──────────────────────
            if not cam_active or cap is None:
                if last_frame is not None:
                    # 마지막 프레임에 필터 적용 후 PAUSED 표시
                    frozen = apply_filter(last_frame, current_filter_key, current_param)
                    param_val = current_param if current_filter_key in PARAM_CONFIG else None
                    p_label   = PARAM_CONFIG[current_filter_key]["label"] \
                                if current_filter_key in PARAM_CONFIG else None
                    display = draw_hud(frozen, current_filter_name,
                                       param_val, p_label, fps,
                                       cam_active=False, paused=True)
                else:
                    # 아직 프레임이 없음 → 에러 화면
                    display = make_status_frame(
                        width, height,
                        title=f"Camera Offline (Target Device: {device_index})",
                        subtitle="[UP/DN] Change Device Index  [R] Reconnect  [Q] Quit",
                        icon="||", icon_color=COLOR_ORANGE
                    )
                cv2.imshow(window_name, display)
                time.sleep(0.033)
                continue

            # ── 카메라 ON 상태: 프레임 읽기 ───────────────────────────────────
            ret, frame = cap.read()

            if not ret or frame is None:
                read_fail_count += 1
                if read_fail_count >= MAX_READ_FAIL:
                    print(f"[WARN] {MAX_READ_FAIL} consecutive read failures. Auto-reconnecting...")
                    cap.release()
                    cap = None
                    # 에러 화면 잠시 표시
                    err_disp = make_status_frame(
                        width, height,
                        title="Connection Lost - Reconnecting...",
                        subtitle="[R] Manual reconnect  [Q] Quit"
                    )
                    cv2.imshow(window_name, err_disp)
                    cv2.waitKey(1)
                    time.sleep(2.0)

                    cap = open_camera(device_index, width, height, retry=3)
                    if cap is not None:
                        read_fail_count = 0
                        print("[INFO] Auto-reconnect successful.")
                    else:
                        print("[ERROR] Auto-reconnect failed. Camera set to OFF.")
                        cam_active = False
                time.sleep(0.033)
                continue

            # 정상 프레임
            read_fail_count = 0
            last_frame = frame.copy()   # 항상 마지막 성공 프레임 저장

            # ── 필터 적용 ─────────────────────────────────────────────────────
            filtered = apply_filter(frame, current_filter_key, current_param)

            # ── FPS 계산 ──────────────────────────────────────────────────────
            frame_count += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                fps         = frame_count / elapsed
                frame_count = 0
                fps_timer   = time.perf_counter()

            # ── HUD 오버레이 ──────────────────────────────────────────────────
            param_val = current_param if current_filter_key in PARAM_CONFIG else None
            p_label   = PARAM_CONFIG[current_filter_key]["label"] \
                        if current_filter_key in PARAM_CONFIG else None
            display = draw_hud(filtered, current_filter_name,
                               param_val, p_label, fps,
                               cam_active=True, paused=False)

            cv2.imshow(window_name, display)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    except Exception as e:
        import traceback
        print(f"[ERROR] Unexpected error: {e}")
        traceback.print_exc()

    finally:
        # ── 반드시 실행: 리소스 정리 ──────────────────────────────────────────
        if cap is not None:
            try:
                cap.release()
                print("[INFO] cap.release() done.")
            except Exception as e:
                print(f"[WARN] cap.release() error: {e}")
        try:
            cv2.destroyAllWindows()
            print("[INFO] All windows closed.")
        except Exception as e:
            print(f"[WARN] destroyAllWindows() error: {e}")


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime Webcam Filter (filters.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Keys:\n"
            "  [SPACE]   Toggle camera ON/OFF\n"
            "  [R]       Force reconnect camera\n"
            "  [0]       No filter (original)\n"
            "  [1]       Cartoon\n"
            "  [2]       Sketch\n"
            "  [3]       Color highlight - Red\n"
            "  [4]       Color highlight - Green\n"
            "  [5]       Color highlight - Blue\n"
            "  [6]       CCTV\n"
            "  [Up/Down] Adjust parameter (Sketch / CCTV only)\n"
            "  [Q/ESC]   Quit\n"
        ),
    )
    parser.add_argument("--device", type=int, default=0,
                        help="Webcam device index (default: 0)")
    parser.add_argument("--width",  type=int, default=640,
                        help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Capture height (default: 480)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(device_index=args.device, width=args.width, height=args.height)
