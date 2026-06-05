# 🎨 Image Filter Studio

OpenCV와 NumPy로 구현한 이미지 필터 모듈에, **Streamlit 웹앱**과 **실시간 웹캠 필터** 애플리케이션을 결합한 프로젝트입니다.

---

## 📁 프로젝트 구조

```
filter/
├── filters.py          # 핵심 필터 알고리즘 모듈
├── app.py              # Streamlit 이미지 업로드 웹앱
├── webcam_filter.py    # 실시간 웹캠 필터 (OpenCV 창)
├── requirements.txt    # 의존성 패키지 목록
└── test_filters.py     # 필터 함수 자동 테스트 (20개 케이스)
```

---

## ✨ 필터 종류

| 필터 | 함수 | 설명 |
|------|------|------|
| 🎨 카툰 | `apply_cartoon_filter` | `bilateralFilter` × 4회 면 단순화 + `adaptiveThreshold` 외곽선 합성 |
| ✏️ 스케치 | `apply_sketch_filter` | 그레이 반전 블러 → `cv2.divide` 연필 효과, ksize 조절 가능 |
| 🔴🟢🔵 색상 강조 | `apply_color_highlight_filter` | HSV에서 특정 색(R/G/B)만 컬러, 나머지 흑백 |
| 📹 CCTV | `apply_cctv_filter` | 흑백 + 노이즈 + 해상도 비례 굵은 스캔라인 + 녹색 틴트 + 가우시안 비네팅 + **CCTV OSD 오버레이 (REC 깜빡임, 타임스탬프, CAM 01, 해상도)** |

---

## ⚙️ 설치

### 1. 저장소 클론

```bash
git clone https://github.com/nyj78/filter.git
cd filter
```

### 2. 가상환경 생성 (권장)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

> ⚠️ **주의** : `opencv-python-headless`가 아닌 반드시 `opencv-python`을 설치해야 합니다.  
> headless 버전은 GUI 창(`cv2.imshow`)을 지원하지 않습니다.
>
> ```bash
> pip uninstall opencv-python-headless
> pip install opencv-python
> ```

---

## 🚀 실행 방법

### 1. Streamlit 웹앱 — `app.py`

이미지 파일 업로드 또는 **실시간 웹캠 스트리밍**을 브라우저 상에서 동작시키며 필터를 적용하고 다운로드할 수 있습니다.

```bash
streamlit run app.py
```

브라우저가 자동으로 열리며 `http://localhost:8501` 로 접속됩니다.

**사용 방법 (작업 모드 선택):**

* **📷 이미지 파일 업로드 모드:**
  1. 왼쪽 사이드바에서 이미지 파일 업로드 (JPG / PNG / BMP / WEBP)
  2. 필터 드롭다운에서 원하는 효과 선택
  3. 파라미터 슬라이더 조절 (스케치 · CCTV 전용)
  4. 메인 화면에서 원본(좌) / 결과(우) 실시간 비교
  5. **결과 이미지 다운로드 (PNG)** 버튼으로 저장
* **🎥 실시간 웹캠 스트리밍 모드:**
  1. 메인 화면의 제어 센터에서 **[▶️ 카메라 켜기]** 버튼 클릭
  2. 로컬 웹캠에서 읽어오는 프레임에 필터가 적용되어 브라우저 화면에 실시간 스트리밍
  3. 사이드바 조절기로 실시간 효과 변환 및 CCTV OSD의 시간 정보 1초 주기 동적 업데이트 확인
  4. 사용 중지 시 **[⏹️ 카메라 끄기]** 버튼을 클릭하여 장치 리소스 해제

> 📌 PNG 파일의 투명(RGBA) 채널은 자동으로 흰 배경에 합성됩니다.

---

### 2. 실시간 웹캠 필터 — `webcam_filter.py`

OpenCV 창에서 웹캠 영상에 필터를 실시간으로 적용합니다.

```bash
# 기본 실행 (웹캠 인덱스 0, 해상도 640×480)
python webcam_filter.py

# 웹캠 인덱스 지정 (두 번째 카메라 사용 시)
python webcam_filter.py --device 1

# 해상도 지정
python webcam_filter.py --width 1280 --height 720

# 도움말
python webcam_filter.py --help
```

**조작 방법** (OpenCV 창에 포커스를 두고 조작):

* **마우스 조작:** HUD 우상단의 **`CAM ON` / `CAM OFF` 버튼 영역**을 마우스 왼쪽 버튼으로 직접 클릭하여 카메라 전원을 간편하게 켜고 끌 수 있습니다.
* **키보드 단축키:**

| 키 | 동작 |
|----|------|
| `SPACE` / `마우스 클릭` | 📷 카메라 **ON / OFF 토글** — OFF 시 마지막 프레임 고정 (cap.release() 안전 호출) |
| `R` | 🔄 카메라 **강제 재연결** 시도 |
| `0` | 원본 (필터 없음) |
| `1` | 🎨 카툰 필터 |
| `2` | ✏️ 스케치 필터 |
| `3` | 🔴 색상 강조 — 빨강 |
| `4` | 🟢 색상 강조 — 초록 |
| `5` | 🔵 색상 강조 — 파랑 |
| `6` | 📹 CCTV 필터 |
| `↑` / `↓` | 파라미터 증가 / 감소 (스케치·CCTV 전용) |
| `Q` / `ESC` | 종료 |

**HUD 상태 표시:**

| 표시 | 의미 |
|------|------|
| `● LIVE` (초록) | 카메라 정상 수신 중 |
| `■ PAUSED` (주황) | 카메라 OFF — 마지막 프레임 고정 표시 |
| 에러 화면 | 카메라 연결 실패 또는 자동 재연결 중 |

**카메라 오류 자동 처리 흐름:**

```
연결 실패 → 3회 재시도 → 에러 화면 표시
                                   ↓
              [SPACE] 또는 [R] 누르면 재연결 시도
프레임 읽기 30회 연속 실패 → 자동 재연결 후 복구
```

---

### 3. 필터 모듈 직접 사용 — `filters.py`

```python
import cv2
from filters import (
    apply_cartoon_filter,
    apply_sketch_filter,
    apply_color_highlight_filter,
    apply_cctv_filter,
)

img = cv2.imread("photo.jpg")

cartoon  = apply_cartoon_filter(img)
sketch   = apply_sketch_filter(img, ksize=21)          # ksize: 3~51 홀수
red_hl   = apply_color_highlight_filter(img, "red")    # "red" | "green" | "blue"
cctv     = apply_cctv_filter(img, noise_level=30)      # noise: 0~120

cv2.imwrite("cartoon.png", cartoon)
```

모듈 내장 테스트 실행:

```bash
python filters.py              # 테스트 이미지 자동 생성 후 output_*.png 저장
python filters.py photo.jpg    # 직접 이미지 지정
```

---

### 4. 자동 테스트 — `test_filters.py`

모든 필터 함수의 정상 입력 / 경계값 / 오류 입력(None)을 자동으로 검증합니다.

```bash
python test_filters.py
```

```
결과: 20 통과 / 0 실패  (총 20개)
모든 테스트 통과!
```

---

## 🛠️ 개발 환경

| 항목 | 최소 버전 |
|------|-----------|
| Python | 3.9 이상 |
| opencv-python | 4.8.0 이상 |
| numpy | 1.24.0 이상 |
| streamlit | 1.28.0 이상 |
| Pillow | 10.0.0 이상 |

---

## ❓ 자주 묻는 문제

**Q. `cv2.imshow`가 동작하지 않아요.**  
A. `opencv-python-headless`가 설치된 경우입니다. 아래 명령으로 교체하세요.
```bash
pip uninstall opencv-python-headless
pip install opencv-python
```

**Q. 웹캠이 인식되지 않아요.**  
A. `--device` 옵션으로 다른 인덱스를 시도하거나, 창에서 `[R]` 키를 눌러 재연결하세요.
```bash
python webcam_filter.py --device 1   # 두 번째 카메라
python webcam_filter.py --device 2   # 세 번째 카메라
```

**Q. 웹캠을 잠깐 끄고 싶어요.**  
A. OpenCV 창에 포커스를 두고 `[SPACE]`를 누르면 카메라가 OFF되고 마지막 프레임이 고정됩니다. 다시 `[SPACE]`를 누르면 복구됩니다.

**Q. PNG 투명 배경이 검게 나와요.**  
A. `app.py`는 RGBA를 흰 배경으로 자동 합성합니다. `filters.py` 직접 사용 시 아래처럼 전처리하세요.
```python
from PIL import Image
import numpy as np, cv2

pil = Image.open("transparent.png").convert("RGBA")
bg  = Image.new("RGBA", pil.size, (255, 255, 255, 255))
img = np.array(Image.alpha_composite(bg, pil).convert("RGB"))
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
```

**Q. CCTV 필터 효과가 눈에 안 띄어요.**  
A. 최신 버전에서 해상도 비례 굵은 스캔라인을 채택하고, 실시간 OSD 오버레이(상태 깜빡임, 타임스탬프, 해상도)를 더해 CCTV 연출 효과가 대폭 향상되었습니다. 슬라이더의 `noise_level`을 조절하여 노이즈 강도를 극대화할 수도 있습니다.

---

## 📊 결과 보고서

프로젝트 구현 상세 설계 및 OpenCV 알고리즘의 세부 설명, 테스트 그라데이션 이미지를 기반으로 한 원본 대 필터별 결과 대조군은 [결과 보고서 (report.md)](report.md)에서 확인하실 수 있습니다.

---

## 📄 라이선스

MIT License
