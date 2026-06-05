"""
test_filters.py  -  filters.py 각 함수 자동 테스트
"""
import sys
import traceback
import numpy as np
import cv2

# ── 테스트용 이미지 생성 ─────────────────────────────────────────────────────
def make_test_img(h=300, w=300):
    """Red, Green, Blue 영역이 뚜렷한 300×300 컬러 이미지"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :w//3,      2] = 200          # 왼쪽 1/3 : Red
    img[:, w//3:2*w//3,1] = 200          # 가운데    : Green
    img[:, 2*w//3:,    0] = 200          # 오른쪽    : Blue
    # 배경에 그라데이션 추가
    for i in range(h):
        img[i, :, :] = np.clip(img[i, :, :].astype(int) + i//3, 0, 255).astype(np.uint8)
    return img

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []

def check(name, fn):
    try:
        out = fn()
        assert isinstance(out, np.ndarray), "반환값이 np.ndarray가 아님"
        assert out.dtype == np.uint8,        f"dtype={out.dtype} (uint8 아님)"
        assert out.size > 0,                 "빈 배열 반환"
        print(f"{PASS}  {name}")
        print(f"       shape={out.shape}  dtype={out.dtype}  min={out.min()}  max={out.max()}")
        results.append((name, True, None))
        return out
    except Exception as e:
        msg = traceback.format_exc()
        print(f"{FAIL}  {name}")
        print(f"       {e}")
        results.append((name, False, str(e)))
        return None

print("=" * 60)
print(" filters.py 테스트 시작")
print("=" * 60)

from filters import (
    apply_cartoon_filter,
    apply_sketch_filter,
    apply_color_highlight_filter,
    apply_cctv_filter,
)

img = make_test_img()
print(f"\n테스트 이미지: shape={img.shape}  dtype={img.dtype}\n")

# ── 1. 카툰 ─────────────────────────────────────────────────────────────────
print("── 1. apply_cartoon_filter ─────────────────")
check("cartoon / 정상 입력",      lambda: apply_cartoon_filter(img))
check("cartoon / None 입력",      lambda: apply_cartoon_filter(None))
check("cartoon / 1채널 이미지",   lambda: apply_cartoon_filter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

# ── 2. 스케치 ────────────────────────────────────────────────────────────────
print("\n── 2. apply_sketch_filter ──────────────────")
check("sketch / ksize=21",        lambda: apply_sketch_filter(img, ksize=21))
check("sketch / ksize=3(최소)",   lambda: apply_sketch_filter(img, ksize=3))
check("sketch / ksize=51(최대)",  lambda: apply_sketch_filter(img, ksize=51))
check("sketch / ksize=20(짝수)",  lambda: apply_sketch_filter(img, ksize=20))   # 홀수 보정 확인
check("sketch / None 입력",       lambda: apply_sketch_filter(None, ksize=21))

# ── 3. 색상 강조 ──────────────────────────────────────────────────────────────
print("\n── 3. apply_color_highlight_filter ─────────")
check("highlight / red",          lambda: apply_color_highlight_filter(img, "red"))
check("highlight / green",        lambda: apply_color_highlight_filter(img, "green"))
check("highlight / blue",         lambda: apply_color_highlight_filter(img, "blue"))
check("highlight / RED(대문자)",   lambda: apply_color_highlight_filter(img, "RED"))
check("highlight / 잘못된 색상",   lambda: apply_color_highlight_filter(img, "yellow"))
check("highlight / None 입력",    lambda: apply_color_highlight_filter(None, "red"))

# ── 4. CCTV ─────────────────────────────────────────────────────────────────
print("\n── 4. apply_cctv_filter ────────────────────")
check("cctv / noise=30",          lambda: apply_cctv_filter(img, noise_level=30))
check("cctv / noise=0",           lambda: apply_cctv_filter(img, noise_level=0))
check("cctv / noise=120",         lambda: apply_cctv_filter(img, noise_level=120))
check("cctv / noise=-5(음수)",    lambda: apply_cctv_filter(img, noise_level=-5))
check("cctv / noise=999(초과)",   lambda: apply_cctv_filter(img, noise_level=999))
check("cctv / None 입력",         lambda: apply_cctv_filter(None, noise_level=30))

# ── 결과 요약 ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f" 결과: {passed} 통과 / {failed} 실패  (총 {len(results)}개)")
print("=" * 60)

if failed:
    print("\n실패 목록:")
    for name, ok, err in results:
        if not ok:
            print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("\n모든 테스트 통과!")
    sys.exit(0)
