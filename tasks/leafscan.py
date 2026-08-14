"""tasks/leafscan.py — LeafScan 과제 정의.

라벨 체계(정본 순서)·폴더↔작물 매핑·목표값·프리셋을 소유한다.
데이터에 실제로 존재하는 클래스만 head 로 만들되, 순서는 이 정본을 따른다.
→ 상추 한 작물만 있어도(crop head=1) 4작물이 들어와도 같은 코드가 동작한다.
"""

# 정본 순서 (모델 번들 heads 순서의 기준)
CANON_CROPS = ["상추", "케일", "겨자채", "근대"]
CANON_STAGES = ["정식기", "생육기", "수확기"]

# 폴더명(영문) ↔ 작물(한글). 04_데이터_명세서 §1.
FOLDER_TO_CROP = {
    "lettuce": "상추",
    "chard": "근대",
    "kale": "케일",
    "leaf_mustard": "겨자채",
}
CROP_TO_FOLDER = {v: k for k, v in FOLDER_TO_CROP.items()}

# 생육단계 주 지표 대상 (소수 클래스)
PRIMARY_STAGE = "정식기"

# 목표값 (리포트의 목표 델타 표시에 사용)
TARGETS = {
    "stage_macro_f1": 0.80,
    "stage_recall_정식기": 0.65,
    "crop_accuracy": 0.94,
}


def order_present(values, canon):
    """values 중 canon 순서로 정렬한 리스트 (canon 밖 값은 뒤에 정렬해 붙임)."""
    present = set(values)
    ordered = [c for c in canon if c in present]
    extra = sorted(v for v in present if v not in canon)
    return ordered + extra
