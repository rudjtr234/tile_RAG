import os
import json
import math
import statistics as stats
from eval import REG_Evaluator

# =========================
# ✅ 경로 설정
# =========================
MAJ_PATH = "/home/mts/ssd_16tb/member/jks/reg2025_tile_RAG/evaluation/reg/metric/json/predictions_final_weighted_v0.1.1.json"
WEI_PATH = "/home/mts/ssd_16tb/member/jks/reg2025_tile_RAG/evaluation/reg/metric/json/predictions_final_majority.json"

# =========================
# ✅ 유사도 모델
# =========================
EMBEDDING_MODEL = "dmis-lab/biobert-v1.1"
SPACY_MODEL = "en_core_sci_lg"

# =========================
# ✅ evaluator 초기화
# =========================
evaluator = REG_Evaluator(embedding_model=EMBEDDING_MODEL, spacy_model=SPACY_MODEL)

# =========================
# ✅ JSON 로드
# =========================
with open(MAJ_PATH, "r") as f:
    maj_data = json.load(f)
with open(WEI_PATH, "r") as f:
    wei_data = json.load(f)

maj_dict = {item["id"]: item["report"] for item in maj_data}
wei_dict = {item["id"]: item["report"] for item in wei_data}

# 공통 ID만 사용
common_ids = sorted(set(maj_dict.keys()) & set(wei_dict.keys()))

# =========================
# ✅ majority vs weighted 유사도 측정
# =========================
pairs = [(maj_dict[sid], wei_dict[sid]) for sid in common_ids]
score_mean = evaluator.evaluate_dummy(pairs)

# per-slide 점수 계산
per_scores = []
changed_ids = []
for sid in common_ids:
    s = evaluator.evaluate_dummy([(maj_dict[sid], wei_dict[sid])])
    per_scores.append(s)
    if maj_dict[sid] != wei_dict[sid]:
        changed_ids.append((sid, s))

# 통계 함수
def pct(arr, p):
    if not arr: return float("nan")
    k = (len(arr)-1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c: return arr[int(k)]
    return arr[f] + (arr[c]-arr[f]) * (k - f)

# =========================
# ✅ 결과 출력
# =========================
print(f"\n비교 대상 슬라이드 수: {len(common_ids):,}")
print(f"평균 유사도: {score_mean:.4f}")
print(f"중앙값: {stats.median(per_scores):.4f} | "
      f"P50: {pct(sorted(per_scores),0.50):.4f} | "
      f"P75: {pct(sorted(per_scores),0.75):.4f} | "
      f"P90: {pct(sorted(per_scores),0.90):.4f} | "
      f"P95: {pct(sorted(per_scores),0.95):.4f} | "
      f"P99: {pct(sorted(per_scores),0.99):.4f}")

same_cnt = len(common_ids) - len(changed_ids)
print(f"동일 캡션 수: {same_cnt} | 상이 캡션 수: {len(changed_ids)}")

# 예시 5개만 출력
if changed_ids:
    print("\n상이한 예시 5개 (slide_id, 유사도):")
    for sid, s in sorted(changed_ids, key=lambda x: x[1])[:5]:
        print(f" - {sid}: {s:.4f}")
