# compare_with_gt.py
import os, json, glob

# ===== 기본 설정 (무옵션 실행 가능) =====
GT_PATH_DEFAULT = "/home/mts/ssd_16tb/member/jks/reg2025_tile_RAG/evaluation/reg/metric/json/train.json"
PATTERN_A = "predictions_v0.4.0.json"  # UNI2
PATTERN_B = "predictions_v0.6.0.json"  # GigaPath

# 텍스트 비교 규칙
IGNORE_CASE = True
STRIP_WS    = True

# 필요하면 여기만 바꿔서 미리보기 숫자 지정 (0이면 미리보기 출력 안 함)
SHOW_MISMATCH_EXAMPLES = 0

def normalize_text(s: str) -> str:
    if s is None: return ""
    t = s
    if STRIP_WS:   t = " ".join(t.strip().split())
    if IGNORE_CASE: t = t.lower()
    return t

def pick_latest(pattern: str):
    files = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p))
    return files[-1] if files else None

def load_gt(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt = {}
    if isinstance(data, dict):
        for k, v in data.items():
            gt[str(k)] = normalize_text(v)
    else:
        for item in data:
            gt[str(item["id"])] = normalize_text(item["report"])
    return gt

def load_pred(path):
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    pred = {}
    for item in arr:
        pred[str(item["id"])] = normalize_text(item["report"])
    return pred

def main():
    gt_path = GT_PATH_DEFAULT
    a_path  = pick_latest(PATTERN_A)  # UNI2
    b_path  = pick_latest(PATTERN_B)  # GigaPath

    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"GT 없음: {gt_path}")
    if not a_path:
        raise FileNotFoundError(f"UNI2 예측 없음: {PATTERN_A}")
    if not b_path:
        raise FileNotFoundError(f"GigaPath 예측 없음: {PATTERN_B}")

    print(f"[INFO] GT      : {gt_path}")
    print(f"[INFO] UNI2    : {a_path}")
    print(f"[INFO] GigaPath: {b_path}")

    gt = load_gt(gt_path)
    pa = load_pred(a_path)
    pb = load_pred(b_path)

    ids = sorted(set(gt) & set(pa) & set(pb))
    if not ids:
        raise RuntimeError("공통 ID가 없습니다.")

    both_right = both_wrong = a_only = b_only = 0
    preview = []

    for sid in ids:
        g = gt[sid]; a = pa[sid]; b = pb[sid]
        a_ok = (a == g); b_ok = (b == g)

        if a_ok and b_ok:
            both_right += 1
        elif a_ok and not b_ok:
            a_only += 1
            if len(preview) < SHOW_MISMATCH_EXAMPLES:
                preview.append(("UNI2_only", sid, g, a, b))
        elif (not a_ok) and b_ok:
            b_only += 1
            if len(preview) < SHOW_MISMATCH_EXAMPLES:
                preview.append(("Giga_only", sid, g, a, b))
        else:
            both_wrong += 1
            if len(preview) < SHOW_MISMATCH_EXAMPLES:
                preview.append(("both_wrong", sid, g, a, b))

    n = len(ids)
    acc_a  = (both_right + a_only) / n
    acc_b  = (both_right + b_only) / n
    acc_or = (both_right + a_only + b_only) / n     # 둘 중 하나라도 맞으면 맞음(상한)
    err_overlap = both_wrong / n                     # 둘 다 틀림 비율
    complement  = (a_only + b_only) / n              # 상호보완도(한쪽만 맞음)

    print("\n=== SUMMARY ===")
    print(f"Total samples                : {n}")
    print(f"UNI2 accuracy                : {acc_a*100:.2f}%")
    print(f"GigaPath accuracy            : {acc_b*100:.2f}%")
    print(f"Both correct                 : {both_right}")
    print(f"UNI2 only correct            : {a_only}")
    print(f"GigaPath only correct        : {b_only}")
    print(f"Both wrong                   : {both_wrong}")
    print(f"Error overlap (both wrong)   : {err_overlap*100:.2f}%")
    print(f"Complementarity (xor correct): {complement*100:.2f}%")
    print(f"Late-fusion OR upper bound   : {acc_or*100:.2f}%")

    if SHOW_MISMATCH_EXAMPLES > 0 and preview:
        print(f"\n--- Mismatch preview (up to {SHOW_MISMATCH_EXAMPLES}) ---")
        for tag, sid, g, a, b in preview:
            print(f"[{tag}] id={sid}\n  GT : {g}\n  UNI2: {a}\n  Giga: {b}\n")

if __name__ == "__main__":
    main()
