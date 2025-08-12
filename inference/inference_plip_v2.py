import os
import json
from collections import defaultdict
from typing import List, Dict, Any

from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from chromadb import PersistentClient


# =========================
# ✅ 기본 설정 (필요 시 수정)
# =========================
root_dir = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/test_set_v0.1.0"
db_path = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.3.0"
collection_name = "tile_embeddings_plip"

top_k = 3                        # 상위 K 이웃
vote_mode = "weighted"           # "weighted" or "majority"
output_path = "predictions_v0.3.5.json"

# 🔧 튜닝 파라미터
SIM_THRESHOLD = 0.28             # 유사도 τ (0~1). 낮은 후보 컷. None이면 비활성
USE_SOFTMAX   = True             # weighted에서 softmax 가중치 사용
SOFTMAX_T     = 0.1              # softmax 온도(T). 작을수록 상위 후보 집중
LOG_CONFIDENCE = True            # 슬라이드별 confidence 로깅


# =========================
# ✅ 모델, DB 초기화 (PLIP)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("vinid/plip").to(device)
processor = CLIPProcessor.from_pretrained("vinid/plip")

client = PersistentClient(path=db_path)
collection = client.get_or_create_collection(name=collection_name)


# =========================
# ✅ 유틸: 소프트맥스 가중치
# =========================
def softmax_w(scores: List[float], T: float = 0.1) -> List[float]:
    x = np.array(scores, dtype=np.float32) / max(T, 1e-6)
    x -= x.max()
    w = np.exp(x)
    w_sum = w.sum()
    return (w / (w_sum + 1e-9)).tolist()


# =========================
# ✅ 유틸: 이미지 임베딩
# =========================
@torch.inference_mode()
def image_embedding(pil_img: Image.Image) -> List[float]:
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
    return feats.squeeze().detach().cpu().tolist()


# =========================
# ✅ 슬라이드 전체 처리
# =========================
def main() -> None:
    # 슬라이드 디렉토리 수집
    slide_dirs = sorted([
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    results: List[Dict[str, Any]] = []

    for slide_dir in slide_dirs:
        slide_id = os.path.basename(slide_dir)
        tile_paths = sorted([
            os.path.join(slide_dir, f)
            for f in os.listdir(slide_dir)
            if f.lower().endswith(".jpg")
        ])

        if not tile_paths:
            print(f"⚠️ 타일 없음: {slide_id} → 스킵")
            continue

        vote_scores = defaultdict(float)   # 캡션별 가중치 합
        slide_confidence = 0.0            # 슬라이드 신뢰도(총 가중치 합)

        for path in tile_paths:
            try:
                img = Image.open(path).convert("RGB")
                emb = image_embedding(img)

                q = collection.query(
                    query_embeddings=[emb],
                    n_results=top_k,
                    include=["metadatas", "distances"]
                )

                metas = q.get("metadatas", [[]])[0]
                dists = q.get("distances", [[]])[0]
                if not metas or not dists:
                    continue

                # distance → cosine similarity
                sims: List[float] = []
                items: List[Dict[str, Any]] = []
                for m, d in zip(metas, dists):
                    try:
                        sim = max(0.0, 1.0 - float(d))  # cosine sim (가정: distance=1-cos)
                    except Exception:
                        continue
                    if SIM_THRESHOLD is not None and sim < SIM_THRESHOLD:
                        continue
                    items.append(m or {})
                    sims.append(sim)

                if not items:
                    continue

                # 가중치 계산
                if vote_mode == "weighted":
                    if USE_SOFTMAX:
                        ws = softmax_w(sims, T=SOFTMAX_T)
                    else:
                        ws = sims  # similarity 그대로 가중치로 사용
                else:
                    ws = [1.0] * len(items)  # majority

                # 캡션별 점수 누적
                for m, w in zip(items, ws):
                    caption = m.get("caption", "(없음)")
                    vote_scores[caption] += float(w)
                    slide_confidence += float(w)

            except Exception as e:
                print(f"❌ 오류 발생: {path} → {e}")

        if not vote_scores:
            print(f"⚠️ 캡션 없음: {slide_id}")
            continue

        final_caption, final_score = max(vote_scores.items(), key=lambda x: x[1])

        # 로그 출력
        print(f"\n✅ 최종 병리 리포트: {slide_id} "
              f"(score={final_score:.3f}, conf={slide_confidence:.3f}, "
              f"mode={vote_mode}, K={top_k}, tau={SIM_THRESHOLD}, T={SOFTMAX_T if USE_SOFTMAX else None})")
        print(f"📄 {final_caption}")

        # 결과 저장 오브젝트
        obj = {
            "id": f"{slide_id}.tiff",
            "report": final_caption
        }
        if LOG_CONFIDENCE:
            obj.update({
                "confidence": round(slide_confidence, 3),
                "topk": top_k,
                "mode": vote_mode,
                "tau": SIM_THRESHOLD,
                "softmax_T": SOFTMAX_T if USE_SOFTMAX else None
            })
        results.append(obj)

    # JSON 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📁 전체 결과 JSON 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
