import os
import json
from collections import defaultdict
from PIL import Image
import torch
import numpy as np
from chromadb import PersistentClient

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# =========================
# ✅ 기본 설정
# =========================
root_dir = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/test_set_v0.1.0"
db_path  = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.4.0"  # UNI2 임베딩 DB
collection_name = "tile_embeddings_UNI2"

top_k = 3                    # ← 이웃 개수 (바로 반영)
vote_mode = "majority"       # "majority" 또는 "weighted"
output_path = "predictions_v0.4.2.json"

# weighted 모드 세부 옵션
USE_SOFTMAX = True           # softmax로 정규화
SOFTMAX_T   = 0.1            # softmax 온도 (작을수록 상위 이웃에 더 가중)

# =========================
# ✅ 디바이스
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# ✅ UNI2-h 로드 (timm)
# =========================
timm_kwargs = {
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "mlp_ratio": 2.66667*2,
    "num_classes": 0,           # 특징 벡터 반환
    "no_embed_class": True,
    "mlp_layer": timm.layers.SwiGLUPacked,
    "act_layer": torch.nn.SiLU,
    "reg_tokens": 8,
    "dynamic_img_size": True,
}

model = timm.create_model(
    "hf-hub:MahmoodLab/UNI2-h",
    pretrained=True,
    **timm_kwargs,
).to(device)
model.eval()

# 권장 전처리
data_cfg = resolve_data_config({}, model=model)
transform = create_transform(**data_cfg)

# =========================
# ✅ ChromaDB
# =========================
client = PersistentClient(path=db_path)
collection = client.get_or_create_collection(name=collection_name)

# =========================
# ✅ 유틸
# =========================
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def softmax(x, T=1.0, axis=-1, eps=1e-12):
    x = np.array(x, dtype=np.float64)
    x = x / max(T, eps)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + eps)

# =========================
# ✅ 전체 처리
# =========================
results = []
slide_dirs = sorted(
    [os.path.join(root_dir, d) for d in os.listdir(root_dir)
     if os.path.isdir(os.path.join(root_dir, d))]
)

use_fp16 = (device.type == "cuda")

for slide_dir in slide_dirs:
    slide_id = os.path.basename(slide_dir)
    tile_paths = sorted([os.path.join(slide_dir, f) for f in os.listdir(slide_dir)
                         if f.lower().endswith(".jpg")])

    if not tile_paths:
        print(f"⚠️ 타일 없음: {slide_id} → 스킵")
        continue

    # 슬라이드 내 캡션 누적 카운터 (문자열 → 점수)
    caption_scores = defaultdict(float)
    total_votes = 0.0

    for path in tile_paths:
        try:
            img = Image.open(path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                if use_fp16:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        feats = model(img_t)          # (1, 1536)
                else:
                    feats = model(img_t)

            feats = l2_normalize(feats, dim=-1)        # L2 정규화
            embedding = feats.squeeze(0).detach().cpu().tolist()

            q = collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["metadatas", "distances"]
            )

            # 검색 결과 파싱
            if not q["metadatas"] or not q["metadatas"][0]:
                continue

            metas = q["metadatas"][0]                 # 길이 = top_k
            dists = q.get("distances", [[]])[0]       # 길이 = top_k (cosine distance: 작을수록 유사)
            # 유사도로 변환 (cosine similarity = 1 - cosine distance)
            sims  = [1.0 - float(d) for d in dists]

            if vote_mode == "majority":
                # top_k 이웃 각각 1표씩
                for m in metas:
                    cap = m.get("caption", "(없음)")
                    caption_scores[cap] += 1.0
                    total_votes += 1.0

            elif vote_mode == "weighted":
                if USE_SOFTMAX:
                    weights = softmax(sims, T=SOFTMAX_T)
                else:
                    # 선형 정규화(합이 1이 되도록)
                    s = np.array(sims, dtype=np.float64)
                    s = np.clip(s, 0.0, None)
                    denom = s.sum() if s.sum() > 0 else 1.0
                    weights = s / denom

                for m, w in zip(metas, weights):
                    cap = m.get("caption", "(없음)")
                    caption_scores[cap] += float(w)
                    total_votes += float(w)

            else:
                raise ValueError(f"지원하지 않는 vote_mode: {vote_mode}")

        except Exception as e:
            print(f"❌ 오류 발생: {path} → {e}")

    if not caption_scores:
        print(f"⚠️ 캡션 없음: {slide_id}")
        continue

    # 최종 캡션 선택
    final_caption = max(caption_scores.items(), key=lambda x: x[1])[0]
    final_score   = caption_scores[final_caption]

    print(f"\n✅ 최종 병리 리포트: {slide_id}  |  mode={vote_mode}, top_k={top_k}")
    print(f"📄 {final_caption}")
    print(f"🧮 누적 점수: {final_score:.4f} (총 가중 표 수: {total_votes:.4f})")

    results.append({
        "id": f"{slide_id}.tiff",
        "report": final_caption
    })

# =========================
# ✅ JSON 저장
# =========================
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n📁 전체 결과 JSON 저장 완료: {output_path}")
