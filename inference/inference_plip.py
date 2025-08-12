
\
import os
import json
from collections import defaultdict
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from chromadb import PersistentClient

# =========================
# ✅ 기본 설정
# =========================
root_dir = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/test_set_v0.1.0"
db_path = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.3.0"
collection_name = "tile_embeddings_plip"

top_k = 3                     # ← 여기서 3 또는 5 등으로 조절 (top-K 이웃)
# vote_mode = "majority"
vote_mode = "weighted"        # "majority" or "weighted"
output_path = "predictions_v0.3.4.json"

# =========================
# ✅ 모델, DB 초기화 (PLIP)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("vinid/plip").to(device)
processor = CLIPProcessor.from_pretrained("vinid/plip")

client = PersistentClient(path=db_path)
collection = client.get_or_create_collection(name=collection_name)

# =========================
# ✅ 유틸: 임베딩 추출
# =========================
@torch.inference_mode()
def image_embedding(pil_img):
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze().cpu().tolist()  # list[float]

# =========================
# ✅ 슬라이드 전체 처리
# =========================
results = []
slide_dirs = sorted([
    os.path.join(root_dir, d)
    for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
])

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

    # 투표 점수 누적(슬라이드 단위)
    vote_scores = defaultdict(float)

    for path in tile_paths:
        try:
            img = Image.open(path).convert("RGB")
            emb = image_embedding(img)

            q = collection.query(
                query_embeddings=[emb],
                n_results=top_k,                    # ← 상위 K
                include=["metadatas", "distances"]  # distances 포함 (weighted용)
            )

            metas = q.get("metadatas", [[]])[0]
            dists = q.get("distances", [[]])[0]

            # 이 타일의 K개 이웃 모두 반영
            for m, d in zip(metas, dists):
                caption = (m or {}).get("caption", "(없음)")
                if vote_mode == "weighted":
                    # cosine distance 가정 → 유사도 근사치로 (1 - d) 사용, 음수 방지
                    w = max(0.0, 1.0 - float(d))
                    vote_scores[caption] += w
                else:
                    # majority: 동일 가중치 1
                    vote_scores[caption] += 1.0

        except Exception as e:
            print(f"❌ 오류 발생: {path} → {e}")

    if not vote_scores:
        print(f"⚠️ 캡션 없음: {slide_id}")
        continue

    # 최종 캡션 선택
    final_caption = max(vote_scores.items(), key=lambda x: x[1])[0]
    final_score = vote_scores[final_caption]

    print(f"\n✅ 최종 병리 리포트: {slide_id} (점수: {final_score:.2f}, mode={vote_mode}, K={top_k})")
    print(f"📄 {final_caption}")

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

