import os
import json
from collections import defaultdict
from PIL import Image
import torch
from chromadb import PersistentClient

# =========================
# ✅ 기본 설정
# =========================
root_dir = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/test_set_v0.1.0"
db_path  = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.7.0"  # ← Optimus 임베딩이 들어있는 DB 권장
collection_name = "tile_embeddings_HOPT1"  # ← H-optimus-1 임베딩 컬렉션

top_k = 1                    # 3 또는 5 등으로 조절 (top-K 이웃)
vote_mode = "majority"        # "majority" or "weighted"
output_path = "predictions_v0.7.0.json"

# =========================
# ✅ 모델, 전처리 (H-optimus-1 / timm)
# =========================
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# H-optimus-1 로드
model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True).to(device)
model.eval()

# 권장 전처리 구성 (모델 카드 기준 자동 세팅)
cfg = resolve_data_config({}, model=model)
transform = create_transform(**cfg)

# =========================
# ✅ ChromaDB 초기화
# =========================
client = PersistentClient(path=db_path)
collection = client.get_or_create_collection(name=collection_name)
# 컬렉션 metric이 'cosine'인지 확인(권장). Optimus 임베딩 차원과 일치해야 함.

# =========================
# ✅ 유틸: 임베딩 추출 (H-optimus-1)
#   - timm ViT 계열 호환: forward_features → forward_head(pre_logits=True)
#   - 일부 버전/설정에서 dict/tuple 반환 가능 → 안전 폴백 포함
# =========================
@torch.inference_mode()
def image_embedding(pil_img: Image.Image):
    x = transform(pil_img).unsqueeze(0).to(device)   # (1, C, H, W)

    try:
        feats = model.forward_features(x)                # (1, D, …) 또는 (1, D)
        feats = model.forward_head(feats, pre_logits=True)  # (1, D)
    except Exception:
        # 일부 timm 버전/구성에서 바로 임베딩이 반환되도록 정의되어 있을 수 있음
        feats = model(x)
        # (배치, D) 보장 위해 flatten
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if hasattr(feats, "logits"):
            feats = feats.logits

    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.squeeze(0).cpu().tolist()           # list[float]

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
