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
db_path  = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.8.0"

# Virchow2 임베딩으로 구축된 컬렉션을 사용하세요.
collection_name = "tile_embeddings_virchow2"

top_k = 1                      # 3 또는 5 등으로 조절
vote_mode = "majority"         # "majority" or "weighted"
output_path = "predictions_v0.8.0.json"

# =========================
# ✅ Virchow2 로드 (로컬 경로)
# =========================
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔒 로컬 모델 디렉토리 (이미 존재함)
VIRCHOW2_LOCAL_PATH = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/virchow2"

# 로컬 폴더에 config.json + pytorch_model.bin(or model.safetensors)가 있으면 timm이 바로 로드합니다.
model = timm.create_model(VIRCHOW2_LOCAL_PATH, pretrained=True).to(device)
model.eval()

# 권장 전처리 구성 자동 해석
cfg = resolve_data_config({}, model=model)
transform = create_transform(**cfg)

# =========================
# ✅ ChromaDB 초기화
# =========================
client = PersistentClient(path=db_path)
collection = client.get_or_create_collection(name=collection_name)
# 권장: metric='cosine', 임베딩 차원은 Virchow2 임베딩과 동일해야 함.

# =========================
# ✅ 유틸: Virchow2 임베딩 추출
# =========================
@torch.inference_mode()
def image_embedding(pil_img: Image.Image):
    x = transform(pil_img).unsqueeze(0).to(device)   # (1, C, H, W)
    # timm 표준: forward_features -> backbone 출력
    feats = model.forward_features(x)                # (1, D, ...) 또는 (1, D)
    # pre_logits 특징 추출 (분류기 앞 벡터)
    feats = model.forward_head(feats, pre_logits=True)  # (1, D)
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
                n_results=top_k,
                include=["metadatas", "distances"]
            )

            metas = q.get("metadatas", [[]])[0]
            dists = q.get("distances", [[]])[0]

            # 이 타일의 K개 이웃 모두 반영
            for m, d in zip(metas, dists):
                caption = (m or {}).get("caption", "(없음)")
                if vote_mode == "weighted":
                    # cosine distance 가정 → 유사도 근사 (1 - d)
                    w = max(0.0, 1.0 - float(d))
                    vote_scores[caption] += w
                else:
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
