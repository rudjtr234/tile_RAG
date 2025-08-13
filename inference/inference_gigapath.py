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
db_path  = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.6.0"

# ⚠️ GigaPath 임베딩으로 구축된 컬렉션명으로 바꾸세요
collection_name = "tile_embeddings_gigapath"

top_k = 1
vote_mode = "majority"        # "majority" or "weighted"
output_path = "predictions_v0.6.0_gigapath.json"

# =========================
# ✅ 모델, 전처리 (GigaPath / timm)
# =========================
import timm
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GigaPath 타일 인코더 로드 (HF 동의 및 토큰 필요할 수 있음)
model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device)
model.eval()

# 권장 전처리
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# =========================
# ✅ ChromaDB 초기화
# =========================
client = PersistentClient(path=db_path)
collection = client.get_or_create_collection(name=collection_name)
# ⚠️ 컬렉션의 metric은 cosine 권장.
# ⚠️ 반드시 DB도 GigaPath 임베딩으로 구축되어 있어야 유효한 검색이 됩니다.

# =========================
# ✅ 유틸: 임베딩 추출 (GigaPath)
# =========================
@torch.inference_mode()
def image_embedding(pil_img: Image.Image):
    x = transform(pil_img).unsqueeze(0).to(device, non_blocking=True)   # (1, C, H, W)
    feat = model(x)                        # 보통 (1, 1536)
    if isinstance(feat, (list, tuple)):
        feat = feat[0]
    if feat.dim() > 2:                     # 안전 장치: 만약 spatial이 있으면 평균
        feat = feat.mean(dim=(2, 3))
    feat = torch.nn.functional.normalize(feat, dim=-1)  # L2 정규화
    return feat.squeeze(0).cpu().tolist()

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

            for m, d in zip(metas, dists):
                caption = (m or {}).get("caption", "(없음)")
                if vote_mode == "weighted":
                    w = max(0.0, 1.0 - float(d))  # cosine distance → 유사도로 변환
                    vote_scores[caption] += w
                else:
                    vote_scores[caption] += 1.0

        except Exception as e:
            print(f"❌ 오류 발생: {path} → {e}")

    if not vote_scores:
        print(f"⚠️ 캡션 없음: {slide_id}")
        continue

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
