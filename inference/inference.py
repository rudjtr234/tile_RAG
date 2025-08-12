import os
import json
from collections import Counter
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from chromadb import PersistentClient

# ✅ 기본 설정
root_dir = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/test_set_v0.1.0"
db_path = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.2.0"
collection_name = "tile_embeddings"
top_k = 3
output_path = "predictions_v0.2.4.json"

# ✅ 모델, DB 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
client = PersistentClient(path=db_path)
collection = client.get_or_create_collection(name=collection_name)

# ✅ 전체 결과 저장 리스트
results = []

# ✅ 슬라이드 전체 처리
slide_dirs = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

for slide_dir in slide_dirs:
    slide_id = os.path.basename(slide_dir)
    tile_paths = sorted([os.path.join(slide_dir, f) for f in os.listdir(slide_dir) if f.endswith(".jpg")])

    if not tile_paths:
        print(f"⚠️ 타일 없음: {slide_id} → 스킵")
        continue

    captions = []
    for path in tile_paths:
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.squeeze().cpu().tolist()

            results_query = collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )

            # ✅ top-k 모두 투표 반영
            metas_batch = results_query.get("metadatas", [[]])
            if metas_batch and len(metas_batch[0]) > 0:
                for m in metas_batch[0]:
                    cap = (m or {}).get("caption", "(없음)")
                    captions.append(cap)
            else:
                # 검색 결과가 없을 때 대비 (옵션)
                captions.append("(없음)")

        except Exception as e:
            print(f"❌ 오류 발생: {path} → {e}")

    if not captions:
        print(f"⚠️ 캡션 없음: {slide_id}")
        continue

    # 🔍 최빈 캡션 선택 (top-k가 모두 반영된 상태)
    caption_counts = Counter(captions)
    most_common_caption, count = caption_counts.most_common(1)[0]

    print(f"\n✅ 최종 병리 리포트: {slide_id} (빈도수: {count}회)")
    print(f"📄 {most_common_caption}")

    # 🔐 결과 저장
    result_entry = {
        "id": f"{slide_id}.tiff",
        "report": most_common_caption
    }
    results.append(result_entry)

# ✅ JSON 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n📁 전체 결과 JSON 저장 완료: {output_path}")
