import os
from collections import Counter
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from chromadb import PersistentClient

# ✅ 설정
img_dir = "/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/data/REG_2025_tile_preprocess_final_v.0.2.1/PIT_01_00333_01"
db_path = "/home/mts/ssd_16tb/member/jks/reg2025_tile_RAG/chroma_db"
collection_name = "tile_embeddings"
top_k = 1  # 타일마다 가장 유사한 것만

# ✅ 모델 & DB
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
client = PersistentClient(path=db_path)
collection = client.get_or_create_collection(name=collection_name)

# ✅ 타일 이미지 경로 수집
tile_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")])

# ✅ 결과 저장
captions = []
all_results = []

for path in tile_paths:
    try:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embedding = image_features.squeeze().cpu().tolist()

        # 🔍 DB 검색
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        
        metadata = results["metadatas"][0][0]
        caption = metadata.get("caption", "(없음)")
        distance = results["distances"][0][0]
        tile_name = os.path.basename(path)

        captions.append(caption)
        all_results.append((tile_name, caption, distance))

    except Exception as e:
        print(f"❌ 오류 발생: {path} → {e}")

# ✅ Caption 통계
caption_counts = Counter(captions)
most_common_caption, count = caption_counts.most_common(1)[0]

# ✅ 최종 결과 출력
print(f"\n✅ 최종 병리 리포트 (빈도수 기준: {count}회)\n📄 {most_common_caption}")

