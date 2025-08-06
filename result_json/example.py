import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from chromadb import PersistentClient

# ✅ 설정
image_path = "/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/data/REG_2025_tile_preprocess_final_v.0.2.1/PIT_01_00954_01/PIT_01_00954_01_11648_2688.jpg"
db_path = "/home/mts/ssd_16tb/member/jks/reg2025_tile_RAG/chroma_db"
collection_name = "tile_embeddings"
top_k = 5  # 상위 몇 개 유사한 결과를 볼지

# ✅ 모델 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# ✅ 이미지 임베딩
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(device)
with torch.no_grad():
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
embedding = image_features.squeeze().cpu().tolist()

# ✅ ChromaDB 쿼리
client = PersistentClient(path=db_path)
collection = client.get_or_create_collection(name=collection_name)

results = collection.query(
    query_embeddings=[embedding],
    n_results=top_k,
    include=["metadatas", "documents", "distances"]
)

# ✅ 결과 출력
print(f"\n🔍 [Query Image] {image_path}\n")
for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
    print(f"[Top {i+1}]")
    print(f"📁 Slide ID  : {meta.get('slide_id')}")
    print(f"🧩 Tile Name : {meta.get('tile_name')}")
    print(f"📄 Caption   : {meta.get('caption', '(없음)')}")
    print(f"📐 Distance  : {dist:.4f}")
    print("-" * 50)
