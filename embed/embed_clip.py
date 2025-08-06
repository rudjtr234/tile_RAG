import os
import json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from chromadb import PersistentClient

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로딩
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# ChromaDB 설정
chroma_client = PersistentClient(path="/home/mts/ssd_16tb/member/jks/reg2025_tile_RAG/chroma_db")
collection = chroma_client.get_or_create_collection(name="tile_embeddings")

# 📘 Groundtruth JSON 불러오기
with open("/home/mts/ssd_16tb/member/jks/reg2025_tile_RAG/embed/ground_truth_all.json", "r") as f:
    groundtruth = json.load(f)

slide_to_caption = {
    item["id"].replace(".tiff", ""): item["report"] for item in groundtruth
}

# 개별 임베딩 함수
def get_embedding(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu().numpy()

# 타일 폴더 내 임베딩 및 저장
def embed_and_store(img_dir, slide_id, slide_to_caption):
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    if not files:
        print(f"⚠️ 타일 없음: {img_dir}")
        return

    embeddings = []
    ids = []
    metadatas = []

    for fname in tqdm(files, desc=f"Embedding {slide_id}"):
        img_path = os.path.join(img_dir, fname)
        emb = get_embedding(img_path)
        embeddings.append(emb.tolist())
        tile_id = f"{slide_id}_{fname}"
        ids.append(tile_id)
        metadata = {
            "slide_id": slide_id,
            "tile_name": fname,
            "tile_path": img_path
        }
        if slide_id in slide_to_caption:
            metadata["caption"] = slide_to_caption[slide_id]
        metadatas.append(metadata)

    collection.add(embeddings=embeddings, ids=ids, metadatas=metadatas)

# 메인 루프
if __name__ == "__main__":
    root_dir = "/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/data/REG_2025_tile_preprocess_final_v.0.2.1/"
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])[:1000]

    for sub in subdirs:
        slide_dir = os.path.join(root_dir, sub)

        try:
            # 🔥 기존 데이터 삭제 후 재삽입
            collection.delete(where={"slide_id": sub})
            embed_and_store(slide_dir, sub, slide_to_caption)
            print(f"✅ 저장 완료: {sub}")
        except Exception as e:
            print(f"❌ 오류 발생: {sub} → {e}")
