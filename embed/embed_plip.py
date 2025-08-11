import os
import json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from chromadb import PersistentClient

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 로딩 (PLIP)
model = CLIPModel.from_pretrained("vinid/plip").to(device)
processor = CLIPProcessor.from_pretrained("vinid/plip")

# ✅ ChromaDB 설정
chroma_client = PersistentClient(path="/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.2.1")
collection = chroma_client.get_or_create_collection(name="tile_embeddings_plip")

# ✅ Groundtruth JSON 불러오기
with open("/home/mts/ssd_16tb/member/jks/reg2025_tile_RAG/embed/ground_truth_all.json", "r") as f:
    groundtruth = json.load(f)

# ✅ 슬라이드 ID → 캡션 맵
slide_to_caption = {
    item["id"].replace(".tiff", ""): item["report"] for item in groundtruth
}

# ✅ 공통된 슬라이드 ID만 추출
root_dir = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/train_set_v0.1.0"
gt_slide_ids = set(slide_to_caption.keys())
tile_slide_ids = set([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
matched_ids = sorted(list(gt_slide_ids & tile_slide_ids))

# ✅ 임베딩 함수
def get_embedding(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu().numpy()

# ✅ 타일 디렉토리 단위 저장 함수
def embed_and_store(img_dir, slide_id, caption):
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
        metadatas.append({
            "slide_id": slide_id,
            "tile_name": fname,
            "tile_path": img_path,
            "caption": caption
        })

    collection.add(embeddings=embeddings, ids=ids, metadatas=metadatas)

# ✅ 메인 루프
if __name__ == "__main__":
    for slide_id in matched_ids:
        slide_dir = os.path.join(root_dir, slide_id)

        try:
            # 🔥 기존 slide_id에 해당하는 데이터 삭제 후 재삽입
            collection.delete(where={"slide_id": slide_id})
            embed_and_store(slide_dir, slide_id, slide_to_caption[slide_id])
            print(f"✅ 저장 완료: {slide_id}")
        except Exception as e:
            print(f"❌ 오류 발생: {slide_id} → {e}")
