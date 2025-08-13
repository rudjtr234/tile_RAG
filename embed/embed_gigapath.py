import os
import json
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from chromadb import PersistentClient

# 🔁 추가: gigapath 타일 인코더 로딩을 위한 timm/torchvision
import timm
from torchvision import transforms

# =========================
# ✅ 기본 설정
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# ✅ GigaPath 타일 인코더 로딩
#    - 사전에 HF 동의 & 토큰 필요:
#      export HF_TOKEN=xxxxx
# =========================
# 참고: https://huggingface.co/prov-gigapath/prov-gigapath (Model card)
tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device)
tile_encoder.eval()

# GigaPath 권장 전처리 (HF Model Card 예시와 동일)
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# =========================
# ✅ ChromaDB 설정
# =========================
chroma_client = PersistentClient(
    path="/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.6.0"
)
collection = chroma_client.get_or_create_collection(name="tile_embeddings_gigapath")

# =========================
# ✅ Groundtruth JSON 불러오기
# =========================
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

# =========================
# ✅ 임베딩 함수 (GigaPath)
#   - 출력: 1536-d 벡터 (L2 정규화)
# =========================
@torch.inference_mode()
def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device, non_blocking=True)
    feat = tile_encoder(x)                 # [1, 1536]
    feat = torch.nn.functional.normalize(feat, dim=-1)  # L2 norm
    return feat.squeeze(0).cpu().numpy()

# =========================
# ✅ 타일 디렉토리 단위 저장 함수
# =========================
def embed_and_store(img_dir, slide_id, caption):
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
    if not files:
        print(f"⚠️ 타일 없음: {img_dir}")
        return

    embeddings, ids, metadatas = [], [], []

    for fname in tqdm(files, desc=f"Embedding {slide_id} (GigaPath)"):
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

# =========================
# ✅ 메인 루프
# =========================
if __name__ == "__main__":
    for slide_id in matched_ids:
        slide_dir = os.path.join(root_dir, slide_id)
        try:
            # 기존 slide_id 데이터 삭제 후 재삽입
            collection.delete(where={"slide_id": slide_id})
            embed_and_store(slide_dir, slide_id, slide_to_caption[slide_id])
            print(f"✅ 저장 완료 (GigaPath): {slide_id}")
        except Exception as e:
            print(f"❌ 오류 발생: {slide_id} → {e}")
