
import os
import json
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from chromadb import PersistentClient
import os, timm, torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timm_kwargs = {
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "mlp_ratio": 2.66667*2,
    "num_classes": 0,
    "no_embed_class": True,
    "mlp_layer": timm.layers.SwiGLUPacked,
    "act_layer": torch.nn.SiLU,
    "reg_tokens": 8,
    "dynamic_img_size": True,
}

model = timm.create_model(
    "hf-hub:MahmoodLab/UNI2-h",
    pretrained=True,
    **timm_kwargs,
).to(device)
model.eval()

# 권장 전처리 (반드시 이걸로 입력 만들기)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

# ======================
# ✅ ChromaDB 설정
# ======================
chroma_client = PersistentClient(
    path="/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.4.1"
)
collection = chroma_client.get_or_create_collection(name="tile_embeddings_final_UNI2")

# ======================
# ✅ Groundtruth JSON 불러오기
# ======================
with open("/home/mts/ssd_16tb/member/jks/reg2025_tile_RAG/embed/ground_truth_all.json", "r") as f:
    groundtruth = json.load(f)

# ✅ 슬라이드 ID → 캡션 맵
slide_to_caption = {
    item["id"].replace(".tiff", ""): item["report"] for item in groundtruth
}

# ✅ 공통된 슬라이드 ID만 추출
root_dir = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/reg_2025_final_v.0.2.0"
gt_slide_ids = set(slide_to_caption.keys())
tile_slide_ids = set([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
matched_ids = sorted(list(gt_slide_ids & tile_slide_ids))

# ======================
# ✅ 임베딩 함수 (UNI2-h)
#   - 출력: (1536,) numpy
#   - L2 정규화
# ======================
@torch.inference_mode()
def get_embedding(img_path: str) -> np.ndarray:
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W] (기본 224)
    # 속도 향상: FP16 자동 캐스트 (CUDA일 때만)
    use_amp = (device.type == "cuda")
    ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else torch.no_grad()
    with ctx:
        feat = model(tensor)  # [1, 1536]
    feat = torch.nn.functional.normalize(feat, dim=-1)  # L2 정규화
    return feat.squeeze(0).detach().cpu().numpy()

# ======================
# ✅ 타일 디렉토리 단위 저장 함수
# ======================
def embed_and_store(img_dir: str, slide_id: str, caption: str):
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg"))])
    if not files:
        print(f"⚠️ 타일 없음: {img_dir}")
        return

    embeddings, ids, metadatas = [], [], []

    for fname in tqdm(files, desc=f"Embedding {slide_id}"):
        img_path = os.path.join(img_dir, fname)
        emb = get_embedding(img_path)  # (1536,)
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

# ======================
# ✅ 메인 루프
# ======================
if __name__ == "__main__":
    for slide_id in matched_ids:
        slide_dir = os.path.join(root_dir, slide_id)

        try:
            # 🔥 기존 slide_id 데이터 삭제 후 재삽입
            collection.delete(where={"slide_id": slide_id})
            embed_and_store(slide_dir, slide_id, slide_to_caption[slide_id])
            print(f"✅ 저장 완료: {slide_id}")
        except Exception as e:
            print(f"❌ 오류 발생: {slide_id} → {e}")






