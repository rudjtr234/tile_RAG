import os
import json
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from chromadb import PersistentClient

import timm
from torchvision import transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# ======================
# ✅ 디바이스
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# ✅ H-optimus-1 로드
#    - 전처리는 timm의 config 기반으로 자동 생성
#    - 출력: (1, D)  (D는 모델 임베딩 차원; 동적 처리)
# ======================
model = timm.create_model(
    "hf-hub:bioptimus/H-optimus-1",
    pretrained=True,
    init_values=1e-5,
    dynamic_img_size=False,   # 권장: False
).to(device)
model.eval()

# ✅ 모델 권장 전처리 (mean/std/resize를 모델 카드에 맞춰 자동 세팅)
data_cfg = resolve_data_config(model=model)
hopt_transform = create_transform(**data_cfg)

# ======================
# ✅ ChromaDB 설정
# ======================
chroma_client = PersistentClient(
    path="/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.7.0"
)
collection = chroma_client.get_or_create_collection(name="tile_embeddings_HOPT1")

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
root_dir = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/train_set_v0.1.0"
gt_slide_ids = set(slide_to_caption.keys())
tile_slide_ids = set([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
matched_ids = sorted(list(gt_slide_ids & tile_slide_ids))

# ======================
# ✅ 임베딩 함수 (H-optimus-1)
#   - 출력: (D,) numpy
#   - L2 정규화
# ======================
@torch.inference_mode()
def get_embedding(img_path: str) -> np.ndarray:
    image = Image.open(img_path).convert("RGB")
    tensor = hopt_transform(image).unsqueeze(0).to(device)  # [1, 3, H, W] (보통 224x224)
    use_amp = (device.type == "cuda")
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            feat = model(tensor)  # [1, D]
    else:
        feat = model(tensor)
    feat = torch.nn.functional.normalize(feat, dim=-1)  # L2 정규화
    return feat.squeeze(0).detach().cpu().numpy()       # (D,)

# ======================
# ✅ 타일 디렉토리 단위 저장 함수
# ======================
def embed_and_store(img_dir: str, slide_id: str, caption: str, batch_size: int = 512):
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not files:
        print(f"⚠️ 타일 없음: {img_dir}")
        return

    embeddings, ids, metadatas = [], [], []

    for fname in tqdm(files, desc=f"Embedding {slide_id}"):
        img_path = os.path.join(img_dir, fname)
        emb = get_embedding(img_path)  # (D,)
        embeddings.append(emb.tolist())
        tile_id = f"{slide_id}_{fname}"
        ids.append(tile_id)
        metadatas.append({
            "slide_id": slide_id,
            "tile_name": fname,
            "tile_path": img_path,
            "caption": caption
        })

        # 🔁 메모리/파일사이즈 폭증 방지: 주기적으로 커밋
        if len(ids) >= batch_size:
            collection.add(embeddings=embeddings, ids=ids, metadatas=metadatas)
            embeddings, ids, metadatas = [], [], []

    # 남은 잔여분 커밋
    if ids:
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
