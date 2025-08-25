import os
import json
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from chromadb import PersistentClient
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch.multiprocessing as mp
from math import ceil

# =========================
# 환경 기본
# =========================
DB_PATH = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.4.2"
COLL_NAME = "tile_embeddings_UNI2_AUG"  # 기존과 동일 이름 사용
ROOT_DIR  = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/train_set_v0.1.0"
GT_PATH   = "/home/mts/ssd_16tb/member/jks/reg2025_tile_RAG/embed/ground_truth_all.json"

# =========================
# 모델/전처리 작성자 함수(프로세스별 초기화)
# =========================
def build_model_and_transform(device):
    timm_kwargs = {
        "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24,
        "init_values": 1e-5, "embed_dim": 1536, "mlp_ratio": 2.66667*2,
        "num_classes": 0, "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU,
        "reg_tokens": 8, "dynamic_img_size": True,
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs).to(device)
    model.eval()
    data_cfg = resolve_data_config({}, model=model)
    transform = create_transform(**data_cfg)
    return model, transform

# ======================
# 공용 유틸
# ======================
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

@torch.inference_mode()
def _forward_batch(model, tensor: torch.Tensor, device, use_fp16: bool) -> torch.Tensor:
    ctx = (torch.autocast(device_type="cuda", dtype=torch.float16) if (use_fp16 and device.type=="cuda") else torch.no_grad())
    with ctx:
        feats = model(tensor)           # [N, D]
    return l2_normalize(feats, dim=-1)  # [N, D]

# ======================
# 2x2 멀티크롭(각 크롭 임베딩 반환)
# ======================
@torch.inference_mode()
def embed_2x2(img_path: str, model, transform, device, use_fp16: bool):
    image = Image.open(img_path).convert("RGB")
    W, H = image.size
    gx = gy = 2

    # 균등 2x2 분할
    cw, ch = W // gx, H // gy
    crops = []
    for ry in range(gy):
        for rx in range(gx):
            x0, y0 = rx * cw, ry * ch
            x1, y1 = min(x0 + cw, W), min(y0 + ch, H)
            crops.append(image.crop((x0, y0, x1, y1)))

    batch = torch.stack([transform(c) for c in crops], dim=0).to(device)  # [4,3,224,224]
    feats = _forward_batch(model, batch, device, use_fp16)                # [4, D]
    return [feats[i].detach().cpu().numpy() for i in range(feats.shape[0])]

from itertools import count

# ======================
# 타일 디렉토리 저장 (ids 포함, 메타데이터 최소)
# ======================
def embed_and_store(img_dir: str, slide_id: str, collection, slide_to_caption, model, transform, device, use_fp16: bool):
    files = [f for f in sorted(os.listdir(img_dir))
             if f.lower().endswith((".jpg", ".jpeg")) and os.path.isfile(os.path.join(img_dir, f))]
    if not files:
        print(f"⚠️ 타일 없음: {img_dir}")
        return

    BATCH_LIMIT = 4096
    embeddings, metadatas, ids = [], [], []
    local_counter = count(1)  # 슬라이드 내부 일련번호: 1,2,3,...

    tiff_id = f"{slide_id}.tiff"
    caption = slide_to_caption.get(slide_id, "")

    for fname in tqdm(files, desc=f"[{slide_id}] Embedding", leave=False):
        img_path = os.path.join(img_dir, fname)
        base = os.path.splitext(fname)[0]  # 부모 896 타일 이름
        try:
            emb_list = embed_2x2(img_path, model, transform, device, use_fp16)  # 4개 임베딩
            for emb in emb_list:
                embeddings.append(emb.tolist())
                metadatas.append({
                    "slide_id": slide_id,
                    "tiff_id": tiff_id,
                    "tile_name": base,
                    "caption": caption
                })
                # ✅ 고유 ID: 슬라이드 이름 + 일련번호 (예: PIT_01_00001_01_00000001)
                n = next(local_counter)
                ids.append(f"{slide_id}_{n:08d}")

        except Exception as e:
            print(f"  ↳ 건너뜀(에러): {fname} → {e}")
            continue

        # 중간 flush
        if len(embeddings) >= BATCH_LIMIT:
            collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
            embeddings, metadatas, ids = [], [], []

    # 잔여 flush
    if embeddings:
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

# ======================
# 워커(프로세스) 함수: GPU 하나 담당
# ======================
def worker(rank: int, slide_ids_chunk, world_size: int):
    # 디바이스 설정
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        use_fp16 = True
    else:
        device = torch.device("cpu")
        use_fp16 = False

    # 프로세스별 리소스 초기화
    model, transform = build_model_and_transform(device)
    chroma_client = PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLL_NAME)

    # GT 로드 (프로세스별 독립 로드)
    with open(GT_PATH, "r") as f:
        groundtruth = json.load(f)
    slide_to_caption = {item["id"].replace(".tiff", ""): item["report"] for item in groundtruth}

    # 처리 루프
    for slide_id in slide_ids_chunk:
        slide_dir = os.path.join(ROOT_DIR, slide_id)
        try:
            # 동일 slide_id 기존 데이터 제거(동시에 여러 워커가 같은 slide를 만지지 않으므로 안전)
            collection.delete(where={"slide_id": slide_id})
            embed_and_store(slide_dir, slide_id, collection, slide_to_caption, model, transform, device, use_fp16)
            print(f"[GPU{rank}] ✅ 저장 완료: {slide_id}")
        except Exception as e:
            print(f"[GPU{rank}] ❌ 오류 발생: {slide_id} → {e}")

# ======================
# 메인: 슬라이드 분할 후 멀티 GPU 실행
# ======================
def split_evenly(items, n_parts):
    n = len(items)
    if n_parts <= 1:
        return [items]
    step = ceil(n / n_parts)
    return [items[i:i+step] for i in range(0, n, step)]

def main():
    # GT와 슬라이드 목록 준비(메인 프로세스)
    with open(GT_PATH, "r") as f:
        groundtruth = json.load(f)
    slide_to_caption = {item["id"].replace(".tiff", ""): item["report"] for item in groundtruth}

    gt_slide_ids = set(slide_to_caption.keys())
    tile_slide_ids = set([d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))])
    matched_ids = sorted(list(gt_slide_ids & tile_slide_ids))

    if not matched_ids:
        print("⚠️ 처리할 슬라이드가 없습니다.")
        return

    # GPU 수
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        # 싱글 프로세스 경로
        print("🔧 Single GPU/CPU 모드로 실행합니다.")
        worker(rank=0, slide_ids_chunk=matched_ids, world_size=1)
        return

    print(f"🚀 Multi-GPU 실행: GPU {num_gpus}개")
    chunks = split_evenly(matched_ids, num_gpus)

    # spawn
    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(num_gpus):
        chunk = chunks[rank] if rank < len(chunks) else []
        p = mp.Process(target=worker, args=(rank, chunk, num_gpus), daemon=False)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("🎉 모든 작업 완료")

# ======================
# 엔트리포인트
# ======================
if __name__ == "__main__":
    main()
