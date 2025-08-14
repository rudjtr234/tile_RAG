import os
import json
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from chromadb import PersistentClient

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# =========================
# 🔧 경로/설정
# =========================
root_dir = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/train_set_v0.1.0"
groundtruth_json = "/home/mts/ssd_16tb/member/jks/reg2025_tile_RAG/embed/ground_truth_all.json"

db_path = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.8.0"
collection_name = "tile_embeddings_virchow2"

model_dir = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/virchow2"
safetensors_path = os.path.join(model_dir, "model.safetensors")
bin_path         = os.path.join(model_dir, "pytorch_model.bin")

BATCH_ADD = 4096
IMG_EXTS = (".jpg", ".jpeg", ".png")

# =========================
# ✅ 디바이스
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# ✅ Virchow2(비전 백본) 생성 + 가중치 로드
# - ViT-H/14 + SwiGLU + 확장 MLP(≈6832)
# =========================
def build_virchow2_model() -> torch.nn.Module:
    """
    Virchow2 체크포인트에 맞춘 ViT-H/14:
    - embed_dim=1280, depth=32, heads=16
    - SwiGLU + hidden_dim=6832 (→ mlp_ratio 정확히 5.3375)
    - SiLU, LayerScale init=1e-5
    """
    embed_dim = 1280
    hidden_dim = 6832                        # ★ 체크포인트 기준
    mlp_ratio_exact = hidden_dim / embed_dim # = 5.3375

    model = timm.create_model(
        "vit_huge_patch14_224",
        pretrained=False,
        num_classes=0,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        mlp_ratio=mlp_ratio_exact,   # ★ 반올림 없이 정확히 5.3375
        init_values=1e-5,
        no_embed_class=True,
    )
    model.reset_classifier(0)
    model.eval().to(device)

    # === 가중치 로드 ===
    state = None
    if os.path.exists(safetensors_path):
        try:
            from safetensors.torch import load_file
            state = load_file(safetensors_path)
            print(f"▶ Loaded safetensors: {safetensors_path}")
        except Exception as e:
            print(f"⚠️ safetensors 로드 실패 → {e}")

    if state is None and os.path.exists(bin_path):
        try:
            state = torch.load(bin_path, map_location="cpu")
            print(f"▶ Loaded bin: {bin_path}")
        except Exception as e:
            raise RuntimeError(f"pytorch_model.bin 로드 실패: {e}")

    if state is None:
        raise FileNotFoundError("Virchow2 체크포인트를 찾을 수 없습니다.")

    clean = {k.replace("module.", ""): v for k, v in state.items()}

    # pos_embed shape 불일치 시 제거
    if "pos_embed" in clean:
        try:
            if hasattr(model, "pos_embed") and clean["pos_embed"].shape != model.pos_embed.shape:
                print("⚠️ pos_embed shape 불일치 → 키 제거")
                del clean["pos_embed"]
        except Exception:
            pass

    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"→ load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:   print("  (info) missing 예:", missing[:6])
    if unexpected: print("  (info) unexpected 예:", unexpected[:6])

    # (선택) 첫 블록 MLP 크기 확인 로그
    try:
        fc1_w = model.blocks[0].mlp.fc1.weight
        fc2_w = model.blocks[0].mlp.fc2.weight
        print(f"[CHECK] fc1: {tuple(fc1_w.shape)}, fc2: {tuple(fc2_w.shape)} "
              f"(기대: fc1={(hidden_dim*1, embed_dim)}, fc2={(embed_dim, hidden_dim//2)})")
    except Exception:
        pass

    return model


# =========================
# ✅ 전처리(transform)
# =========================
def build_transform(model: torch.nn.Module):
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return transform


# =========================
# ✅ 임베딩 함수
# =========================
@torch.no_grad()
def get_embedding(img_path: str, transform, model) -> np.ndarray:
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise RuntimeError(f"이미지 열기 실패: {img_path} → {e}")

    x = transform(img).unsqueeze(0).to(device)       # [1, 3, 224, 224]
    feat = model.forward_features(x)                  # [1, N, D] 또는 [1, D]
    if isinstance(feat, (list, tuple)):
        feat = feat[0]
    if feat.ndim == 3:                                # [B, N, D] → 패치 토큰 평균풀링
        feat = feat.mean(dim=1)
    emb = torch.nn.functional.normalize(feat, dim=-1)  # L2 normalize
    return emb.squeeze(0).cpu().numpy()              # [D]


# =========================
# ✅ 배치 add 유틸
# =========================
def flush_batch(collection, ids, embs, metas):
    if not ids:
        return
    collection.add(ids=ids, embeddings=embs, metadatas=metas)
    ids.clear(); embs.clear(); metas.clear()


# =========================
# ✅ 메인
# =========================
def main():
    # 모델/전처리
    model = build_virchow2_model()
    transform = build_transform(model)

    # Groundtruth 로드
    with open(groundtruth_json, "r") as f:
        groundtruth = json.load(f)
    slide_to_caption = {item["id"].replace(".tiff", ""): item["report"] for item in groundtruth}

    gt_slide_ids = set(slide_to_caption.keys())
    tile_slide_ids = set([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    matched_ids = sorted(list(gt_slide_ids & tile_slide_ids))

    # ChromaDB
    client = PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    for slide_id in matched_ids:
        slide_dir = os.path.join(root_dir, slide_id)
        files = sorted([f for f in os.listdir(slide_dir) if f.lower().endswith(IMG_EXTS)])
        if not files:
            print(f"⚠️ 타일 없음: {slide_dir}")
            continue

        # 기존 동일 slide_id 메타 삭제(리빌드)
        try:
            collection.delete(where={"slide_id": slide_id})
        except Exception as e:
            print(f"⚠️ delete 실패(무시 가능): {slide_id} → {e}")

        caption = slide_to_caption[slide_id]
        ids, embeddings, metadatas = [], [], []

        pbar = tqdm(files, desc=f"[Virchow2] {slide_id}", ncols=100)
        for fname in pbar:
            img_path = os.path.join(slide_dir, fname)
            try:
                emb = get_embedding(img_path, transform, model)
            except Exception as e:
                print(f"❌ 임베딩 실패: {img_path} → {e}")
                continue

            tile_id = f"{slide_id}_{fname}"
            ids.append(tile_id)
            embeddings.append(emb.tolist())
            metadatas.append({
                "slide_id": slide_id,
                "tile_name": fname,
                "tile_path": img_path,
                "caption": caption
            })

            if len(ids) >= BATCH_ADD:
                flush_batch(collection, ids, embeddings, metadatas)

        # 남은 배치 마무리
        flush_batch(collection, ids, embeddings, metadatas)
        print(f"✅ 저장 완료: {slide_id}")

    print("🎉 모든 처리 완료")


if __name__ == "__main__":
    main()
