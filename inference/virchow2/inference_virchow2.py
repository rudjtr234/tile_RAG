#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from collections import defaultdict
from typing import List, Tuple, Dict, Any

import torch
from PIL import Image
from chromadb import PersistentClient

# =========================
# ✅ 기본 설정
# =========================
root_dir = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/test_set_v0.1.0"
db_path  = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.8.0"

collection_name = "tile_embeddings_virchow2"
top_k = 3
vote_mode = "majority"  # or "weighted"
output_path = "predictions_v0.8.0.json"

VIRCHOW2_DIR = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/virchow2"
CKPT_ST = os.path.join(VIRCHOW2_DIR, "model.safetensors")
CKPT_PT = os.path.join(VIRCHOW2_DIR, "pytorch_model.bin")

# =========================
# ✅ timm / transforms
# =========================
import timm
from safetensors.torch import load_file
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefixes: Tuple[str, ...] = ("module.", "model.")) -> Dict[str, Any]:
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def load_virchow2_vith14() -> torch.nn.Module:
    # 1) 체크포인트 로드
    if os.path.exists(CKPT_ST):
        raw_state = load_file(CKPT_ST)
        print(f"▶ Loaded safetensors: {CKPT_ST}")
    else:
        raw_state = torch.load(CKPT_PT, map_location="cpu")
        if isinstance(raw_state, dict) and "state_dict" in raw_state:
            raw_state = raw_state["state_dict"]
        print(f"▶ Loaded bin: {CKPT_PT}")

    state = _strip_prefix_if_present(raw_state)

    # 2) hidden_dim 추론 (fc2: [embed_dim, hidden_dim])
    fc2_key = None
    for k in ("blocks.0.mlp.fc2.weight", "blocks.0.mlp.fc2.fc.weight", "blocks.0.mlp.fc2_layer.weight"):
        if k in state:
            fc2_key = k
            break
    if fc2_key is None:
        raise RuntimeError("체크포인트에서 fc2.weight 키를 찾을 수 없습니다.")
    fc2_w = state[fc2_key]
    embed_dim = int(fc2_w.shape[0])   # 1280
    hidden_dim = int(fc2_w.shape[1])  # 3416 (ckpt)
    # ★ SwiGLU일 때는 mlp_ratio를 '두 배'로 넣어야 hidden_dim이 맞습니다.
    mlp_ratio = 2.0 * hidden_dim / float(embed_dim)  # = 5.3375
    print(f"[ckpt] embed_dim={embed_dim}, hidden_dim={hidden_dim}, mlp_ratio_for_SwiGLU={mlp_ratio}")

    # 3) timm 모델 생성 (SwiGLU + SiLU)
    model = timm.create_model(
        "vit_huge_patch14_224",
        pretrained=False,
        num_classes=0,
        global_pool="",
        img_size=224,
        mlp_ratio=mlp_ratio,                 # ★ 2배로 넣는다
        mlp_layer=timm.layers.SwiGLUPacked,  # ★ SwiGLU
        act_layer=torch.nn.SiLU,             # ★ SiLU
        init_values=1e-5,
        dynamic_img_size=True,
        # 필요 시 ckpt에 맞게 아래 옵션도 사용
        # reg_tokens=4,
        # no_embed_class=True,
    ).to(device)

    # 4) pos_embed shape 불일치 시 제거
    if "pos_embed" in state and hasattr(model, "pos_embed"):
        try:
            if state["pos_embed"].shape != model.pos_embed.shape:
                print("⚠️ pos_embed shape 불일치 → 제거")
                del state["pos_embed"]
        except Exception:
            pass

    # 5) 가중치 로드
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load_state_dict] missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing[:10]:    print("  - missing (앞 10개):", missing[:10])
    if unexpected[:10]: print("  - unexpected (앞 10개):", unexpected[:10])

    model.eval()
    return model


print("🔧 모델 로딩 중...")
model = load_virchow2_vith14()

# 전처리: DB 구축 때와 동일하게 resolve_data_config → create_transform 사용
cfg = resolve_data_config({}, model=model)
transform = create_transform(**cfg)
print("✅ 모델/전처리 준비 완료")

# =========================
# ✅ ChromaDB 초기화
# =========================
client = PersistentClient(path=db_path)
collection = client.get_or_create_collection(name=collection_name)
# metric='cosine' 가정 (distances ≈ 1 - cos_sim)

# =========================
# ✅ 유틸: Virchow2 임베딩 추출 (DB와 동일 파이프라인)
# =========================
@torch.inference_mode()
def image_embedding(pil_img: Image.Image) -> List[float]:
    x = transform(pil_img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    feats = model.forward_features(x)               # [1, N, D] 또는 [1, D]
    if isinstance(feats, (list, tuple)):
        feats = feats[0]
    if feats.ndim == 3:                             # 패치 토큰 평균풀링
        feats = feats.mean(dim=1)
    emb = torch.nn.functional.normalize(feats, dim=-1)  # L2 normalize
    return emb.squeeze(0).cpu().tolist()


# =========================
# ✅ 데이터 유틸
# =========================
def iter_slide_dirs(root: str) -> List[str]:
    return sorted([
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])


def iter_tile_paths(slide_dir: str) -> List[str]:
    return sorted([
        os.path.join(slide_dir, f)
        for f in os.listdir(slide_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


def query_topk(emb: List[float], k: int) -> Tuple[List[Dict[str, Any]], List[float]]:
    q = collection.query(
        query_embeddings=[emb],
        n_results=k,
        include=["metadatas", "distances"],
    )
    metas = q.get("metadatas", [[]])[0] or []
    dists = q.get("distances", [[]])[0] or []
    return metas, dists


def vote_scores_update(vote_scores: defaultdict, metas: List[Dict[str, Any]], dists: List[float], mode: str):
    for m, d in zip(metas, dists):
        caption = (m or {}).get("caption", "(없음)")
        if mode == "weighted":
            w = max(0.0, 1.0 - float(d))  # cosine distance → sim
            vote_scores[caption] += w
        else:
            vote_scores[caption] += 1.0


# =========================
# ✅ 메인
# =========================
def main():
    results = []
    slide_dirs = iter_slide_dirs(root_dir)

    if not slide_dirs:
        print("⚠️ 처리할 슬라이드 디렉토리가 없습니다.")
        return

    for slide_dir in slide_dirs:
        slide_id = os.path.basename(slide_dir)
        tile_paths = iter_tile_paths(slide_dir)

        if not tile_paths:
            print(f"⚠️ 타일 없음: {slide_id} → 스킵")
            continue

        vote_scores = defaultdict(float)

        for path in tile_paths:
            try:
                img = Image.open(path).convert("RGB")
                emb = image_embedding(img)
                metas, dists = query_topk(emb, top_k)
                vote_scores_update(vote_scores, metas, dists, vote_mode)
            except Exception as e:
                print(f"❌ 오류 발생: {path} → {e}")

        if not vote_scores:
            print(f"⚠️ 캡션 없음: {slide_id}")
            continue

        final_caption, final_score = max(vote_scores.items(), key=lambda x: x[1])

        print(f"\n✅ 최종 병리 리포트: {slide_id} (점수: {final_score:.2f}, mode={vote_mode}, K={top_k})")
        print(f"📄 {final_caption}")

        results.append({
            "id": f"{slide_id}.tiff",
            "report": final_caption
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n📁 전체 결과 JSON 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
