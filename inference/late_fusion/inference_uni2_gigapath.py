# UNI2 + GigaPath 동시 검색 → 결합(RRF/zsum/softsum) → 슬라이드 단위 투표 → JSON 저장 (파서 없이 한 번에 실행)

import os
import json
from collections import defaultdict, Counter
from PIL import Image
import math
import torch
import numpy as np

from chromadb import PersistentClient
from chromadb.config import Settings

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms

# =========================
# ✅ 기본 설정 (필요 시 아래 값만 수정)
# =========================
ROOT_DIR = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/test_set_v0.1.0"

# DB 경로 & 컬렉션명 (※ 네가 준 이름 그대로)
UNI2_DB_PATH  = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.4.0"
GIGA_DB_PATH  = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.6.0"
UNI2_COL_NAME = "tile_embeddings_UNI2"
GIGA_COL_NAME = "tile_embeddings_gigapath"

# 검색/결합 파라미터
K_BASE        = 60           # 각 DB에서 먼저 가져올 후보 수
FUSION_TOPK   = 10            # 두 DB 결합 후 상위 몇 개를 슬라이드 투표에 반영할지
METRIC        = "cosine"     # 'cosine' | 'l2' | 'similarity'  (Chroma 반환 점수 성격)
FUSION_METHOD = "rrf"        # 'rrf' | 'zsum' | 'softsum' (초기엔 rrf 권장)
W_UNI2, W_GIGA = 0.5, 0.5    # zsum/softsum에서만 사용

# 슬라이드 수준 투표 방식
VOTE_MODE     = "weighted"   # 'majority' | 'weighted'

# 출력 파일
OUTPUT_PATH   = "predictions_v.0.9.2.json"

# =========================
# ✅ 유틸
# =========================
def z_norm(scores):
    if len(scores) <= 1:
        return [0.0 for _ in scores]
    mu = sum(scores)/len(scores)
    var = sum((s-mu)**2 for s in scores)/(len(scores)-1)
    std = var**0.5 if var > 0 else 1.0
    return [(s - mu)/std for s in scores]

def softmax(xs):
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e/s for e in exps]

def to_similarity_from_distance(dist_list, metric: str = "cosine"):
    # 거리(작을수록 유사) → 유사도(클수록 유사): -distance
    if metric.lower() in ["cosine", "l2", "distance"]:
        return [-float(d) for d in dist_list]
    return [float(d) for d in dist_list]  # 이미 similarity인 경우

def rrf_fuse(ids1, ids2, k=60):
    # 랭크 기반 결합 (스케일 프리)
    fused = defaultdict(float)
    for rank_list in [ids1, ids2]:
        for r, cid in enumerate(rank_list, start=1):
            fused[cid] += 1.0/(k + r)
    return fused

# =========================
# ✅ 임베딩: UNI2 (네가 준 코드 그대로)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# UNI2-h 로드
timm_kwargs = {
    "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24,
    "init_values": 1e-5, "embed_dim": 1536, "mlp_ratio": 2.66667*2,
    "num_classes": 0, "no_embed_class": True,
    "mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU,
    "reg_tokens": 8, "dynamic_img_size": True,
}
uni2_model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs).to(device)
uni2_model.eval()
uni2_cfg = resolve_data_config({}, model=uni2_model)
uni2_transform = create_transform(**uni2_cfg)


@torch.inference_mode()
def image_embedding_uni2(pil_img: Image.Image):
    img_t = uni2_transform(pil_img).unsqueeze(0).to(device, non_blocking=True)
    with torch.no_grad():
        feats = uni2_model(img_t)  # (1, 1536)
    if isinstance(feats, (list, tuple)):
        feats = feats[0]
    if feats.dim() > 2:
        feats = feats.mean(dim=(2, 3))
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.squeeze(0).detach().cpu().tolist()

# =========================
# ✅ 임베딩: GigaPath (timm 허브)
# =========================
giga_model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device)
giga_model.eval()
giga_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

@torch.inference_mode()
def image_embedding_gigapath(pil_img: Image.Image):
    x = giga_transform(pil_img).unsqueeze(0).to(device, non_blocking=True)
    feat = giga_model(x)
    if isinstance(feat, (list, tuple)):
        feat = feat[0]
    if feat.dim() > 2:
        feat = feat.mean(dim=(2, 3))
    feat = torch.nn.functional.normalize(feat, dim=-1)
    return feat.squeeze(0).cpu().tolist()

# =========================
# ✅ 메인 실행
# =========================
def main():
    # --- DB/컬렉션 연결 ---
    client_uni2 = PersistentClient(path=UNI2_DB_PATH, settings=Settings(allow_reset=False))
    client_giga = PersistentClient(path=GIGA_DB_PATH, settings=Settings(allow_reset=False))
    col_uni2 = client_uni2.get_or_create_collection(name=UNI2_COL_NAME)
    col_giga = client_giga.get_or_create_collection(name=GIGA_COL_NAME)

    # --- 슬라이드 디렉토리 ---
    slide_dirs = sorted(
        [os.path.join(ROOT_DIR, d) for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    )

    results = []
    use_fp16 = (device.type == "cuda")

    for slide_dir in slide_dirs:
        slide_id = os.path.basename(slide_dir)
        tile_paths = sorted([os.path.join(slide_dir, f) for f in os.listdir(slide_dir) if f.lower().endswith(".jpg")])

        if not tile_paths:
            print(f"⚠️ 타일 없음: {slide_id} → 스킵")
            continue

        vote_scores = defaultdict(float)

        for path in tile_paths:
            try:
                img = Image.open(path).convert("RGB")

                # 1) 각 DB 전용 임베딩
                if use_fp16 and device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        emb_uni2 = image_embedding_uni2(img)
                        emb_giga = image_embedding_gigapath(img)
                else:
                    emb_uni2 = image_embedding_uni2(img)
                    emb_giga = image_embedding_gigapath(img)

                # 2) 각 DB에서 후보 검색
                q_u = col_uni2.query(
                    query_embeddings=[emb_uni2],
                    n_results=K_BASE,
                    include=["metadatas", "distances"]
                )
                q_g = col_giga.query(
                    query_embeddings=[emb_giga],
                    n_results=K_BASE,
                    include=["metadatas", "distances"]
                )

                ids_u   = q_u.get("ids",[[]])[0]
                dists_u = q_u.get("distances",[[]])[0]
                metas_u = q_u.get("metadatas",[[]])[0]

                ids_g   = q_g.get("ids",[[]])[0]
                dists_g = q_g.get("distances",[[]])[0]
                metas_g = q_g.get("metadatas",[[]])[0]

                # 3) 결합 스코어 계산
                meta_map = {}
                for cid, m in zip(ids_u, metas_u):
                    meta_map[cid] = m
                for cid, m in zip(ids_g, metas_g):
                    if cid not in meta_map:
                        meta_map[cid] = m

                if FUSION_METHOD == "rrf":
                    fused = rrf_fuse(ids_u, ids_g, k=60)
                    merged = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:FUSION_TOPK]
                else:
                    sim_u = to_similarity_from_distance(dists_u, metric=METRIC)
                    sim_g = to_similarity_from_distance(dists_g, metric=METRIC)
                    fused_scores = defaultdict(float)

                    if FUSION_METHOD == "zsum":
                        zn_u = z_norm(sim_u)
                        zn_g = z_norm(sim_g)
                        for cid, s in zip(ids_u, zn_u):
                            fused_scores[cid] += W_UNI2*float(s)
                        for cid, s in zip(ids_g, zn_g):
                            fused_scores[cid] += W_GIGA*float(s)

                    elif FUSION_METHOD == "softsum":
                        p_u = softmax(sim_u)
                        p_g = softmax(sim_g)
                        for cid, s in zip(ids_u, p_u):
                            fused_scores[cid] += W_UNI2*float(s)
                        for cid, s in zip(ids_g, p_g):
                            fused_scores[cid] += W_GIGA*float(s)

                    else:
                        raise ValueError("알 수 없는 FUSION_METHOD")

                    merged = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:FUSION_TOPK]

                # 4) 슬라이드 캡션 투표
                for cid, fscore in merged:
                    meta = meta_map.get(cid, {}) or {}
                    caption = meta.get("caption", "(없음)")
                    if VOTE_MODE == "weighted":
                        vote_scores[caption] += float(fscore)
                    else:
                        vote_scores[caption] += 1.0

            except Exception as e:
                print(f"❌ 오류 발생: {path} → {e}")

        if not vote_scores:
            print(f"⚠️ 캡션 없음: {slide_id}")
            continue

        final_caption, final_score = max(vote_scores.items(), key=lambda x: x[1])
        print(f"\n✅ 최종 병리 리포트: {slide_id} (점수: {final_score:.2f}, vote={VOTE_MODE}, fusion={FUSION_METHOD}, K_base={K_BASE}, N={FUSION_TOPK})")
        print(f"📄 {final_caption}")

        results.append({
            "id": f"{slide_id}.tiff",
            "report": final_caption
        })

    # 5) 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📁 전체 결과 JSON 저장 완료: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
