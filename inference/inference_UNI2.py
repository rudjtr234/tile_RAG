import os
import json
from collections import Counter
from PIL import Image
import torch
import numpy as np
from chromadb import PersistentClient

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# ✅ 기본 설정
root_dir = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/test_set_v0.1.0"
db_path = "/home/mts/ssd_16tb/member/jks/tile_RAG_data/vectorDB/tile_RAG_embedding_db_v0.4.0"  # ⚠️ UNI2 임베딩으로 재구축 필요
collection_name = "tile_embeddings_UNI2"
top_k = 1
output_path = "predictions_v0.4.0.json"

# ✅ 디바이스
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ UNI2-h 로드 (timm)
#   - ViT-H/14 기반, num_classes=0 => forward가 임베딩을 반환
#   - 아래 kwargs는 MahmoodLab 권장 설정 예시
timm_kwargs = {
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "mlp_ratio": 2.66667*2,
    "num_classes": 0,           # 특징 벡터 반환
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

# ✅ 권장 전처리 생성
data_cfg = resolve_data_config({}, model=model)
transform = create_transform(**data_cfg)

# ✅ ChromaDB
client = PersistentClient(path=db_path)
collection = client.get_or_create_collection(name=collection_name)

# ✅ 유틸: L2 정규화
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

# ✅ 전체 결과
results = []

# ✅ 슬라이드 전체 처리
slide_dirs = sorted(
    [os.path.join(root_dir, d) for d in os.listdir(root_dir)
     if os.path.isdir(os.path.join(root_dir, d))]
)

# 선택: FP16 추론 (GPU일 때만 권장)
use_fp16 = (device.type == "cuda")

for slide_dir in slide_dirs:
    slide_id = os.path.basename(slide_dir)
    tile_paths = sorted([os.path.join(slide_dir, f) for f in os.listdir(slide_dir) if f.lower().endswith(".jpg")])

    if not tile_paths:
        print(f"⚠️ 타일 없음: {slide_id} → 스킵")
        continue

    captions = []

    for path in tile_paths:
        try:
            img = Image.open(path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                if use_fp16:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        feats = model(img_t)          # (1, 1536)
                else:
                    feats = model(img_t)

            feats = l2_normalize(feats, dim=-1)        # L2 정규화
            embedding = feats.squeeze(0).detach().cpu().tolist()

            results_query = collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )

            # 상위 1개만 사용(top_k 조절 가능)
            if len(results_query["metadatas"]) == 0 or len(results_query["metadatas"][0]) == 0:
                # 검색 실패 시 스킵
                continue

            metadata = results_query["metadatas"][0][0]  # top-1
            caption = metadata.get("caption", "(없음)")
            captions.append(caption)

        except Exception as e:
            print(f"❌ 오류 발생: {path} → {e}")

    if not captions:
        print(f"⚠️ 캡션 없음: {slide_id}")
        continue

    # 🔍 최빈 캡션 선택 (top_k=1이면 사실상 top1의 누적)
    caption_counts = Counter(captions)
    most_common_caption, count = caption_counts.most_common(1)[0]

    print(f"\n✅ 최종 병리 리포트: {slide_id} (빈도수: {count}회)")
    print(f"📄 {most_common_caption}")

    results.append({
        "id": f"{slide_id}.tiff",
        "report": most_common_caption
    })

# ✅ JSON 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n📁 전체 결과 JSON 저장 완료: {output_path}")
