# reg2025_tile_RAG

**타일 기반 디지털 병리 이미지 검색 및 추론을 위한 RAG 시스템**

본 프로젝트는 병리 타일 이미지를 벡터화하여 데이터베이스에 저장하고, 타일 이미지에 따라 유사 타일을 검색하여 캡션 정보를 추론하는 Retrieval-Augmented Generation(RAG) 구조로 구성되어 있습니다.


## 📁 알고리즘 구조

![Alt text](tile_RAG.png)


---

## 📁 프로젝트 구조

<img width="668" height="475" alt="image" src="https://github.com/user-attachments/assets/e91d7ff8-336a-47a4-aebc-aa83d6c495aa" />


---

## 🔧 주요 기능

### 1️⃣ 타일 임베딩 (embed/)

- `embed_clip.py`  
  - CLIP 모델을 사용하여 병리 타일을 임베딩합니다.
  - ChromaDB 등에 저장 가능하도록 벡터 데이터로 변환합니다.
 
- `embed_plip.py`  
  - PLIP(`vinid/plip`) 모델을 사용하여 병리 타일을 임베딩합니다.  
  - PLIP은 CLIP 구조를 기반으로 병리학 특화 데이터로 학습된 모델로, 병리학적 특징을 보다 잘 반영할 수 있습니다.
  - 임베딩 후 ChromaDB에 저장하는 구조는 `embed_clip.py`와 동일합니다.
 
- `embed_medclip.py`  
  - MedCLIP(`mahmoodlab/MedCLIP`) 모델 사용
  - 의료 전반(병리, 영상, 방사선 등) 이미지-텍스트 매핑에 특화
  - 병리뿐만 아니라 다양한 의료 이미지를 처리할 수 있는 범용 의료 특화 모델

- `ground_truth_all.json`  
  - 각 타일 디렉토리와 대응되는 정답 병리 리포트 캡션 의미.
  - 타일 디렉토리(.tiff)는 슬라이드 ID 기준으로 매핑되며, 추론 시 라벨 정답으로 활용됩니다.

### 2️⃣ 추론 및 검색 (inference/)

- `inference.py`  
  - 입력 질의(Query)형식에 따라 test 타일 임베딩 
  - 임베딩 타일을 사용하여 DB에서 유사 타일 검색
  - 검색된 타일 정보 기반으로 유사도 기반 모델이 응답 생성 (RAG)
 
- `inference_plip.py`
  - PLIP 임베딩 기반 RAG 추론 (병리 특화)

- `inference_medclip.py`
  - MedCLIP 임베딩 기반 RAG 추론 (범용 의료 특화)

- `result_json/`  
  - 추론 결과를 JSON 형식으로 저장합니다.
  - 슬라이드 ID별로 최종 추론된 병리 리포트를 포함한 JSON 파일이 저장되는 디렉토리입니다.
  - 후속 평가나 시각화에 사용됩니다.

---

## 🚀 실행 방법

```bash
# 타일 벡터화 (최초 1회) (해당 원하는 디렉토리에서 실행(모델 별로 실행 가능) 
cd embed/
python embed_clip.py

# 질의 기반 유사 타일 검색 및 응답
cd ../inference/
python inference.py (해당 원하는 디렉토리에서 실행)  # 지정된 디렉토리 별 답변 추론.
