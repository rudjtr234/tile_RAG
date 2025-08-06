from chromadb import PersistentClient

client = PersistentClient(path="/home/mts/ssd_16tb/member/jks/reg2025_tile_RAG/chroma_db")
collection = client.get_or_create_collection(name="tile_embeddings")

print(f"총 타일 개수: {collection.count()}")

# 샘플 5개 메타데이터 출력
results = collection.get(limit=5)
for i in range(len(results['ids'])):
    print(f"\n📌 ID: {results['ids'][i]}")
    print(f"📄 Metadata: {results['metadatas'][i]}")
