import json
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

CONNECTION_STRING = "postgresql+psycopg://nurasyk@localhost:5432/postgres"
COLLECTION_NAME = "test_collection"

with open("test_docs.json") as f:
    docs = json.load(f)

texts = [doc["title"] + "\n" + doc["content"] for doc in docs]
metadatas = [{"source": "test_docs.json", "id": i} for i in range(len(docs))]

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

vector_store.add_texts(texts=texts, metadatas=metadatas)

print(f"✅")

query = "sample text"
results = vector_store.similarity_search(query, k=1)

if results:
    print(f"[Search] {results[0].page_content[:50]}...")