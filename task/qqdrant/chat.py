import asyncio
import time
import random
from operator import itemgetter

from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_core.language_models import FakeListLLM
from sqlalchemy.ext.asyncio import create_async_engine


DB_URL = "postgresql+psycopg://nurasyk@localhost:5432/postgres"
COLLECTION_NAME = "test_collection"
MAX_CONCURRENT_USERS = 20


engine = create_async_engine(
    DB_URL,
    pool_size=MAX_CONCURRENT_USERS, 
    max_overflow=10
)

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=engine,
    use_jsonb=True,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = FakeListLLM(responses=["[DB_TEST_OK]"])

prompt = ChatPromptTemplate.from_template("Context: {context} Query: {question}")

chain = (
    {
        "context": itemgetter("question") | retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": itemgetter("question")
    }
    | prompt
    | llm
    | StrOutputParser()
)

semaphore = asyncio.Semaphore(MAX_CONCURRENT_USERS)

async def run_query(user_id):
    query = "Что говорится в документах?"
    async with semaphore:
        # Убираем все задержки для максимального стресса
        start = time.time()
        try:
            await chain.ainvoke({"question": query})
            elapsed = time.time() - start
            print(f"👤 User {user_id:02d} | Latency: {elapsed:.3f}s")
            return elapsed
        except Exception as e:
            print(f"❌ User {user_id} Error: {e}")
            return None

async def main():
    start_global = time.perf_counter()
    
    tasks = [run_query(i) for i in range(MAX_CONCURRENT_USERS)]
    results = await asyncio.gather(*tasks)
    
    total_duration = time.perf_counter() - start_global
    valid = [r for r in results if r is not None]
    
    avg_latency = sum(valid) / len(valid) if valid else 0
    
    print("-" * 30)
    print(f"📊 ИТОГИ:")
    print(f"Всего времени: {total_duration:.2f} сек")
    print(f"Среднее время на запрос: {avg_latency:.2f} сек")
    print(f"Успешных запросов: {len(valid)}/{MAX_CONCURRENT_USERS}")
    if total_duration > 0:
        print(f"Производительность: {len(valid)/total_duration:.1f} req/sec")
    print("-" * 30)
    
    await engine.dispose()
if __name__ == "__main__":
    asyncio.run(main())