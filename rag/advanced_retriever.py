from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from config.settings import llm, embeddings, DB_PATH

# 1. Base Vector DB
vectorstore = Chroma(
    persist_directory=DB_PATH, 
    embedding_function=embeddings, 
    collection_name="my_db"
)

# 2. Base MMR Retriever
base_retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)

# 🚀 3. Advanced RAG: 멀티 쿼리 (Query Expansion) 적용
# 사용자의 질문을 다양한 관점에서 3개로 재작성하여 검색 정확도를 비약적으로 높입니다.
advanced_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)